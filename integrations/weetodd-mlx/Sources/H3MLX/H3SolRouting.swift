import Foundation
import MLX

/// Independently expressed Sol equations with BF16 coarse K/V and Float32
/// pooling/routing. This is an experimental approximation, not DT FP16 parity.
/// No learned VSA gate/compression, global cache or full token mask is used.
enum H3SolRouting {
  struct Prepared {
    let geometry: H3SolGeometry
    let queryMeansF32: MLXArray
    let keyMeansBF16: MLXArray
    let valueMeansBF16: MLXArray
    /// Feature-interleaved [mean, diagonal variance], equal block weights.
    let keyStatsF32: MLXArray
    /// A set bit means the original 64-row key block must be consumed exactly.
    let exactRouteBits: MLXArray
    let exactCounts: MLXArray
    var storageBytes: Int {
      queryMeansF32.nbytes + keyMeansBF16.nbytes + valueMeansBF16.nbytes
        + keyStatsF32.nbytes + exactRouteBits.nbytes + exactCounts.nbytes
    }
  }

  struct Oracle: Sendable {
    let queryMeans: [Float]
    let keyMeans: [Float]
    let valueMeans: [Float]
    let keyStats: [Float]
    let exactRouteBits: [UInt32]
    let exactCounts: [UInt32]
  }

  static func select(scoreBase2: Float, thresholdBase2: Float, forced: Bool) -> Bool {
    forced || scoreBase2 > thresholdBase2
  }

  static func threshold(mu: Float, projectedVariance: Float, scale: Float, tau: Float) -> Float {
    let g = scale * Float(1.4426950408889634)
    return g * mu + tau * sqrt(g * g * projectedVariance + 1e-6)
  }

  static func roundBF16(_ value: Float) -> Float {
    let words = value.bitPattern
    if words & 0x7f80_0000 == 0x7f80_0000 { return value }
    let rounded = words &+ 0x7fff &+ ((words >> 16) & 1)
    return Float(bitPattern: rounded & 0xffff_0000)
  }

  /// Bounded independent CPU equation oracle in [H,T,D] row order. Input and
  /// coarse K/V are explicitly BF16-rounded; this is not a dense-bit oracle.
  static func cpuOracle(query: [Float], key: [Float], value: [Float],
    geometry g: H3SolGeometry) throws -> Oracle {
    let elements = g.heads * g.rows * g.dimension
    guard g.rows <= 1024, query.count == elements, key.count == elements, value.count == elements,
      query.allSatisfy({ $0.isFinite && abs($0) <= 16 }),
      key.allSatisfy({ $0.isFinite && abs($0) <= 16 }),
      value.allSatisfy({ $0.isFinite && abs($0) <= 1_000_000 }) else {
      throw H3CheckpointError.invalid("Invalid bounded CPU Sol oracle input.")
    }
    try Task.checkCancellation()
    let J = g.keyBlocks, D = g.dimension
    var qc = [Float](repeating: 0, count: g.heads * J * D)
    var kc = qc, vc = qc
    for h in 0..<g.heads {
      for j in 0..<J {
        for d in 0..<D {
          var q: Float = 0, k: Float = 0, v: Float = 0
          for r in (j * 64)..<(j * 64 + g.keyCount(j)) {
            let i = (h * g.rows + r) * D + d
            q += roundBF16(query[i]); k += roundBF16(key[i]); v += roundBF16(value[i])
          }
          let i = (h * J + j) * D + d, count = Float(g.keyCount(j))
          qc[i] = q / count; kc[i] = roundBF16(k / count); vc[i] = roundBF16(v / count)
        }
      }
      try Task.checkCancellation()
    }
    var stats = [Float](repeating: 0, count: g.heads * D * 2)
    for h in 0..<g.heads {
      for d in 0..<D {
        var m: Float = 0, second: Float = 0
        for j in 0..<J {
          let k = kc[(h * J + j) * D + d]
          m += k; second += k * k
        }
        m /= Float(J)
        stats[(h * D + d) * 2] = m
        // Metal contracts the final secondMoment - mean * mean. Explicit
        // addingProduct reproduces that single rounding without changing the
        // ordered block reductions or the independent pooling oracle.
        stats[(h * D + d) * 2 + 1] = max(0, (second / Float(J)).addingProduct(-m, m))
      }
    }
    var bits = [UInt32](repeating: 0, count: g.heads * J * g.routeWords)
    var counts = [UInt32](repeating: 0, count: g.heads * J)
    let gain = g.scale * Float(1.4426950408889634)
    for h in 0..<g.heads {
      for b in 0..<J {
        var mu: Float = 0, variance: Float = 0
        for d in 0..<D {
          let q = qc[(h * J + b) * D + d]
          mu += q * stats[(h * D + d) * 2]
          variance += q * q * stats[(h * D + d) * 2 + 1]
        }
        let limit = threshold(mu: mu, projectedVariance: variance, scale: g.scale, tau: g.tau)
        for j in 0..<J {
          var dot: Float = 0
          for d in 0..<D { dot += qc[(h * J + b) * D + d] * kc[(h * J + j) * D + d] }
          if select(scoreBase2: gain * dot, thresholdBase2: limit,
            forced: g.requiresExact(queryBlock: b, keyBlock: j)) {
            bits[(h * J + b) * g.routeWords + j / 32] |= UInt32(1) << (j % 32)
            counts[h * J + b] += 1
          }
        }
      }
      try Task.checkCancellation()
    }
    return Oracle(queryMeans: qc, keyMeans: kc, valueMeans: vc, keyStats: stats,
      exactRouteBits: bits, exactCounts: counts)
  }

  static func prepare(query: MLXArray, key: MLXArray, value: MLXArray,
    geometry g: H3SolGeometry) throws -> Prepared {
    guard Device.defaultDevice().deviceType == .gpu,
      [query, key, value].allSatisfy({ $0.shape == [1,g.heads,g.rows,128] && $0.dtype == .bfloat16 }) else {
      throw H3CheckpointError.invalid("Sol routing requires admitted GPU BF16 Q/K/V geometry.")
    }
    try Task.checkCancellation()
    eval([query, key, value]) // stride access is valid only after completion
    guard [query, key, value].allSatisfy({ $0.strides.last == 1 && $0.strides.allSatisfy { $0 >= 0 } }) else {
      throw H3CheckpointError.invalid("Sol routing requires contiguous features and nonnegative strides.")
    }
    let H = g.heads, J = g.keyBlocks
    // Complete element admission shares the existing pooling read. Only four
    // Float32 scalars per head/block escape; no full-sized Bool/abs tensors.
    var pooled = pool([query,key,value], template: [("HEADS",H),("ROWS",g.rows),("BLOCKS",J)],
      grid:(J*128,H,1), threadGroup:(128,1,1),
      outputShapes:[[1,H,J,128],[1,H,J,128],[1,H,J,128],[H,J,4]],
      outputDTypes:[.float32,.bfloat16,.bfloat16,.float32])
    eval(pooled)
    try Task.checkCancellation()
    try validatePooledAdmission(pooled[3].asArray(Float.self), heads: H, blocks: J)
    pooled.removeLast() // admission summary has no consumer dependency
    let stats = statistics([pooled[1]], template:[("HEADS",H),("BLOCKS",J)],
      grid:(128,H,1), threadGroup:(128,1,1),
      outputShapes:[[1,H,128,2]], outputDTypes:[.float32])[0]
    let parameters = MLXArray([g.scale,g.tau])
    let routes = route([pooled[0],pooled[1],stats,parameters],
      template:[("HEADS",H),("ROWS",g.rows),("BLOCKS",J),("WORDS",g.routeWords),
        ("BEGIN",g.approximationRange.lowerBound),("END",g.approximationRange.upperBound),("RADIUS",g.localRadius)],
      grid:(128,J,H), threadGroup:(128,1,1),
      outputShapes:[[1,H,J,g.routeWords],[1,H,J]], outputDTypes:[.uint32,.uint32])
    eval(routes)
    try Task.checkCancellation()
    return Prepared(geometry:g,queryMeansF32:pooled[0],keyMeansBF16:pooled[1],valueMeansBF16:pooled[2],
      keyStatsF32:stats,exactRouteBits:routes[0],exactCounts:routes[1])
  }

  /// Safe arithmetic bound, not a claim that RMS/rotary outputs stay within 16.
  /// At 1e6, serial 64 pooling sums are<=6.4e7, rounded BF16 key moments
  /// are<1.02e12, and 128-feature q^2*variance sums are<2e26.
  /// Allowed |scale|<=1 and |tau|<=16 keep threshold arithmetic far below
  /// Float32.max. No input is clamped or rescaled by this admission.
  static let maximumInputMagnitude: Float = 1_000_000

  /// Pure scalar contract, shared by the GPU handoff and GPU-free tests.
  /// Each tuple is [nonfinite flag, maxAbsQ, maxAbsK, maxAbsV].
  static func validatePooledAdmission(_ summary: [Float], heads: Int, blocks: Int) throws {
    guard (1...56).contains(heads), (1...625).contains(blocks),
      summary.count == heads * blocks * 4 else {
      throw H3CheckpointError.invalid("Invalid bounded Sol numeric admission summary.")
    }
    for offset in stride(from: 0, to: summary.count, by: 4) {
      guard summary[offset] == 0,
        summary[offset + 1].isFinite, (0...maximumInputMagnitude).contains(summary[offset + 1]),
        summary[offset + 2].isFinite, (0...maximumInputMagnitude).contains(summary[offset + 2]),
        summary[offset + 3].isFinite, (0...maximumInputMagnitude).contains(summary[offset + 3]) else {
        let tuple = offset / 4
        let head = tuple / blocks, block = tuple % blocks
        throw H3CheckpointError.invalid("Sol numeric admission failed at head \(head), key block \(block): nonfinite flag=\(summary[offset]), max|Q|=\(summary[offset+1]), max|K|=\(summary[offset+2]), max|V|=\(summary[offset+3]); limit=\(maximumInputMagnitude).")
      }
    }
  }

  private static let pool = MLXFast.metalKernel(name:"weetodd_h3_sol_bf16_pool_bounded_admission",
    inputNames:["q","k","v"], outputNames:["qc","kc","vc","admission"], source:"""
      uint d = thread_index_in_threadgroup;
      uint j = threadgroup_position_in_grid.x;
      uint h = threadgroup_position_in_grid.y;
      uint count = min(uint(64),uint(ROWS)-j*64);
      volatile float qs=0.0f,ks=0.0f,vs=0.0f;
      float bad=0.0f,max_q=0.0f,max_k=0.0f,max_v=0.0f;
      for (uint r=j*64;r<j*64+count;++r) {
        float qv=float(q[long(h)*q_strides[1]+long(r)*q_strides[2]+d]);
        float kv=float(k[long(h)*k_strides[1]+long(r)*k_strides[2]+d]);
        float vv=float(v[long(h)*v_strides[1]+long(r)*v_strides[2]+d]);
        qs=qs+qv;ks=ks+kv;vs=vs+vv;
        if (isfinite(qv)) max_q=max(max_q,abs(qv)); else bad=1.0f;
        if (isfinite(kv)) max_k=max(max_k,abs(kv)); else bad=1.0f;
        if (isfinite(vv)) max_v=max(max_v,abs(vv)); else bad=1.0f;
      }
      uint offset=(h*BLOCKS+j)*128+d;
      qc[offset]=float(qs)/float(count);
      kc[offset]=bfloat(float(ks)/float(count));
      vc[offset]=bfloat(float(vs)/float(count));
      // Four simdgroups cover all 128 features; every valid row was visited.
      // The reduction only combines nonnegative maxima and a 0/1 flag, so it
      // cannot alter the serial Float32 pooling arithmetic above.
      uint lane=thread_index_in_simdgroup;
      uint sg=simdgroup_index_in_threadgroup;
      float sg_bad=simd_max(bad),sg_q=simd_max(max_q);
      float sg_k=simd_max(max_k),sg_v=simd_max(max_v);
      threadgroup float partial[16];
      if(lane==0) {
        partial[sg*4]=sg_bad;partial[sg*4+1]=sg_q;
        partial[sg*4+2]=sg_k;partial[sg*4+3]=sg_v;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(d==0) {
        for(uint field=0;field<4;++field) {
          float result=0.0f;
          for(uint group=0;group<4;++group) result=max(result,partial[group*4+field]);
          admission[(h*BLOCKS+j)*4+field]=result;
        }
      }
      """,ensureRowContiguous:false)

  private static let statistics = MLXFast.metalKernel(name:"weetodd_h3_sol_key_statistics",
    inputNames:["kc"], outputNames:["stats"], source:"""
      uint d=thread_index_in_threadgroup;
      uint h=threadgroup_position_in_grid.y;
      volatile float sum=0.0f,second=0.0f;
      for (uint j=0;j<BLOCKS;++j) {
        float k=float(kc[(h*BLOCKS+j)*128+d]);
        float square=k*k;
        sum=sum+k;second=second+square;
      }
      float mean=float(sum)/float(BLOCKS);
      stats[(h*128+d)*2]=mean;
      stats[(h*128+d)*2+1]=max(0.0f,float(second)/float(BLOCKS)-mean*mean);
      """,ensureRowContiguous:true)

  private static let route = MLXFast.metalKernel(name:"weetodd_h3_sol_diagonal_routes",
    inputNames:["qc","kc","stats","parameters"], outputNames:["bits","counts"], source:"""
      uint tid=thread_index_in_threadgroup;
      uint lane=thread_index_in_simdgroup;
      uint sg=simdgroup_index_in_threadgroup;
      uint b=threadgroup_position_in_grid.y;
      uint h=threadgroup_position_in_grid.z;
      threadgroup uchar selected[BLOCKS];
      float mu=0.0f,variance=0.0f;
      for (uint d=lane;d<128;d+=32) {
        float q=qc[(h*BLOCKS+b)*128+d];
        mu+=q*stats[(h*128+d)*2];
        variance+=q*q*stats[(h*128+d)*2+1];
      }
      float gain=parameters[0]*1.4426950408889634f;
      float limit=gain*simd_sum(mu)+parameters[1]*sqrt(gain*gain*simd_sum(variance)+1e-6f);
      bool protected_query=b*64<BEGIN || min((b+1)*64,uint(ROWS))>END;
      for (uint j=sg;j<BLOCKS;j+=4) {
        float dot=0.0f;
        for (uint d=lane;d<128;d+=32) dot+=qc[(h*BLOCKS+b)*128+d]*float(kc[(h*BLOCKS+j)*128+d]);
        bool exact=protected_query || j*64<BEGIN || min((j+1)*64,uint(ROWS))>END
          || abs(int(b)-int(j))<=RADIUS || gain*simd_sum(dot)>limit;
        if (lane==0) selected[j]=exact ? 1 : 0;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid<WORDS) {
        uint result=0;
        for (uint bit=0;bit<32 && tid*32+bit<BLOCKS;++bit) result|=uint(selected[tid*32+bit])<<bit;
        bits[(h*BLOCKS+b)*WORDS+tid]=result;
      }
      if (tid==0) {
        uint count=0;for(uint j=0;j<BLOCKS;++j)count+=uint(selected[j]);
        counts[h*BLOCKS+b]=count;
      }
      """,ensureRowContiguous:true)
}
