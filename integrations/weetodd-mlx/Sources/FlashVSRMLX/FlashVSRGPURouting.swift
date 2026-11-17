import Foundation
import MLX

/// Offload serial draft dot products and exact thresholding. Host sqrt, division,
/// exp and denominator order are retained: GPU exp is not a host expf substitute.
/// All temporaries belong to one call; no model or global geometry state is held.
enum FlashVSRGPURouting {
  static let maximumPairs = 8_388_608
  static let maximumWorkspaceBytes: UInt64 = 512 * 1024 * 1024
  private static let traceEnabled = ProcessInfo.processInfo.environment["WEETODD_UPSCALER_TRACE"] == "1"
  private static func trace(_ message: String) {
    if traceEnabled { FileHandle.standardError.write(Data(("Flash routing " + message + "\n").utf8)) }
  }
  private static let dots = MLXFast.metalKernel(name: "weetodd_flash_route_dots_v1",
    inputNames: ["q", "k", "query_rows", "key_rows"], outputNames: ["scores"],
    source: """
      uint i = thread_position_in_grid.x;
      uint pairs = query_rows_shape[0], bh = i / pairs, pair = i % pairs;
      uint d = q_shape[3];
      uint qi = (bh * q_shape[2] + query_rows[pair]) * d;
      uint ki = (bh * k_shape[2] + key_rows[pair]) * d;
      float score = 0.0f;
      #pragma clang loop unroll(disable)
      for (uint c = 0; c < d; ++c) {
        float product = q[qi + c] * k[ki + c];
        score = score + product;
      }
      scores[i] = score;
      """, header: "#pragma METAL fp contract(off)\n")

  static func rawScores(query q: MLXArray, key k: MLXArray, candidates: [[Int]]) throws -> MLXArray {
    try Task.checkCancellation()
    guard q.ndim == 4, k.ndim == 4, q.shape.allSatisfy({ $0 > 0 }), k.shape.allSatisfy({ $0 > 0 }),
      q.shape[0..<2] == k.shape[0..<2], q.dim(3) == k.dim(3), (1...128).contains(q.dim(3)),
      q.dtype == .float32, k.dtype == .float32, q.size <= Int(UInt32.max), k.size <= Int(UInt32.max),
      candidates.count == q.dim(2), candidates.allSatisfy({ !$0.isEmpty && $0.allSatisfy { (0..<k.dim(2)).contains($0) } }) else {
      throw FlashVSRError.invalid("Invalid GPU draft routing tensors or candidates.")
    }
    let pairs = candidates.reduce(0) { $0 + $1.count }, bh = q.dim(0) * q.dim(1)
    guard pairs <= maximumPairs / bh else { throw FlashVSRError.invalid("GPU draft routing exceeds its workspace bound.") }
    let rows = candidates.enumerated().flatMap { row, keys in Array(repeating: Int32(row), count: keys.count) }
    let ids = candidates.flatMap { $0.map(Int32.init) }
    return dots([q, k, MLXArray(rows), MLXArray(ids)], grid: (bh * pairs, 1, 1),
      threadGroup: (256, 1, 1), outputShapes: [[bh, pairs]], outputDTypes: [.float32])[0]
  }

  static func select(probabilities p: MLXArray, candidates: [[Int]], capacity: Int) throws -> [[[Int]]] {
    try Task.checkCancellation()
    let pairs = candidates.reduce(0) { $0 + $1.count }
    guard p.ndim == 2, p.dim(0) > 0, p.dim(1) == pairs, pairs > 0, capacity > 0,
      p.dtype == .float32, candidates.allSatisfy({ !$0.isEmpty }),
      pairs <= maximumPairs / p.dim(0) else { throw FlashVSRError.invalid("Invalid GPU routing threshold geometry.") }
    // Equality must still rank: the old strict threshold drops tied minima.
    if pairs < capacity { return Array(repeating: candidates, count: p.dim(0)) }
    let threshold = sorted(p, axis: 1)[0..., pairs - capacity].expandedDimensions(axis: 1)
    let chosen = (p .> threshold).asArray(Bool.self)
    try Task.checkCancellation()
    var result: [[[Int]]] = []
    for bh in 0..<p.dim(0) {
      var offset = bh * pairs, rows: [[Int]] = []
      for keys in candidates {
        let selected = keys.enumerated().compactMap { chosen[offset + $0.offset] ? $0.element : nil }
        rows.append(selected.isEmpty ? [keys[0]] : selected)
        offset += keys.count
      }
      result.append(rows)
    }
    return result
  }

  static func route(q: MLXArray, k: MLXArray, rows: Int, columns: Int, localRange: Int, topK: Int) throws -> [[[[Int]]]] {
    try Task.checkCancellation()
    guard q.ndim == 5, k.ndim == 5, q.shape.allSatisfy({ $0 > 0 }), k.shape.allSatisfy({ $0 > 0 }),
      q.shape[0..<2] == k.shape[0..<2], q.dim(3) == k.dim(3), q.dim(4) == k.dim(4),
      rows > 0, columns > 0, rows <= 4096, columns <= 4096, topK >= 0, topK < Int.max else {
      throw FlashVSRError.invalid("Invalid FlashVSR GPU routing geometry.")
    }
    func fallback(_ reason: String) throws -> [[[[Int]]]] {
      trace("CPU fallback=\(reason)")
      return try FlashVSRAttention.routeReference(q: q, k: k, rows: rows, columns: columns, localRange: localRange, topK: topK)
    }
    guard Device.defaultDevice().deviceType == .gpu, GPU.deviceInfo().architecture == "applegpu_g15d",
      [DType.float32, .bfloat16].contains(q.dtype),
      q.dtype == k.dtype, (1...128).contains(q.dim(4)), q.dim(0) * q.dim(1) <= 64 else { return try fallback("device-or-layout") }
    let candidates = try FlashVSRBlockRouting.candidates(queryBlocks: q.dim(2), keyBlocks: k.dim(2),
      rows: rows, columns: columns, localRange: localRange)
    let bh = q.dim(0) * q.dim(1), pairs = candidates.reduce(0) { $0 + $1.count }
    guard pairs <= maximumPairs / bh else { return try fallback("pair-budget") }
    let qp = mean(q.asType(.float32), axis: 3), kp = mean(k.asType(.float32), axis: 3)
    guard qp.nbytes + kp.nbytes <= 64 * 1024 * 1024 else { return try fallback("pool-budget") }
    // These bounds exclude subnormal products/partial sums and overflow without
    // a readback of pooled vectors. Unsafe data retains the original path/error.
    func safe(_ x: MLXArray) -> MLXArray {
      let magnitude = abs(x)
      return all((magnitude .== 0) .|| ((magnitude .>= 1e-10) .&& (magnitude .<= 1e10)))
    }
    guard (safe(qp) .&& safe(kp)).item(Bool.self) else { return try fallback("numeric-range") }
    let raw = try rawScores(query: qp, key: kp, candidates: candidates)
    let capacity = min(topK + 1, q.dim(2) * k.dim(2))
    let flat: [[[Int]]]
    if pairs < capacity {
      // Finite raw dots / sqrt(d)>=1 give finite logits. exp(logit-max) is in
      // [0,1], at least one term is 1, and a physical row cannot overflow its sum.
      guard all(isFinite(raw)).item(Bool.self) else { return try fallback("nonfinite-dot") }
      trace("GPU all-candidates pairs=\(pairs) range=\(localRange)")
      flat = Array(repeating: candidates, count: bh)
    } else {
      trace("GPU ranked pairs=\(pairs) range=\(localRange)")
      let scores = raw.asArray(Float.self), divisor = sqrt(Float(q.dim(4)))
      var probabilities: [Float] = []; probabilities.reserveCapacity(scores.count)
      for head in 0..<bh {
        var offset = head * pairs
        for keys in candidates {
          try Task.checkCancellation()
          let row = scores[offset..<offset + keys.count].map { $0 / divisor }
          let peak = row.max()!, exponents = row.map { exp($0 - peak) }, total = exponents.reduce(0, +)
          for e in exponents {
            let p = e / total
            guard p.isFinite else { throw FlashVSRError.invalid("Nonfinite FlashVSR draft routing.") }
            probabilities.append(p)
          }
          offset += keys.count
        }
      }
      flat = try select(probabilities: MLXArray(probabilities, [bh, pairs]), candidates: candidates, capacity: capacity)
    }
    try Task.checkCancellation()
    return (0..<q.dim(0)).map { batch in Array(flat[batch * q.dim(1)..<(batch + 1) * q.dim(1)]) }
  }
}
