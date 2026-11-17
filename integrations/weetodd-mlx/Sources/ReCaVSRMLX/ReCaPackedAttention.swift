import Foundation
import MLX

/// Integer-only packing preserves each original SDPA call and reduction shape.
/// Spatial descriptors are reused; temporal masks and Q/K/V gathers are formed
/// together on the GPU. Parameter and stream state remain in their engine owners.
final class ReCaPackedTilePlan {
  private static let traceEnabled = ProcessInfo.processInfo.environment["WEETODD_UPSCALER_TRACE"] == "1"
  struct Geometry {
    let height: Int, width: Int, windowHeight: Int, windowWidth: Int, slidesAtEdge: Bool
    var span: Int { height * width }
    func bounds(_ p: Int, extent: Int, window: Int) -> Range<Int> {
      if slidesAtEdge {
        let n = min(extent, window), start = min(max(0, p - n / 2), extent - n)
        return start..<start + n
      }
      return max(0, p - window / 2)..<min(extent, p + window / 2)
    }
  }
  struct Tile {
    let id: Int, top: Int, left: Int, qr: Int, qc: Int, kt: Int, kl: Int, kr: Int, kc: Int
    var shapeKey: String { "\(qr),\(qc),\(kr),\(kc)" }
    var fields: [Int32] { [top, left, qr, qc, kt, kl, kr, kc].map(Int32.init) }
  }
  struct Group {
    let tiles: [Tile]
    fileprivate let metadata: MLXArray, owner: UUID
  }
  let geometry: Geometry
  let tiles: [Tile], groups: [Group]
  private let owner: UUID
  private var inverseByFrames: [Int: MLXArray] = [:]

  init(height: Int, width: Int, windowHeight: Int, windowWidth: Int,
    slidesAtEdge: Bool, batchSize: Int = 8) throws {
    guard [height, width, windowHeight, windowWidth].allSatisfy({ (1...1024).contains($0) }),
      (1...8).contains(batchSize) else { throw ReCaError.invalid("Unsupported packed ReCaVSR attention geometry.") }
    let token = UUID(); owner = token
    let g = Geometry(height: height, width: width, windowHeight: windowHeight,
      windowWidth: windowWidth, slidesAtEdge: slidesAtEdge)
    geometry = g
    var built: [Tile] = []
    for top in stride(from: 0, to: height, by: 4) {
      for left in stride(from: 0, to: width, by: 8) {
        let qr = min(4, height - top), qc = min(8, width - left)
        let ys = (top..<top + qr).map { g.bounds($0, extent: height, window: windowHeight) }
        let xs = (left..<left + qc).map { g.bounds($0, extent: width, window: windowWidth) }
        let y = ys.map(\.lowerBound).min()!..<ys.map(\.upperBound).max()!
        let x = xs.map(\.lowerBound).min()!..<xs.map(\.upperBound).max()!
        built.append(Tile(id: built.count, top: top, left: left, qr: qr, qc: qc,
          kt: y.lowerBound, kl: x.lowerBound, kr: y.count, kc: x.count))
      }
    }
    tiles = built
    var buckets: [String: [Tile]] = [:], keys: [String] = []
    for t in built {
      if buckets[t.shapeKey] == nil { keys.append(t.shapeKey) }
      buckets[t.shapeKey, default: []].append(t)
    }
    var batches: [Group] = []
    for key in keys {
      let list = buckets[key]!
      for first in stride(from: 0, to: list.count, by: batchSize) {
        let selected = Array(list[first..<min(first + batchSize, list.count)])
        batches.append(Group(tiles: selected,
          metadata: MLXArray(selected.flatMap(\.fields), [selected.count, 8]), owner: token))
      }
    }
    groups = batches
  }

  private func validate(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) throws -> (Int, Int) {
    let span = geometry.span
    guard q.ndim == 3, k.ndim == 3, k.shape == v.shape,
      q.shape.allSatisfy({ $0 > 0 }), k.shape.allSatisfy({ $0 > 0 }),
      q.dim(0) % span == 0, k.dim(0) % span == 0, k.dim(0) >= q.dim(0),
      q.dim(2) == k.dim(2), q.dim(1) % k.dim(1) == 0,
      [DType.bfloat16, .float32].contains(q.dtype), [64, 128].contains(q.dim(2)),
      q.size <= Int(UInt32.max), k.size <= Int(UInt32.max), v.size <= Int(UInt32.max) else {
      throw ReCaError.invalid("Invalid packed ReCaVSR attention tensors.")
    }
    let qf = q.dim(0) / span, kf = k.dim(0) / span
    guard (1...6).contains(qf), (qf...8).contains(kf) else {
      throw ReCaError.invalid("Invalid packed ReCaVSR attention plane count.")
    }
    return (qf, kf)
  }

 private static let kernel=MLXFast.metalKernel(name:"weetodd_reca_packed_tiles_v1",inputNames:["q","k","v","meta","cfg"],outputNames:["pq","pk","pv","pm"],source:"""
    uint i = thread_position_in_grid.x;
    int H = cfg[0], W = cfg[1], qf = cfg[2], kf = cfg[3];
    int qr = cfg[4], qc = cfg[5], kr = cfg[6], kc = cfg[7];
    int WH = cfg[8], WW = cfg[9], slide = cfg[10];
    int qh = q_shape[1], kh = k_shape[1], D = q_shape[2];
    uint groups = meta_shape[0];
    uint nq = qf * qr * qc, nk = kf * kr * kc;
    uint qGroup = nq * qh * D, kGroup = nk * kh * D, maskGroup = nq * nk;
    if (i < groups * qGroup) {
      uint b = i / qGroup, e = i % qGroup;
      uint c = e % D, head = (e / D) % qh, token = e / (qh * D);
      uint plane = token / (qr * qc), pixel = token % (qr * qc);
      int y = meta[b * 8] + pixel / qc, x = meta[b * 8 + 1] + pixel % qc;
      uint source = ((plane * H * W + y * W + x) * qh + head) * D + c;
      pq[i] = q[source];
    }
    if (i < groups * kGroup) {
      uint b = i / kGroup, e = i % kGroup;
      uint c = e % D, head = (e / D) % kh, token = e / (kh * D);
      uint plane = token / (kr * kc), pixel = token % (kr * kc);
      int y = meta[b * 8 + 4] + pixel / kc, x = meta[b * 8 + 5] + pixel % kc;
      uint source = ((plane * H * W + y * W + x) * kh + head) * D + c;
      pk[i] = k[source];
      pv[i] = v[source];
    }
    if (i < groups * maskGroup) {
      uint b = i / maskGroup, e = i % maskGroup;
      uint qpixel = (e / nk) % (qr * qc), kpixel = (e % nk) % (kr * kc);
      int qy = meta[b * 8] + qpixel / qc, qx = meta[b * 8 + 1] + qpixel % qc;
      int ky = meta[b * 8 + 4] + kpixel / kc, kx = meta[b * 8 + 5] + kpixel % kc;
      int yh, yl, xh, xl;
      if (slide != 0) {
        int nh = min(H, WH), nw = min(W, WW);
        yl = min(max(0, qy - nh / 2), H - nh); yh = yl + nh;
        xl = min(max(0, qx - nw / 2), W - nw); xh = xl + nw;
      } else {
        yl = max(0, qy - WH / 2); yh = min(H, qy + WH / 2);
        xl = max(0, qx - WW / 2); xh = min(W, qx + WW / 2);
      }
      pm[i] = ky >= yl && ky < yh && kx >= xl && kx < xh;
    }
    """)

  private struct Packing {
    let inputs: [MLXArray], shapes: [[Int]], dtype: DType, count: Int
    var signature: String { "\(shapes)-\(dtype)" }
  }
  private func packing(_ group: Group, query q: MLXArray, key k: MLXArray, value v: MLXArray) throws -> Packing {
    try Task.checkCancellation()
    let (qf, kf) = try validate(q, k, v)
    guard group.owner == owner, let t = group.tiles.first else {
      throw ReCaError.invalid("Packed ReCaVSR attention group belongs to another plan.")
    }
    let g = geometry, n = group.tiles.count, nq = qf * t.qr * t.qc, nk = kf * t.kr * t.kc
    let qs = [n, nq, q.dim(1), q.dim(2)], ks = [n, nk, k.dim(1), k.dim(2)], ms = [n, nq, nk]
    let count = max(qs.reduce(1, *), max(ks.reduce(1, *), ms.reduce(1, *)))
    guard count <= Int(Int32.max) else { throw ReCaError.invalid("Packed ReCaVSR attention exceeds its Metal grid bound.") }
    let cfg = MLXArray([g.height, g.width, qf, kf, t.qr, t.qc, t.kr, t.kc,
      g.windowHeight, g.windowWidth, g.slidesAtEdge ? 1 : 0].map(Int32.init))
    return Packing(inputs: [q,k.asType(q.dtype),v.asType(q.dtype),group.metadata,cfg],
      shapes: [qs,ks,ks,ms], dtype: q.dtype, count: count)
  }

  private static func pack(_ p: Packing, _ inputs: [MLXArray]) -> [MLXArray] {
    kernel(inputs, grid: (p.count,1,1), threadGroup: (256,1,1), outputShapes: p.shapes,
      outputDTypes: [p.dtype,p.dtype,p.dtype,.bool])
  }
  func pack(_ group: Group, query q: MLXArray, key k: MLXArray, value v: MLXArray) throws -> [MLXArray] {
    let p = try packing(group,query:q,key:k,value:v)
    return Self.pack(p,p.inputs)
  }

  private func inverse(frames: Int) -> MLXArray {
    if let cached = inverseByFrames[frames] { return cached }
    var inverse = Array(repeating: Int32(0), count: frames * geometry.span), position = 0
    for tile in tiles { for t in 0..<frames {
      for y in tile.top..<tile.top + tile.qr { for x in tile.left..<tile.left + tile.qc {
        inverse[t * geometry.span + y * geometry.width + x] = Int32(position); position += 1
      } }
    } }
    let result = MLXArray(inverse); inverseByFrames[frames] = result
    return result
  }

  func apply(query q: MLXArray, key k: MLXArray, value v: MLXArray,
    maximumWorkspaceBytes: UInt64? = nil, compiled: Bool = false,
    asynchronous: Bool = false,
    checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> MLXArray {
    try checkCancellation()
    let (qf, kf) = try validate(q, k, v), scalarBytes = q.dtype == .float32 ? 4 : 2
    let transient = groups.map { group -> UInt64 in
      let t = group.tiles[0], nq = qf * t.qr * t.qc, nk = kf * t.kr * t.kc
      let qBytes = nq * q.dim(1) * q.dim(2) * scalarBytes
      let kvBytes = nk * k.dim(1) * k.dim(2) * scalarBytes * 2
      // Account for overlapping completion lifetimes of two bounded submissions.
      return UInt64(group.tiles.count) * UInt64(qBytes * 2 + kvBytes + nq * nk) * 2
    }.max() ?? 0
    guard transient <= (maximumWorkspaceBytes ?? UInt64(max(0, Memory.memoryLimit))) else {
      throw ReCaError.invalid("Packed ReCaVSR attention exceeds its workspace budget.")
    }
    if Self.traceEnabled {
      let message = "ReCa attention compiled=\(compiled) async-depth=\(asynchronous ? 2:1) workspace=\(transient) groups=\(groups.count)\n"
      FileHandle.standardError.write(Data(message.utf8))
    }
    var parts = Array<MLXArray?>(repeating: nil, count: tiles.count)
    var functions: [String: ([MLXArray]) -> [MLXArray]] = [:]
    var pending: [[MLXArray]] = []
    // All independent submissions must finish before a stage can be unloaded.
    defer { for batch in pending { eval(batch) } }
    for group in groups {
      try checkCancellation()
      let p = try packing(group,query:q,key:k,value:v)
      let result: [MLXArray]
      if compiled, functions[p.signature] != nil || functions.count < 64 {
        if functions[p.signature] == nil {
          // Capture only geometry. Tensor inputs, metadata and configuration are
          // explicit, so compile-cache ownership cannot retain model/stream state.
          let layout = Packing(inputs: [], shapes: p.shapes, dtype: p.dtype, count: p.count)
          let n = group.tiles.count
          functions[p.signature] = compile { (a: [MLXArray]) in
            let packed = Self.pack(layout,a)
            return (0..<n).map {
              ReCaAttentionPlan.attend(packed[0][$0],packed[1][$0],packed[2][$0],mask:packed[3][$0])
            }
          }
        }
        result = functions[p.signature]!(p.inputs)
      } else {
        let packed = Self.pack(p,p.inputs)
        result = group.tiles.indices.map {
          ReCaAttentionPlan.attend(packed[0][$0],packed[1][$0],packed[2][$0],mask:packed[3][$0])
        }
      }
      if asynchronous {
        asyncEval(result); pending.append(result)
        if pending.count == 2 { eval(pending.removeFirst()) }
      } else { eval(result) }
      for (i, tile) in group.tiles.enumerated() { parts[tile.id] = result[i] }
    }
    for batch in pending { eval(batch) }; pending.removeAll()
    try checkCancellation()
    return concatenated(parts.map { $0! }, axis: 0).take(inverse(frames: qf), axis: 0)
  }

  /// No Metal or model allocation occurs during admission.
  static func admissionBytes(height h: Int, width w: Int, windowHeight wh: Int,
    windowWidth ww: Int, sliding: Bool, queryFrames qf: Int, keyFrames kf: Int,
    queryChannels: Int, keyChannels: Int, scalarBytes: Int) -> UInt64 {
    let whole = sliding ? h <= wh && w <= ww : h <= wh / 2 && w <= ww / 2
    if whole || h > 1024 || w > 1024 { return 0 }
    let tiles = (h + 3) / 4 * ((w + 7) / 8), batch = min(8, tiles)
    let nq = qf * min(4, h) * min(8, w), nk = kf * min(h, wh + 3) * min(w, ww + 7)
    let bytes = (nq * queryChannels * scalarBytes * 2 + nk * keyChannels * scalarBytes * 2 + nq * nk) * batch * 2
    return UInt64(bytes + tiles * 1024 + h * w * qf * 16)
  }
}
