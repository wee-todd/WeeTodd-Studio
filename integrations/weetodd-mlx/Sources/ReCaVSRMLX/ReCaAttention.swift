import Foundation
import MLX

/// Bounded spatial tiles. All current planes and the explicitly routed past planes
/// are visible; temporal causality is a block/router decision, never a token mask.
final class ReCaAttentionPlan {
  let height: Int, width: Int, windowHeight: Int, windowWidth: Int
  let slidesAtEdge: Bool
  private let scheduling: ReCaAttentionScheduling
  private(set) var packed: ReCaPackedTilePlan?

  init(height: Int, width: Int, windowHeight: Int, windowWidth: Int, slidesAtEdge: Bool,
    scheduling: ReCaAttentionScheduling = .automatic) {
    self.height = height; self.width = width
    self.windowHeight = windowHeight; self.windowWidth = windowWidth
    self.slidesAtEdge = slidesAtEdge; self.scheduling = scheduling
  }
  func reset() { packed = nil }

  private func bounds(_ position: Int, extent: Int, window: Int) -> Range<Int> {
    if slidesAtEdge {
      let count = min(extent, window), start = min(max(0, position - count / 2), extent - count)
      return start..<(start + count)
    }
    return max(0, position - window / 2)..<min(extent, position + window / 2)
  }
  static func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    MLXFast.scaledDotProductAttention(queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
      keys: k.asType(q.dtype).transposed(1, 0, 2).expandedDimensions(axis: 0),
      values: v.asType(q.dtype).transposed(1, 0, 2).expandedDimensions(axis: 0),
      scale: 1 / sqrt(Float(q.dim(-1))), mask: mask).squeezed(axis: 0).transposed(1, 0, 2)
  }
  func apply(query: MLXArray, key: MLXArray, value: MLXArray) throws -> MLXArray {
    let span = height * width
    guard span > 0, query.ndim == 3, key.ndim == 3, key.shape == value.shape,
      query.shape.allSatisfy({ $0 > 0 }), key.shape.allSatisfy({ $0 > 0 }),
      query.dim(0) % span == 0, key.dim(0) % span == 0, key.dim(0) >= query.dim(0),
      query.dim(2) == key.dim(2), query.dim(1) % key.dim(1) == 0 else {
      throw ReCaError.invalid("Invalid ReCaVSR attention grid or GQA dimensions.")
    }
    if [0, height - 1].allSatisfy({ bounds($0, extent: height, window: windowHeight).count == height }) &&
      [0, width - 1].allSatisfy({ bounds($0, extent: width, window: windowWidth).count == width }) {
      try Task.checkCancellation(); return Self.attend(query, key, value)
    }
    let qframes = query.dim(0) / span, kframes = key.dim(0) / span
    if Device.defaultDevice().deviceType == .gpu,
      [DType.float32, .bfloat16].contains(query.dtype), [64, 128].contains(query.dim(2)),
      [height, width, windowHeight, windowWidth].allSatisfy({ (1...1024).contains($0) }),
      (1...6).contains(qframes), (qframes...8).contains(kframes),
      query.size <= Int(UInt32.max), key.size <= Int(UInt32.max), value.size <= Int(UInt32.max) {
      if packed == nil {
        packed = try ReCaPackedTilePlan(height: height, width: width,
          windowHeight: windowHeight, windowWidth: windowWidth, slidesAtEdge: slidesAtEdge)
      }
      return try packed!.apply(query: query, key: key, value: value,
        asynchronous: scheduling.usesAsync(architecture:GPU.deviceInfo().architecture))
    }
    var parts: [MLXArray] = [], order: [Int] = []
    for top in stride(from: 0, to: height, by: 4) { for left in stride(from: 0, to: width, by: 8) {
      try Task.checkCancellation()
      let spatial = (top..<min(height, top + 4)).flatMap { y in (left..<min(width, left + 8)).map { y * width + $0 } }
      let ys = spatial.map { bounds($0 / width, extent: height, window: windowHeight) }
      let xs = spatial.map { bounds($0 % width, extent: width, window: windowWidth) }
      let rows = ys.map(\.lowerBound).min()!..<ys.map(\.upperBound).max()!
      let columns = xs.map(\.lowerBound).min()!..<xs.map(\.upperBound).max()!
      let localKeys = rows.flatMap { y in columns.map { y * width + $0 } }
      let qi = (0..<qframes).flatMap { t in spatial.map { t * span + $0 } }
      let ki = (0..<kframes).flatMap { t in localKeys.map { t * span + $0 } }
      var mask: [Bool] = []; mask.reserveCapacity(qi.count * ki.count)
      for _ in 0..<qframes { for i in spatial.indices { for _ in 0..<kframes {
        for k in localKeys { mask.append(ys[i].contains(k / width) && xs[i].contains(k % width)) }
      } } }
      let output = Self.attend(query.take(MLXArray(qi.map(Int32.init)), axis: 0),
        key.take(MLXArray(ki.map(Int32.init)), axis: 0), value.take(MLXArray(ki.map(Int32.init)), axis: 0),
        mask: MLXArray(mask, [qi.count, ki.count]))
      eval(output); parts.append(output); order += qi
    } }
    var inverse = Array(repeating: Int32(0), count: order.count)
    for (i, position) in order.enumerated() { inverse[position] = Int32(i) }
    return concatenated(parts, axis: 0).take(MLXArray(inverse), axis: 0)
  }
}
