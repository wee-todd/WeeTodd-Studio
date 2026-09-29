import Foundation

public struct H3PackedLayout: Sendable {
  public enum Anchor: Sendable, Equatable { case first, last, frame(Int) }

  public let positions: [SIMD3<Float>]
  public let tags: [Int32]
  public let conditionVideoRows: Int
  public let audioStart: Int
  public let videoStart: Int

  public init(geometry: H3Geometry, textTags: [Int32], anchors: [Anchor]) throws {
    guard !textTags.isEmpty, textTags.allSatisfy({ $0 == 0 || $0 == 1 }) else {
      throw H3GeometryError.invalid("H3 text rows must carry text or vision modality tags.")
    }
    let latentHeight = geometry.height / 16
    let latentWidth = geometry.width / 16
    let heightPatches = latentHeight / 2
    let widthPatches = latentWidth / 2
    let rowsPerFrame = heightPatches * widthPatches
    let (conditionRows, overflow) = anchors.count.multipliedReportingOverflow(by: rowsPerFrame)
    guard !overflow else { throw H3GeometryError.invalid("H3 keyframe rows overflow Int.") }
    let count = try geometry.packedRows(textRows: textTags.count,
      conditionVideoRows: conditionRows, conditionAudioRows: 0)
    conditionVideoRows = conditionRows
    audioStart = textTags.count + conditionRows
    videoStart = audioStart + geometry.audioRows

    let sqrtArea = sqrt(Double(latentHeight * latentWidth))
    func grid(_ size: Int) -> [Double] {
      let ratio = Double(size) / sqrtArea
      let left = (1 - ratio) / 2
      return (0..<(size / 2)).map { (left + Double($0) * ratio / Double(size / 2)) * 32 }
    }
    let heights = grid(latentHeight), widths = grid(latentWidth)
    let spatial: [SIMD3<Float>] = heights.flatMap { h in widths.map { w in SIMD3(0, Float(h), Float(w)) } }
    let rescale = 5.0 / 3.0
    let pattern = [1, 4, 4, 4, 4]
    let spans = (0..<geometry.videoLatentFrames).map { rescale * Double(pattern[$0 % pattern.count]) }
    let totalSpan = spans.reduce(0, +)
    let maxPixelFrame = Int((totalSpan / rescale).rounded(.toNearestOrEven)) - 1
    for anchor in anchors {
      if case .frame(let index) = anchor, !(0...maxPixelFrame).contains(index) {
        throw H3GeometryError.invalid("A timed H3 keyframe lies outside the generated timeline.")
      }
    }

    var locations = [SIMD3<Float>]()
    locations.reserveCapacity(count)
    for index in textTags.indices { locations.append(SIMD3(Float(index), 0, 0)) }
    for anchor in anchors {
      let time: Double
      switch anchor {
      case .first: time = Double(textTags.count)
      case .last: time = Double(textTags.count) + totalSpan - rescale
      case .frame(let index): time = Double(textTags.count) + rescale * Double(index)
      }
      for point in spatial { locations.append(SIMD3(Float(time), point.y, point.z)) }
    }
    for channel in 0..<2 {
      let width = channel == 0 ? widths[0] : widths[widths.count - 1]
      for index in 0..<geometry.audioLatentFrames {
        locations.append(SIMD3(Float(textTags.count + index), 0, Float(width)))
      }
    }
    var time = Double(textTags.count)
    for index in 0..<geometry.videoLatentFrames {
      for point in spatial { locations.append(SIMD3(Float(time), point.y, point.z)) }
      time += spans[index]
    }
    positions = locations
    tags = textTags + [Int32](repeating: 0, count: conditionRows)
      + [Int32](repeating: 2, count: geometry.audioRows)
      + [Int32](repeating: 0, count: geometry.videoRows)
  }
}
