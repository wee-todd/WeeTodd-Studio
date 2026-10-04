import Foundation

/// Metadata for a single encoded Ref2VA block. Pixels and latent tensors are
/// staged separately; this value is safe to validate before loading weights.
public enum H3ReferenceSpec: Sendable {
  case image(latentHeight: Int, latentWidth: Int, targetFrame: Int? = nil)
  case video(latentFrames: Int, latentHeight: Int, latentWidth: Int,
    audioLatents: Int, sourceLatentFrames: Int, targetFrame: Int? = nil)
  case audio(latents: Int, targetFrame: Int? = nil)

  fileprivate var targetFrame: Int? {
    switch self {
    case .image(_, _, let frame), .video(_, _, _, _, _, let frame), .audio(_, let frame): frame
    }
  }
}

/// Ref2VA's packed [text, ordered references, target audio, target video]
/// positions. The modality index lists specify the separate latent tensor
/// order and must not be inferred from contiguous row spans.
public struct H3ReferenceLayout: Sendable {
  public let maximumPackedRows: Int
  public let positions: [SIMD3<Float>]
  public let tags: [Int32]
  public let conditionVideoIndices: [Int]
  public let conditionAudioIndices: [Int]
  public let targetVideoIndices: [Int]
  public let targetAudioIndices: [Int]
  public let videoIndices: [Int]
  public let audioIndices: [Int]

  public init(geometry: H3Geometry, textTags: [Int32],
    references: [H3ReferenceSpec]) throws {
    func invalid(_ detail: String) -> H3GeometryError { .invalid(detail) }
    func checkedProduct(_ values: [Int]) throws -> Int {
      var result = 1
      for value in values {
        guard value >= 0 else { throw invalid("Negative H3 reference geometry.") }
        let (next, overflow) = result.multipliedReportingOverflow(by: value)
        guard !overflow else { throw invalid("H3 reference row count overflows Int.") }
        result = next
      }
      return result
    }
    func checkedSum(_ left: Int, _ right: Int) throws -> Int {
      let (result, overflow) = left.addingReportingOverflow(right)
      guard !overflow else { throw invalid("H3 reference row count overflows Int.") }
      return result
    }
    guard !textTags.isEmpty, textTags.allSatisfy({ $0 == 0 || $0 == 1 }),
      (1...12).contains(references.count) else {
      throw invalid("Ref2VA requires text and one to twelve references.")
    }
    let images = references.filter { if case .image = $0 { true } else { false } }.count
    let videos = references.filter { if case .video = $0 { true } else { false } }.count
    let audios = references.filter {
      switch $0 {
      case .audio: true
      case .video(_, _, _, let count, _, _): count > 0
      case .image: false
      }
    }.count
    guard images <= 9, videos <= 3, audios <= 3,
      images + videos > 0 || references.allSatisfy({ $0.targetFrame != nil }) else {
      throw invalid("Ref2VA reference modality counts or audio-only timing are unsupported.")
    }
    let targetHeight = geometry.height / 16
    let targetWidth = geometry.width / 16
    let targetGrid = Self.grid(height: targetHeight, width: targetWidth)
    var conditionVideoRows = 0
    var conditionAudioRows = 0
    for reference in references {
      if let frame = reference.targetFrame, !(0..<geometry.frames).contains(frame) {
        throw invalid("A timed Ref2VA guide lies outside the target clip.")
      }
      switch reference {
      case .image(let height, let width, _):
        guard height >= 2, width >= 2, height.isMultiple(of: 2), width.isMultiple(of: 2) else {
          throw invalid("Ref2VA image latent geometry must fit whole video patches.")
        }
        conditionVideoRows = try checkedSum(conditionVideoRows,
          checkedProduct([height / 2, width / 2]))
      case .video(let frames, let height, let width, let audio, let source, _):
        guard frames > 0, source >= frames, source <= 4_096,
          height >= 2, width >= 2,
          height.isMultiple(of: 2), width.isMultiple(of: 2), audio >= 0 else {
          throw invalid("Ref2VA video latent geometry or source duration is invalid.")
        }
        conditionVideoRows = try checkedSum(conditionVideoRows,
          checkedProduct([frames, height / 2, width / 2]))
        conditionAudioRows = try checkedSum(conditionAudioRows, checkedProduct([audio, 2]))
      case .audio(let latents, _):
        guard latents > 0 else { throw invalid("Ref2VA audio requires positive latent duration.") }
        conditionAudioRows = try checkedSum(conditionAudioRows, checkedProduct([latents, 2]))
      }
    }
    let count = try geometry.packedRows(textRows: textTags.count,
      conditionVideoRows: conditionVideoRows, conditionAudioRows: conditionAudioRows)
    guard count <= geometry.maximumPackedRows else { throw invalid("Ref2VA packed rows exceed the Swift H3 budget.") }

    let rescale = 5.0 / 3.0
    func span(_ frames: Int) -> Double {
      (0..<frames).reduce(0.0) { $0 + rescale * Double(($1 % 5 == 0) ? 1 : 4) }
    }
    func temporal(_ frame: Int, origin: Double) -> Double {
      origin + span(frame)
    }
    var targetOrigin = Double(textTags.count)
    for reference in references where reference.targetFrame == nil {
      switch reference {
      case .image: targetOrigin += 1
      case .audio(let latents, _): targetOrigin += Double(latents)
      case .video(let frames, _, _, let audio, let source, _):
        targetOrigin += max(Double(audio), span(max(frames, source)))
      }
    }

    var positions: [SIMD3<Float>] = []
    var tags: [Int32] = []
    var conditionVideoIndices: [Int] = []
    var conditionAudioIndices: [Int] = []
    positions.reserveCapacity(count)
    tags.reserveCapacity(count)
    for index in textTags.indices {
      positions.append(SIMD3(Float(index), 0, 0))
      tags.append(textTags[index])
    }
    func appendAudio(_ latents: Int, origin: Double, widths: [Double]) {
      guard latents > 0 else { return }
      for channel in 0..<2 {
        for index in 0..<latents {
          conditionAudioIndices.append(positions.count)
          positions.append(SIMD3(Float(origin + Double(index)), 0,
            Float(channel == 0 ? widths[0] : widths[widths.count - 1])))
          tags.append(2)
        }
      }
    }
    func appendVideo(_ frames: Int, height: Int, width: Int,
      origin: Double, sourceFrames: Int) {
      let grid = Self.grid(height: height, width: width)
      for frame in 0..<frames {
        let time: Double
        if sourceFrames > frames {
          let end = temporal(sourceFrames - 1, origin: origin)
          time = frames == 1 ? origin : origin + (end - origin) * Double(frame) / Double(frames - 1)
        } else {
          time = temporal(frame, origin: origin)
        }
        for point in grid {
          conditionVideoIndices.append(positions.count)
          positions.append(SIMD3(Float(time), point.y, point.z))
          tags.append(0)
        }
      }
    }
    var rotary = Double(textTags.count)
    for reference in references {
      let origin = reference.targetFrame.map { targetOrigin + rescale * Double($0) } ?? rotary
      switch reference {
      case .image(let height, let width, _):
        appendVideo(1, height: height, width: width, origin: origin, sourceFrames: 1)
        if reference.targetFrame == nil { rotary += 1 }
      case .audio(let latents, _):
        appendAudio(latents, origin: origin, widths: Self.widthGrid(height: targetHeight, width: targetWidth))
        if reference.targetFrame == nil { rotary += Double(latents) }
      case .video(let frames, let height, let width, let audio, let source, _):
        appendAudio(audio, origin: origin, widths: Self.widthGrid(height: height, width: width))
        appendVideo(frames, height: height, width: width, origin: origin, sourceFrames: source)
        if reference.targetFrame == nil { rotary += max(Double(audio), span(source)) }
      }
    }
    let targetAudioStart = positions.count
    let widths = Self.widthGrid(height: targetHeight, width: targetWidth)
    for channel in 0..<2 {
      for index in 0..<geometry.audioLatentFrames {
        positions.append(SIMD3(Float(targetOrigin + Double(index)), 0,
          Float(channel == 0 ? widths[0] : widths[widths.count - 1])))
        tags.append(2)
      }
    }
    let targetVideoStart = positions.count
    for frame in 0..<geometry.videoLatentFrames {
      let time = Float(temporal(frame, origin: targetOrigin))
      for point in targetGrid {
        positions.append(SIMD3(time, point.y, point.z))
        tags.append(0)
      }
    }
    guard positions.count == count, tags.count == count else {
      throw invalid("Ref2VA packed geometry disagrees with admitted rows.")
    }
    maximumPackedRows = geometry.maximumPackedRows
    self.positions = positions
    self.tags = tags
    self.conditionVideoIndices = conditionVideoIndices
    self.conditionAudioIndices = conditionAudioIndices
    self.targetAudioIndices = Array(targetAudioStart..<targetVideoStart)
    self.targetVideoIndices = Array(targetVideoStart..<count)
    self.videoIndices = conditionVideoIndices + self.targetVideoIndices
    self.audioIndices = conditionAudioIndices + self.targetAudioIndices
  }

  /// FL continuation has a distinct packed order: context video, anchors,
  /// context audio, target audio, target video. Modality tensors retain that order.
  public init(geometry: H3Geometry, textTags: [Int32],
    anchors: [H3PackedLayout.Anchor], contextFrames: Int) throws {
    guard H3Continuation.allowedContextFrames.contains(contextFrames),
      (1...8).contains(anchors.count) else {
      throw H3GeometryError.invalid("Invalid FL2VA continuation context or anchors.")
    }
    let ordinary = try H3PackedLayout(geometry: geometry, textTags: textTags, anchors: anchors)
    let contextVideoFrames = ((contextFrames - 5) / 17) * 5 + 2
    let contextAudioFrames = Int((Double(contextFrames) / 24 * 40).rounded(.toNearestOrEven))
    let context = try H3ReferenceLayout(geometry: geometry, textTags: textTags,
      references: [.video(latentFrames: contextVideoFrames,
        latentHeight: geometry.height / 16, latentWidth: geometry.width / 16,
        audioLatents: contextAudioFrames, sourceLatentFrames: contextVideoFrames,
        targetFrame: 0)])
    let locations = Array(ordinary.positions.prefix(textTags.count))
      + context.conditionVideoIndices.map { context.positions[$0] }
      + Array(ordinary.positions[textTags.count..<ordinary.audioStart])
      + context.conditionAudioIndices.map { context.positions[$0] }
      + Array(ordinary.positions[ordinary.audioStart..<ordinary.videoStart])
      + Array(ordinary.positions[ordinary.videoStart...])
    guard locations.count <= 40_000 else {
      throw H3GeometryError.invalid("FL2VA continuation packed rows exceed the Swift H3 budget.")
    }
    let videoEnd = textTags.count + context.conditionVideoIndices.count + ordinary.conditionVideoRows
    let conditionAudioEnd = videoEnd + context.conditionAudioIndices.count
    let targetAudioEnd = conditionAudioEnd + geometry.audioRows
    maximumPackedRows = 40_000
    positions = locations
    tags = textTags + [Int32](repeating: 0, count: videoEnd - textTags.count)
      + [Int32](repeating: 2, count: targetAudioEnd - videoEnd)
      + [Int32](repeating: 0, count: geometry.videoRows)
    conditionVideoIndices = Array(textTags.count..<videoEnd)
    conditionAudioIndices = Array(videoEnd..<conditionAudioEnd)
    targetAudioIndices = Array(conditionAudioEnd..<targetAudioEnd)
    targetVideoIndices = Array(targetAudioEnd..<locations.count)
    videoIndices = conditionVideoIndices + targetVideoIndices
    audioIndices = conditionAudioIndices + targetAudioIndices
  }

  /// Ref history shares the target rotary clock. Physical order remains
  /// text/references/context-audio/context-video/target; modality tensors place
  /// clean context first, followed by reference rows and target rows.
  public init(referenceLayout ordinary: H3ReferenceLayout,
    geometry: H3Geometry, contextFrames: Int) throws {
    guard H3Continuation.allowedContextFrames.contains(contextFrames) else {
      throw H3GeometryError.invalid("Unsupported Ref2VA context interval.")
    }
    let videoFrames = ((contextFrames - 5) / 17) * 5 + 2
    let audioFrames = Int((Double(contextFrames) / 24 * 40).rounded(.toNearestOrEven))
    let videoRows = videoFrames * (geometry.width / 32) * (geometry.height / 32)
    let audioRows = 2 * audioFrames
    guard ordinary.targetVideoIndices.count == geometry.videoRows,
      ordinary.targetAudioIndices.count == geometry.audioRows,
      let audioStart = ordinary.targetAudioIndices.first,
      let videoStart = ordinary.targetVideoIndices.first,
      audioStart + geometry.audioRows == videoStart,
      ordinary.tags.count == videoStart + geometry.videoRows,
      videoRows <= geometry.videoRows, audioFrames <= geometry.audioLatentFrames,
      ordinary.tags.count + videoRows + audioRows <= 40_000 else {
      throw H3GeometryError.invalid("Ref2VA context exceeds admitted synchronized rows.")
    }
    let contextAudio = Array(ordinary.positions[audioStart..<(audioStart + audioFrames)])
      + Array(ordinary.positions[(audioStart + geometry.audioLatentFrames)..<(audioStart + geometry.audioLatentFrames + audioFrames)])
    let contextVideo = Array(ordinary.positions[videoStart..<(videoStart + videoRows)])
    let extra = audioRows + videoRows
    maximumPackedRows = 40_000
    positions = Array(ordinary.positions[..<audioStart]) + contextAudio + contextVideo
      + Array(ordinary.positions[audioStart...])
    tags = Array(ordinary.tags[..<audioStart])
      + [Int32](repeating: 2, count: audioRows)
      + [Int32](repeating: 0, count: videoRows) + Array(ordinary.tags[audioStart...])
    conditionAudioIndices = Array(audioStart..<(audioStart + audioRows)) + ordinary.conditionAudioIndices
    conditionVideoIndices = Array((audioStart + audioRows)..<(audioStart + extra)) + ordinary.conditionVideoIndices
    targetAudioIndices = ordinary.targetAudioIndices.map { $0 + extra }
    targetVideoIndices = ordinary.targetVideoIndices.map { $0 + extra }
    audioIndices = conditionAudioIndices + targetAudioIndices
    videoIndices = conditionVideoIndices + targetVideoIndices
  }

  private static func widthGrid(height: Int, width: Int) -> [Double] {
    let root = sqrt(Double(height) * Double(width))
    let ratio = Double(width) / root
    let left = (1 - ratio) / 2
    return (0..<(width / 2)).map { (left + Double($0) * ratio / Double(width / 2)) * 32 }
  }

  private static func grid(height: Int, width: Int) -> [SIMD3<Float>] {
    let root = sqrt(Double(height) * Double(width))
    let ratio = Double(height) / root
    let left = (1 - ratio) / 2
    let heights = (0..<(height / 2)).map { (left + Double($0) * ratio / Double(height / 2)) * 32 }
    let widths = widthGrid(height: height, width: width)
    return heights.flatMap { h in widths.map { w in SIMD3(0, Float(h), Float(w)) } }
  }
}
