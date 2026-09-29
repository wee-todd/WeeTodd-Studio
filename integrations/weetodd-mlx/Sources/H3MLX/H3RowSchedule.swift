import Foundation

/// One AdaLN timetable shared by every H3 denoising step. Text follows video;
/// keyframe rows stay at their noise-augmentation level while target video and
/// audio advance on separate clocks.
public struct H3RowSchedule: Sendable {
  public let table: [Float]
  public let indicesByStep: [[Int32]]

  public init(layout: H3PackedLayout, video: H3Schedule, audio: H3Schedule,
    visualConditionStrength: Float = 0.999) throws {
    guard video.timesteps.count == audio.timesteps.count,
      !video.timesteps.isEmpty, visualConditionStrength.isFinite,
      (0...1).contains(visualConditionStrength) else {
      throw H3GeometryError.invalid("H3 audio/video schedules or visual reference strength differ from the admitted contract.")
    }
    let textRows = layout.audioStart - layout.conditionVideoRows
    guard textRows > 0, layout.videoStart >= layout.audioStart,
      layout.tags.count >= layout.videoStart else {
      throw H3GeometryError.invalid("Invalid H3 packed row spans.")
    }
    var used = Set<Float>()
    for index in video.timesteps.indices {
      used.insert(video.timesteps[index])
      used.insert(audio.timesteps[index])
      if layout.conditionVideoRows > 0 {
        used.insert(max(video.timesteps[index], visualConditionStrength))
      }
    }
    table = used.sorted()
    let lookup = Dictionary(uniqueKeysWithValues: table.enumerated().map { ($0.element, Int32($0.offset)) })
    var perStep: [[Int32]] = []
    perStep.reserveCapacity(video.timesteps.count)
    for index in video.timesteps.indices {
      let videoIndex = lookup[video.timesteps[index]]!
      let audioIndex = lookup[audio.timesteps[index]]!
      let referenceIndex = lookup[max(video.timesteps[index], visualConditionStrength)] ?? videoIndex
      var rows = [Int32](repeating: videoIndex, count: layout.tags.count)
      for row in textRows..<layout.audioStart { rows[row] = referenceIndex }
      for row in layout.audioStart..<layout.videoStart { rows[row] = audioIndex }
      perStep.append(rows)
    }
    indicesByStep = perStep
  }
}
