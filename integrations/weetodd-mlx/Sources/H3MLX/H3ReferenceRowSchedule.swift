import Foundation

/// Ref2VA modulation plan for interleaved reference rows. Reference video and
/// audio keep their own noise levels while target AV follows separate clocks.
public struct H3ReferenceRowSchedule: Sendable {
  public let table: [Float]
  public let indicesByStep: [[Int32]]

  public init(layout: H3ReferenceLayout, video: H3Schedule, audio: H3Schedule,
    visualConditionStrength: Float = 0.999,
    audioConditionStrength: Float = 1, cleanVideoPrefixRows: Int = 0,
    cleanAudioPrefixRows: Int = 0) throws {
    guard (0...layout.conditionVideoIndices.count).contains(cleanVideoPrefixRows),
      (0...layout.conditionAudioIndices.count).contains(cleanAudioPrefixRows),
      video.timesteps.count == audio.timesteps.count,
      !video.timesteps.isEmpty,
      visualConditionStrength.isFinite, (0...1).contains(visualConditionStrength),
      audioConditionStrength.isFinite, (0...1).contains(audioConditionStrength) else {
      throw H3GeometryError.invalid("Invalid Ref2VA schedules or reference strengths.")
    }
    var used = Set<Float>()
    for step in video.timesteps.indices {
      used.insert(video.timesteps[step])
      used.insert(audio.timesteps[step])
      if !layout.conditionVideoIndices.isEmpty {
        used.insert(max(video.timesteps[step], visualConditionStrength))
      }
      if !layout.conditionAudioIndices.isEmpty {
        used.insert(max(audio.timesteps[step], audioConditionStrength))
      }
    }
    if cleanVideoPrefixRows > 0 || cleanAudioPrefixRows > 0 { used.insert(1) }
    let sorted = used.sorted()
    guard (1...128).contains(sorted.count) else {
      throw H3GeometryError.invalid("Ref2VA timestep table exceeds the H3 budget.")
    }
    let lookup = Dictionary(uniqueKeysWithValues: sorted.enumerated().map {
      ($0.element, Int32($0.offset))
    })
    var rowsByStep: [[Int32]] = []
    rowsByStep.reserveCapacity(video.timesteps.count)
    for step in video.timesteps.indices {
      let videoTime = video.timesteps[step]
      let audioTime = audio.timesteps[step]
      guard let videoIndex = lookup[videoTime], let audioIndex = lookup[audioTime] else {
        throw H3GeometryError.invalid("Ref2VA target timestep was not admitted.")
      }
      var rows = [Int32](repeating: videoIndex, count: layout.tags.count)
      if !layout.conditionVideoIndices.isEmpty {
        guard let condition = lookup[max(videoTime, visualConditionStrength)] else {
          throw H3GeometryError.invalid("Ref2VA visual condition timestep was not admitted.")
        }
        for row in layout.conditionVideoIndices { rows[row] = condition }
      }
      if !layout.conditionAudioIndices.isEmpty {
        guard let condition = lookup[max(audioTime, audioConditionStrength)] else {
          throw H3GeometryError.invalid("Ref2VA audio condition timestep was not admitted.")
        }
        for row in layout.conditionAudioIndices { rows[row] = condition }
      }
      if cleanVideoPrefixRows > 0 {
        guard let clean = lookup[1] else { throw H3GeometryError.invalid("Missing clean context timestep.") }
        for row in layout.conditionVideoIndices.prefix(cleanVideoPrefixRows) { rows[row] = clean }
      }
      if cleanAudioPrefixRows > 0 {
        guard let clean = lookup[1] else { throw H3GeometryError.invalid("Missing clean audio context timestep.") }
        for row in layout.conditionAudioIndices.prefix(cleanAudioPrefixRows) { rows[row] = clean }
      }
      for row in layout.targetAudioIndices { rows[row] = audioIndex }
      rowsByStep.append(rows)
    }
    table = sorted
    indicesByStep = rowsByStep
  }
}
