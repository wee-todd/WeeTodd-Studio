import Foundation
public enum VoiceEngine: String, Codable, CaseIterable { case fishS2Pro, qwen3TTS }
public enum VoiceReferenceMode: String, Codable, CaseIterable {
  case audioAndTranscript, speakerIdentityOnly, synthetic, customVoice
}
public struct VoiceReference: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var assetID: UUID?
  public var path: String
  public var sourceHash: String = ""
  public var start: Double = 0
  public var duration: Double
  public var channel = "mix"
  public var transcript: String
  public init(assetID: UUID? = nil, path: String = "", start: Double = 0, duration: Double = 5, transcript: String = "") {
    self.assetID = assetID; self.path = path; self.start = start; self.duration = duration; self.transcript = transcript
  }
  public func validate(transcriptRequired: Bool) throws {
    guard !path.isEmpty, start.isFinite, start >= 0, duration.isFinite, duration > 0,
      duration <= 60, ["mix", "left", "right"].contains(channel),
      !transcriptRequired || !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw StudioError.invalid("Choose a voice sample up to 60 seconds and enter the words spoken in the selected range.")
    }
  }
}
public struct VoicePreset: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var name: String
  public var reference: VoiceReference
  public init(name: String, reference: VoiceReference) { self.name = name; self.reference = reference }
}
extension StudioProject {
  public mutating func mapVoicePaths(_ transform: (String) throws -> String) rethrows {
    try voiceDraft?.dialogue?.mapPaths(transform)
    if let reference = voiceDraft?.reference { voiceDraft?.reference?.path = try transform(reference.path) }
    if voicePresets != nil {
      for i in voicePresets!.indices { voicePresets![i].reference.path = try transform(voicePresets![i].reference.path) }
    }
    for i in assets.indices where assets[i].voiceGeneration != nil {
      try assets[i].voiceGeneration!.draft.dialogue?.mapPaths(transform)
      if assets[i].voiceGeneration!.request.turns != nil {
        for j in assets[i].voiceGeneration!.request.turns!.indices {
          if let ref = assets[i].voiceGeneration!.request.turns![j].reference {
            assets[i].voiceGeneration!.request.turns![j].reference?.path = try transform(ref.path)
          }
        }
      }
      assets[i].voiceGeneration!.artifacts = try transform(assets[i].voiceGeneration!.artifacts)
      if let reference = assets[i].voiceGeneration!.draft.reference { assets[i].voiceGeneration!.draft.reference?.path = try transform(reference.path) }
      if let reference = assets[i].voiceGeneration!.request.reference { assets[i].voiceGeneration!.request.reference?.path = try transform(reference.path) }
    }
  }
}
