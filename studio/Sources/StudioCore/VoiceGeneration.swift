import Foundation
public struct VoiceSampling: Codable, Equatable {
  public var temperature: Double = 0.9
  public var topP: Double = 1
  public var topK: Int = 50
  public var maxTokens: Int = 512
  public init() {}
  enum CodingKeys: String, CodingKey { case temperature, topP = "top_p", topK = "top_k", maxTokens = "max_tokens" }
}
public struct VoiceRequest: Codable, Equatable {
  public var engine: VoiceEngine
  public var modelPath: String
  public var precision: String
  public var text: String
  public var referenceMode: VoiceReferenceMode
  public var reference: VoiceReference?
  public var language: String
  public var seed: Int
  public var sampling: VoiceSampling
  public var turns: [VoiceTurnRequest]? = nil
  public var speaker: String? = nil
  public var instruct: String? = nil
  public var voiceDirection: [String]? = nil
  enum CodingKeys: String, CodingKey {
    case engine, modelPath = "model_path", precision, text, referenceMode = "reference_mode"
    case reference, language, seed, sampling, turns, speaker, instruct
    case voiceDirection = "voice_direction"
  }
}
public struct VoiceDraft: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var name = "Voice take"
  public var engine: VoiceEngine = .qwen3TTS
  public var modelID: UUID?
  // Legacy draft location. New drafts resolve modelID through Runtime settings.
  public var modelPath = ""
  public var precision = "auto"
  public var text = ""
  public var referenceMode: VoiceReferenceMode = .audioAndTranscript
  public var reference: VoiceReference?
  public var language = "auto"
  public var seed = 42
  public var sampling = VoiceSampling()
  public var usesDialogue: Bool?
  public var dialogue: VoiceDialogue?
  public var presetVoice: String?
  public var instructions: String?
  public var fishDirection: FishVoiceDirection?
  public init() {}
  public func request() throws -> VoiceRequest {
    if usesDialogue == true { return try dialogueRequest() }
    let direction = engine == .fishS2Pro ? try fishDirection?.tags() ?? [] : []
    let directionBytes = direction.map { "[\($0)] " }.joined().utf8.count
    if engine == .qwen3TTS && FishVoiceTags.containsTags(text) {
      throw StudioError.invalid("Qwen does not support inline delivery tags. Remove bracket tags; use a Base performance sample or CustomVoice delivery instructions.")
    }
    guard !modelPath.isEmpty, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      text.utf8.count + directionBytes <= 32000, seed >= 0, seed < 4_294_967_296, ["auto", "bf16", "8bit"].contains(precision),
      sampling.temperature.isFinite, (0...2).contains(sampling.temperature),
      sampling.topP.isFinite, sampling.topP > 0, sampling.topP <= 1,
      (1...1000).contains(sampling.topK), (1...4096).contains(sampling.maxTokens) else {
      throw StudioError.invalid("Choose a speech model, enter a script, and check sampling settings.")
    }
    guard !(engine == .fishS2Pro && referenceMode == .speakerIdentityOnly),
      !(engine == .qwen3TTS && referenceMode == .synthetic) else {
      throw StudioError.invalid("This reference mode is not supported by the selected speech engine.")
    }
    if referenceMode != .synthetic && referenceMode != .customVoice {
      guard let reference else { throw StudioError.invalid("Add a voice reference before generating speech.") }
      try reference.validate(transcriptRequired: referenceMode == .audioAndTranscript)
    }
    if referenceMode == .customVoice {
      guard engine == .qwen3TTS, QwenVoiceStyle.speakers.contains(presetVoice ?? "ryan"),
        (instructions ?? "").unicodeScalars.count <= 2000, !(instructions ?? "").contains("<|") else {
        throw StudioError.invalid("Choose a Qwen preset speaker and a delivery instruction up to 2,000 characters.")
      }
    }
    return VoiceRequest(engine: engine, modelPath: modelPath, precision: precision, text: text,
      referenceMode: referenceMode, reference: [.synthetic, .customVoice].contains(referenceMode) ? nil : reference,
      language: language, seed: seed, sampling: sampling,
      speaker: referenceMode == .customVoice ? (presetVoice ?? "ryan") : nil,
      instruct: referenceMode == .customVoice ? (instructions ?? "") : nil,
      voiceDirection: direction.isEmpty ? nil : direction)
  }
}
public struct VoiceGeneration: Codable, Equatable {
  public var draft: VoiceDraft
  public var request: VoiceRequest
  public var artifacts: String
  public var sampleRate: Int
  public var frames: Int
  public var truncated: Bool
  public var created = Date()
  public init(draft: VoiceDraft, request: VoiceRequest, artifacts: String,
              sampleRate: Int, frames: Int, truncated: Bool) {
    self.draft = draft; self.request = request; self.artifacts = artifacts
    self.sampleRate = sampleRate; self.frames = frames; self.truncated = truncated
  }
}
extension StudioProject {
  @discardableResult public mutating func placeVoice(_ asset: MediaAsset, clipID: UUID,
                                                     offset: Double = 0) throws -> AudioRegion {
    guard let clip = clips.first(where: { $0.id == clipID }), asset.kind == .audio,
      !asset.path.isEmpty, asset.duration.isFinite, asset.duration > 0,
      offset.isFinite, offset >= 0, offset < clip.duration else {
      throw StudioError.invalid("Select an available voice take and a position inside a clip.")
    }
    let track: AudioTrack
    if let found = audioTracks.first(where: { $0.role == .voice && !$0.replacesSource }) { track = found }
    else { track = AudioTrack(name: "Voice", role: .voice); audioTracks.append(track) }
    var region = AudioRegion(assetID: asset.id, path: asset.path)
    region.trackID = track.id; region.duration = min(asset.duration, clip.duration - offset)
    region.volume = 1; region.fade = 0.01
    region.anchor = ClipAudioAnchor(clipID: clipID, offsetSeconds: offset)
    region.start = try resolvedAudioStart(region, in: self)
    audio.append(region)
    return region
  }
}
