import Foundation

public struct MusicSampling: Codable, Equatable {
  public var temperature: Double = 1
  public var topP: Double = 0.95
  public var topK: Int = 100
  public var repetitionPenalty: Double = 1.2
  public var penaltyWindow: Int = 50
  public var minTokens: Int = 200
  public var maxTokens: Int = 9000
  public init() {}
  public static var score: Self {
    var value = Self()
    value.temperature = 0.7; value.topP = 0.9; value.topK = 30
    value.repetitionPenalty = 1.005; value.penaltyWindow = 100
    value.minTokens = 32; value.maxTokens = 4096
    return value
  }
  enum CodingKeys: String, CodingKey {
    case temperature, topP = "top_p", topK = "top_k", repetitionPenalty = "repetition_penalty"
    case penaltyWindow = "penalty_window", minTokens = "min_tokens", maxTokens = "max_tokens"
  }
  public func validate() throws {
    guard temperature.isFinite, (0...5).contains(temperature), topP.isFinite, topP > 0, topP <= 1,
      topK >= 1, topK <= 184704, repetitionPenalty.isFinite, repetitionPenalty > 0,
      (1...100).contains(penaltyWindow), minTokens >= 0, maxTokens >= 1,
      minTokens <= maxTokens, maxTokens <= 24576 else {
      throw StudioError.invalid("Choose valid music sampling values and token limits within the model context.")
    }
  }
}

public struct MusicRequest: Codable, Equatable {
  public var modelPath: String
  public var vaePath: String?
  public var style: String
  public var lyrics: String
  public var cot: String
  public var abc: String?
  public var seed: Int
  public var cfgScale: Double?
  public var steps: Int
  public var precision: String
  public var abcSampling: MusicSampling
  public var semanticSampling: MusicSampling
  public var memoryMode: String
  enum CodingKeys: String, CodingKey {
    case modelPath = "model_path", vaePath = "vae_path", style, lyrics, cot, abc, seed
    case cfgScale = "cfg_scale", steps, precision, abcSampling = "abc_sampling"
    case semanticSampling = "semantic_sampling", memoryMode = "memory_mode"
  }
}

/// Categories are saved separately from the compiled prompt so a take can restore user intent.
public struct MusicDraft: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var name = "New song"
  public var modelPath = ""
  public var vaePath = ""
  public var genre = "Indie pop"
  public var mood = "Hopeful"
  public var energy = "Medium"
  public var instruments = ""
  public var vocal = "Warm lead vocal"
  public var language = "English"
  public var bpm = 0
  public var instrumental = false
  public var direction = ""
  public var lyrics = "[Verse]\n\n[Chorus]\n"
  public var cot = "full"
  public var abc = ""
  public var seed = 831001
  public var useDefaultGuidance = true
  public var guidance: Double = 1
  public var steps = 32
  public var precision = "auto"
  public var scoreSampling = MusicSampling.score
  public var musicSampling = MusicSampling()
  public var memoryMode = "staged"
  public init() {}
  public var maximumSeconds: Double { Double(musicSampling.maxTokens) / 25 }
  public mutating func setMaximumSeconds(_ seconds: Double) {
    guard seconds.isFinite, seconds >= 1, seconds <= 983 else { return }
    musicSampling.maxTokens = Int(seconds * 25)
    musicSampling.minTokens = min(musicSampling.minTokens, musicSampling.maxTokens)
  }
  public var style: String {
    [genre, mood, energy.isEmpty ? "" : energy.lowercased() + " energy", instruments,
     instrumental ? "instrumental, no vocals" : vocal,
     instrumental ? "" : language, bpm > 0 ? "\(bpm) BPM" : "", direction]
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty }.joined(separator: ", ")
  }
  public func request() throws -> MusicRequest {
    try scoreSampling.validate(); try musicSampling.validate()
    guard !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw StudioError.invalid("Choose a local YuE2 model folder before generating music.")
    }
    guard ["full", "melody", "off"].contains(cot), ["auto", "bf16", "8bit", "4bit"].contains(precision),
      ["staged", "resident"].contains(memoryMode), (1...4096).contains(steps), seed >= 0,
      bpm >= 0, bpm <= 400, !style.isEmpty,
      useDefaultGuidance || (guidance.isFinite && (0...20).contains(guidance)) else {
      throw StudioError.invalid("Check composition mode, seed, precision, steps, tempo, and guidance.")
    }
    let score = abc.trimmingCharacters(in: .whitespacesAndNewlines)
    guard score.isEmpty || cot != "off" else {
      throw StudioError.invalid("A supplied score requires Melody or Full composition mode.")
    }
    return MusicRequest(modelPath: modelPath, vaePath: vaePath.isEmpty ? nil : vaePath,
      style: style, lyrics: instrumental ? "[Instrumental]" : lyrics, cot: cot,
      abc: score.isEmpty ? nil : score, seed: seed, cfgScale: useDefaultGuidance ? nil : guidance,
      steps: steps, precision: precision, abcSampling: scoreSampling,
      semanticSampling: musicSampling, memoryMode: memoryMode)
  }
}

public struct MusicGeneration: Codable, Equatable {
  public var draft: MusicDraft
  public var request: MusicRequest
  public var artifacts: String
  public var created: Date
  public var sampleRate: Int
  public var channels: Int
  public var truncated: Bool
  public var timings: [String: Double]
  public init(draft: MusicDraft, request: MusicRequest, artifacts: String,
              sampleRate: Int = 48000, channels: Int = 2, truncated: Bool = false,
              timings: [String: Double] = [:]) {
    self.draft = draft; self.request = request; self.artifacts = artifacts
    self.created = Date(); self.sampleRate = sampleRate; self.channels = channels
    self.truncated = truncated; self.timings = timings
  }
}

extension StudioProject {
  @discardableResult public mutating func placeMusic(
    _ asset: MediaAsset, at start: Double, trackID: UUID?, sourceIn: Double = 0,
    duration: Double? = nil
  ) throws -> AudioRegion {
    let length = duration ?? (asset.duration - sourceIn)
    guard asset.kind == .audio, !asset.path.isEmpty, start.isFinite, start >= 0,
      sourceIn.isFinite, sourceIn >= 0, length.isFinite, length > 0,
      asset.duration.isFinite, sourceIn + length <= asset.duration + 0.000001 else {
      throw StudioError.invalid("Choose an available music take and a range within its duration.")
    }
    if let trackID, !audioTracks.contains(where: { $0.id == trackID }) {
      throw StudioError.invalid("The selected music track is no longer in this movie.")
    }
    let destination: UUID
    if let trackID { destination = trackID }
    else if let music = audioTracks.first(where: { $0.role == .music && !$0.replacesSource }) { destination = music.id }
    else {
      let music = AudioTrack(); audioTracks.append(music); destination = music.id
    }
    var region = AudioRegion(assetID: asset.id, path: asset.path)
    region.start = start; region.sourceIn = sourceIn; region.duration = length
    region.trackID = destination
    audio.append(region)
    return region
  }
}

extension ProjectStorage {
  /// Preserve the exact stage files; their integrity manifest remains valid after relocation.
  public static func collectMusicArtifacts(_ generation: MusicGeneration, to destination: URL) throws -> MusicGeneration {
    try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: URL(fileURLWithPath: generation.artifacts), to: destination)
    var collected = generation; collected.artifacts = destination.path
    return collected
  }
}
