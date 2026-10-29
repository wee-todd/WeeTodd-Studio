import Foundation

public enum Drama3Error: LocalizedError {
  case invalid(String)
  public var errorDescription: String? {
    if case .invalid(let s) = self { return s }
    return nil
  }
}
public enum Drama3Mode: String, Codable, Sendable {
  case speak, scene, lineContinuation, direct, infill, lineSeparated, streamingText
}
public enum Drama3EditOperation: String, Codable, Sendable { case replace, insert, delete }
public struct Drama3Codes: Codable, Equatable, Sendable {
  public var path: String
  public var frames: Int
  public var sha256: String?
  public init(path: String, frames: Int, sha256: String? = nil) {
    self.path = path
    self.frames = frames
    self.sha256 = sha256
  }
}
public struct Drama3Reference: Codable, Equatable, Sendable {
  public var text: String
  public var audioPath: String
  public var start: Double
  public var duration: Double
  public var audioSHA256: String?
  public var channel: String = "mix"
  public var speaker: Int = 0
  public var codes: Drama3Codes?
  public init(
    text: String, audioPath: String, start: Double = 0, duration: Double, speaker: Int = 0
  ) {
    self.text = text
    self.audioPath = audioPath
    self.start = start
    self.duration = duration
    self.speaker = speaker
  }
}
public struct Drama3History: Codable, Equatable, Sendable {
  public var text: String
  public var codes: Drama3Codes
  public var nextText: String?
  public init(text: String, codes: Drama3Codes, nextText: String? = nil) {
    self.text = text
    self.codes = codes
    self.nextText = nextText
  }
}
public struct Drama3Character: Codable, Equatable, Sendable {
  public var id: UUID
  public var token: Int
  public var name: String
  public var description: String
  public init(id: UUID, token: Int, name: String, description: String = "") {
    self.id = id
    self.token = token
    self.name = name
    self.description = description
  }
}
public struct Drama3Sampling: Codable, Equatable, Sendable {
  public var temperature: Float = 0.7
  public var topP: Float = 0.7
  public var topK: Int = 30
  public var maxFrames: Int = 1024
  public init() {}
}
public struct Drama3Request: Codable, Equatable, Sendable {
  public var version = 1
  public var id = UUID()
  public var modelPath: String
  public var text: String
  public var streamingChunks: [String]?
  public var mode: Drama3Mode = .speak
  public var seed: UInt64 = 42
  public var references: [Drama3Reference] = []
  public var history: [Drama3History] = []
  public var characters: [Drama3Character] = []
  public var prefix: Drama3Codes?
  public var suffix: Drama3Codes?
  public var sourceCodes: Drama3Codes?
  public var editOperation: Drama3EditOperation?
  public var editStart: Double?
  public var editEnd: Double?
  public var sourceAudio: String?
  public var direction = ""
  public var scene = ""
  public var language: String?
  public var durationHint: Double?
  public var sampling = Drama3Sampling()
  public var maxPromptRows = 8192
  public var stream = false
  public var firstChunkFrames = 4
  public var streamIntervalFrames = 7
  public var contextCodecFingerprint: String?
  public var outputDirectory = ""
  public var cacheDirectory = ""
  public var maxResidentBytes: UInt64?
  public init(modelPath: String, text: String) {
    self.modelPath = modelPath
    self.text = text
  }
  public func validate() throws -> Drama3ValidatedRequest {
    func require(_ condition: Bool, _ message: String) throws {
      if !condition { throw Drama3Error.invalid(message) }
    }
    try require(version == 1 && !modelPath.isEmpty, "Choose a compatible local Drama 3 checkpoint.")
    try Drama3Text.validate(text)
    try require(
      text.utf8.count <= 1_000_000 && direction.utf8.count <= 8000 && scene.utf8.count <= 16000,
      "Shorten the script or direction.")
    for s in [direction, scene] where !s.isEmpty { try Drama3Text.validate(s) }
    try require(
      characters.count <= 16 && Set(characters.map(\.id)).count == characters.count
        && Set(characters.map(\.token)).count == characters.count,
      "Use at most 16 distinct characters with stable speaker assignments.")
    try require(
      characters.allSatisfy {
        (0..<16).contains($0.token)
          && !$0.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      }, "Check character names and speaker assignments.")
    for c in characters {
      try Drama3Text.validate(c.name)
      if !c.description.isEmpty { try Drama3Text.validate(c.description) }
    }
    try require(
      references.count <= 64 && history.count <= 64,
      "Reduce the number of references or active history turns.")
    for r in references {
      try Drama3Text.validate(r.text)
      try require(
        !r.audioPath.isEmpty && r.start.isFinite && r.start >= 0 && r.duration.isFinite
          && r.duration > 0 && r.duration <= 60 && (0..<16).contains(r.speaker)
          && ["mix", "left", "right"].contains(r.channel),
        "Choose a reference range up to 60 seconds and a matching transcript.")
    }
    if let streamingChunks {
      try require(
        mode == .streamingText && streamingChunks.count <= 4096 && !streamingChunks.isEmpty,
        "Check the recorded text stream.")
      for chunk in streamingChunks { try Drama3Text.validate(chunk) }
    }
    for h in history {
      if let next = h.nextText { try Drama3Text.validate(next) }
      try Drama3Text.validate(h.text)
      try require(
        h.codes.frames > 0 && h.codes.frames <= 262144 && !h.codes.path.isEmpty,
        "A prior take has invalid audio codes.")
    }
    let s = sampling
    try require(
      s.temperature.isFinite && (0...2).contains(s.temperature) && s.topP.isFinite && s.topP > 0
        && s.topP <= 1 && (1...4096).contains(s.topK) && (1...16384).contains(s.maxFrames),
      "Check Drama 3 sampling settings.")
    try require(
      (1...32768).contains(maxPromptRows) && (1...256).contains(firstChunkFrames)
        && (1...256).contains(streamIntervalFrames), "Check context and streaming budgets.")
    if let language {
      try require(
        language.range(of: "^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?$", options: .regularExpression)
          != nil, "Use a language code such as en, zh or pt-BR.")
    }
    if let durationHint {
      try require(
        durationHint.isFinite && durationHint > 0 && durationHint <= 600,
        "Duration guidance must be between 0 and 600 seconds.")
    }
    if mode == .direct {
      try require(
        !history.isEmpty && !direction.isEmpty, "Select a prior take and enter a director note.")
    }
    if mode == .infill {
      try require(
        (prefix != nil && suffix != nil) || sourceCodes != nil || sourceAudio != nil,
        "Select speech to edit.")
      if prefix == nil {
        try require(
          editStart?.isFinite == true && editEnd?.isFinite == true && editStart! >= 0
            && editEnd! >= editStart!, "Choose a valid edit range.")
      }
    }
    if mode == .infill, editOperation == .delete {
      try require((editEnd ?? 0) > (editStart ?? 0), "Select a nonempty range to delete.")
    }
    if mode == .infill, editOperation == .insert {
      try require(editEnd == editStart, "An insertion uses a single time boundary.")
    }
    let rows =
      references.reduce(0) { $0 + Int(ceil($1.duration * 44100 / 2048)) + $1.text.utf8.count }
      + history.reduce(0) { $0 + $1.codes.frames + $1.text.utf8.count }
    try require(
      rows + min(text.utf8.count, 300) + 64 <= maxPromptRows,
      "Reference/history context exceeds the prompt budget. Shorten samples or remove active history."
    )
    return Drama3ValidatedRequest(request: self)
  }
}
public struct Drama3ValidatedRequest: Sendable { public let request: Drama3Request }
public struct Drama3SpeakerMap: Codable, Equatable {
  public var slots: [UUID: Int] = [:]
  public init() {}
  public mutating func assign(_ ids: [UUID]) throws -> [UUID: Int] {
    guard ids.count <= 16, Set(ids).count == ids.count else {
      throw Drama3Error.invalid("Use at most 16 unique characters.")
    }
    slots = slots.filter { ids.contains($0.key) }
    for id in ids where slots[id] == nil {
      slots[id] = (0..<16).first { !slots.values.contains($0) }!
    }
    return slots
  }
}

extension Drama3Request {
  public mutating func mapMediaPaths(_ transform: (String) throws -> String) rethrows {
    for i in references.indices {
      references[i].audioPath = try transform(references[i].audioPath)
      if var codes = references[i].codes {
        codes.path = try transform(codes.path)
        references[i].codes = codes
      }
    }
    for i in history.indices { history[i].codes.path = try transform(history[i].codes.path) }
    if var c = prefix, !c.path.isEmpty {
      c.path = try transform(c.path)
      prefix = c
    }
    if var c = suffix, !c.path.isEmpty {
      c.path = try transform(c.path)
      suffix = c
    }
    if var c = sourceCodes {
      c.path = try transform(c.path)
      sourceCodes = c
    }
    if let audio = sourceAudio { sourceAudio = try transform(audio) }
  }
  public mutating func resolveMediaPaths(relativeTo root: URL) {
    mapMediaPaths { value in
      value.isEmpty || value.hasPrefix("/")
        ? value : root.appendingPathComponent(value).standardizedFileURL.path
    }
  }
}
