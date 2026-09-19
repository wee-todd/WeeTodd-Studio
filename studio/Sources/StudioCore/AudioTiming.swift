import Foundation

public struct AudioTimingMarker: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var timeSeconds: Double
  public var locked: Bool
  public var label: String
  enum CodingKeys: String, CodingKey { case timeSeconds, locked, label }
  public init(timeSeconds: Double, locked: Bool = false, label: String = "Cut") {
    self.timeSeconds = timeSeconds; self.locked = locked; self.label = label
  }
  public func frame(fps: Double, sourceStart: Double, duration: Double) throws -> Int {
    guard fps.isFinite, fps > 0, sourceStart.isFinite, sourceStart >= 0, duration.isFinite, duration > 0,
      timeSeconds.isFinite, timeSeconds > sourceStart, timeSeconds < sourceStart + duration,
      (timeSeconds-sourceStart)*fps < Double(Int.max) else {
      throw StudioError.invalid("Timing markers must be inside the selected song interval.")
    }
    let frame = Int(((timeSeconds-sourceStart)*fps).rounded())
    guard frame > 0, Double(frame) < (duration*fps).rounded(.up) else { throw StudioError.invalid("Place the marker on an interior movie frame.") }
    return frame
  }
}

public struct AudioWordTiming: Codable, Equatable {
  public var text: String
  public var startSeconds: Double?
  public var endSeconds: Double?
  public var confidence: Double
  public var flags: [String]
  public var observedText: String?
  public var verification: String?
  public var verificationLabel: String? {
    switch verification {
    case "matched": return "Lyrics and recognition agree"
    case "lyric_assisted": return "Lyric-assisted — review"
    case "unresolved": return "Unresolved difference"
    default: return nil
    }
  }
  /// Human-reviewed cut timing stays separate from uncertain acoustic word evidence.
  public func reviewedCutMarker(at sourceSeconds: Double, fps: Double, sourceStart: Double, duration: Double) throws -> AudioTimingMarker {
    let marker = AudioTimingMarker(timeSeconds: sourceSeconds, label: "Reviewed lyric: \(text)")
    _ = try marker.frame(fps: fps, sourceStart: sourceStart, duration: duration)
    return marker
  }
}

public struct AudioEditingCue: Codable, Identifiable {
  public var id: String
  public var kind: String
  public var timeSeconds: Double
  public var frame: Int
  public var snapErrorSeconds: Double
  public var confidence: Double
  public var label: String
}

public struct AudioAnalysisReview: Codable {
  public var sourceSHA256: String
  public var cues: [AudioEditingCue]
  public var words: [AudioWordTiming]
  public var lines: [AudioWordTiming]
  public var extraWords: [AudioWordTiming]
  public var warnings: [String]
  public var beatCount: Int
  public var downbeatCount: Int
  public var elapsedSeconds: Double?
  public var vocalMode: String?
  public var recognizedText: String?
  public var lyricAssistedText: String?
  public init(result: [String: Any]) throws {
    guard let analysis = result["analysis"] as? [String: Any], let hash = analysis["sourceSHA256"] as? String else {
      throw StudioError.invalid("Audio analysis did not return source provenance.")
    }
    sourceSHA256 = hash
    vocalMode = (analysis["vocalAnalysis"] as? [String: Any])?["mode"] as? String
    func decode<T: Decodable>(_ value: Any?, as type: T.Type) throws -> T {
      try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: value ?? []))
    }
    cues = try decode(result["editing_cues"], as: [AudioEditingCue].self)
    let alignment = analysis["alignment"] as? [String: Any]
    recognizedText = alignment?["recognizedText"] as? String
    lyricAssistedText = alignment?["lyricAssistedText"] as? String
    words = try decode(alignment?["words"], as: [AudioWordTiming].self)
    lines = try decode(alignment?["lines"], as: [AudioWordTiming].self)
    extraWords = try decode(alignment?["extraWords"], as: [AudioWordTiming].self)
    warnings = (analysis["limitations"] as? [String] ?? []) + (analysis["structureLimitations"] as? [String] ?? [])
    beatCount = (analysis["beats"] as? [Any])?.count ?? 0
    downbeatCount = (analysis["downbeats"] as? [Any])?.count ?? 0
    elapsedSeconds = analysis["elapsedSeconds"] as? Double
  }
}
