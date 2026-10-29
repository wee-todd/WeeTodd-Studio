import AVFoundation
import CryptoKit
import Foundation

public struct Drama3Receipt: Codable, Sendable {
  public var version = 1
  public var id: UUID
  public var audio = "speech.wav"
  public var codes = "codes.safetensors"
  public var sampleRate = 44100
  public var channels = 1
  public var frames: Int
  public var codeFrames: Int
  public var truncated: Bool
  public var audioSHA256: String
  public var codesSHA256: String
  public var modelFingerprint: String
  public var requestSHA256: String?
  public var warnings: [String]?
  public var codecFingerprint: String?
  public var segmentFrames: [Int]
  public var elapsedSeconds: Double
  public var peakResidentBytes: UInt64
  public var duration: Double { Double(frames) / Double(sampleRate) }
  public init(
    id: UUID, frames: Int, codeFrames: Int, truncated: Bool, audioSHA256: String,
    codesSHA256: String, modelFingerprint: String, segmentFrames: [Int], elapsedSeconds: Double,
    peakResidentBytes: UInt64, codecFingerprint: String? = nil, warnings: [String]? = nil,
    requestSHA256: String? = nil
  ) {
    self.requestSHA256 = requestSHA256
    self.warnings = warnings
    self.codecFingerprint = codecFingerprint
    self.id = id
    self.frames = frames
    self.codeFrames = codeFrames
    self.truncated = truncated
    self.audioSHA256 = audioSHA256
    self.codesSHA256 = codesSHA256
    self.modelFingerprint = modelFingerprint
    self.segmentFrames = segmentFrames
    self.elapsedSeconds = elapsedSeconds
    self.peakResidentBytes = peakResidentBytes
  }
}
public enum Drama3TakeBundle {
  public static func request(at root: URL) throws -> Drama3Request {
    let file = root.appendingPathComponent("request.json")
    guard (try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) <= 16 * 1024 * 1024
    else { throw Drama3Error.invalid("Saved speech request exceeds its bound.") }
    var request = try JSONDecoder().decode(Drama3Request.self, from: Data(contentsOf: file))
    request.resolveMediaPaths(relativeTo: root)
    return request
  }
  public static func verify(at root: URL) throws -> Drama3Receipt {
    let url = root.appendingPathComponent("receipt.json")
    guard (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) <= 1024 * 1024
    else { throw Drama3Error.invalid("Invalid speech receipt.") }
    let r = try JSONDecoder().decode(Drama3Receipt.self, from: Data(contentsOf: url))
    guard r.version == 1, r.sampleRate == 44100, r.channels == 1, r.frames > 0, r.codeFrames > 0,
      r.codeFrames <= 262144, r.frames == r.codeFrames * 2048, (r.segmentFrames.first ?? 0) > 0,
      r.segmentFrames.last == r.codeFrames,
      zip(r.segmentFrames, r.segmentFrames.dropFirst()).allSatisfy({ $0 < $1 }),
      r.elapsedSeconds.isFinite, r.elapsedSeconds >= 0
    else { throw Drama3Error.invalid("Invalid speech result metadata.") }
    for (name, hash) in [(r.audio, r.audioSHA256), (r.codes, r.codesSHA256)] {
      guard !name.hasPrefix("/"), !name.split(separator: "/").contains(".."), hash.count == 64
      else { throw Drama3Error.invalid("Unsafe speech result path.") }
      let target = root.appendingPathComponent(name).resolvingSymlinksInPath()
      let canonical = root.resolvingSymlinksInPath().path + "/"
      guard target.path.hasPrefix(canonical) else {
        throw Drama3Error.invalid("Speech output escaped its take bundle.")
      }
      let h = try FileHandle(forReadingFrom: target)
      defer { try? h.close() }
      var digest = SHA256()
      while let b = try h.read(upToCount: 1024 * 1024), !b.isEmpty { digest.update(data: b) }
      guard digest.finalize().map({ String(format: "%02x", $0) }).joined() == hash else {
        throw Drama3Error.invalid("Speech output checksum differs from its receipt.")
      }
    }
    if let expected = r.requestSHA256 {
      guard
        try Drama3AudioPreparation.fileDigest(root.appendingPathComponent("request.json").path)
          == expected
      else { throw Drama3Error.invalid("The saved speech request changed after generation.") }
    }
    let header = try Drama3TensorFile(root.appendingPathComponent(r.codes))
    guard let codes = header.tensors["codes"], codes.shape == [10, r.codeFrames],
      ["I32", "I64"].contains(codes.dtype)
    else { throw Drama3Error.invalid("Speech code dimensions differ from the receipt.") }
    let audio = try AVAudioFile(forReading: root.appendingPathComponent(r.audio))
    guard audio.length == r.frames, audio.fileFormat.sampleRate == 44100,
      audio.fileFormat.channelCount == 1
    else { throw Drama3Error.invalid("Speech audio differs from its receipt.") }
    return r
  }
}
public struct Drama3Event: Codable, Sendable {
  public var event = "progress"
  public var message: String
  public var fraction: Double?
  public var audioPreviewPath: String?
  public init(_ message: String, fraction: Double? = nil, audioPreviewPath: String? = nil) {
    self.message = message
    self.fraction = fraction
    self.audioPreviewPath = audioPreviewPath
  }
}
