import Darwin
import Foundation
import LTX25Engine
import InferenceContracts

/// A timed edit is a separate LTX task. Keep its full-rate guide, editorial
/// interval, and adapter identity explicit instead of coercing it into 8+3.
public struct MLXRippleRequest: Codable, Sendable {
  public struct Anchor: Codable, Sendable {
    public let frame: Int
    public let path: String
    public let strength: Float

    private struct Key: CodingKey {
      let stringValue: String
      var intValue: Int? { nil }
      init(stringValue: String) { self.stringValue = stringValue }
      init?(intValue: Int) { return nil }
    }
    public init(from decoder: Decoder) throws {
      let raw = try decoder.container(keyedBy: Key.self)
      guard Set(raw.allKeys.map(\.stringValue)) == ["frame", "path", "strength"] else {
        throw LTXError.invalid("Ripple anchor has missing or unsupported fields.")
      }
      let value = try decoder.container(keyedBy: CodingKeys.self)
      frame = try value.decode(Int.self, forKey: .frame)
      path = try value.decode(String.self, forKey: .path)
      strength = try value.decode(Float.self, forKey: .strength)
    }
    enum CodingKeys: String, CodingKey { case frame, path, strength }
  }

  public let version: Int, engine: String, task: String
  public let gemmaRoot: String, transformerRoot: String, connectorCheckpoint: String
  public let videoCheckpoint: String, audioCheckpoint: String, adapterPath: String
  public let adapterStrength: Float, guidePath: String, firstReferencePath: String
  public let sourcePath: String, sourceSHA256: String
  public let sourceStart: Double, duration: Double, editorialFrames: Int
  public let width: Int, height: Int, frames: Int, fps: Double, seed: UInt64
  public let prompt: String, referenceStrength: Float, anchors: [Anchor]
  public let audioPolicy: String, ffmpegPath: String, outputDirectory: String

  enum CodingKeys: String, CodingKey, CaseIterable {
    case version, engine, task, prompt, width, height, frames, fps, seed, duration, anchors
    case gemmaRoot = "gemma_root", transformerRoot = "transformer_root"
    case connectorCheckpoint = "connector_checkpoint", videoCheckpoint = "video_checkpoint"
    case audioCheckpoint = "audio_checkpoint", adapterPath = "adapter_path"
    case adapterStrength = "adapter_strength", guidePath = "guide_path"
    case firstReferencePath = "first_reference_path"
    case sourcePath = "source_path", sourceSHA256 = "source_sha256"
    case sourceStart = "source_start", editorialFrames = "editorial_frames"
    case referenceStrength = "reference_strength", audioPolicy = "audio_policy"
    case ffmpegPath = "ffmpeg_path"
    case outputDirectory = "output_directory"
  }
  private struct Key: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
  }

  public init(from decoder: Decoder) throws {
    let raw = try decoder.container(keyedBy: Key.self)
    guard Set(raw.allKeys.map(\.stringValue)) == Set(CodingKeys.allCases.map(\.rawValue)) else {
      throw LTXError.invalid("Ripple request has missing or unsupported fields.")
    }
    let c = try decoder.container(keyedBy: CodingKeys.self)
    version = try c.decode(Int.self, forKey: .version)
    engine = try c.decode(String.self, forKey: .engine)
    task = try c.decode(String.self, forKey: .task)
    gemmaRoot = try c.decode(String.self, forKey: .gemmaRoot)
    transformerRoot = try c.decode(String.self, forKey: .transformerRoot)
    connectorCheckpoint = try c.decode(String.self, forKey: .connectorCheckpoint)
    videoCheckpoint = try c.decode(String.self, forKey: .videoCheckpoint)
    audioCheckpoint = try c.decode(String.self, forKey: .audioCheckpoint)
    adapterPath = try c.decode(String.self, forKey: .adapterPath)
    adapterStrength = try c.decode(Float.self, forKey: .adapterStrength)
    guidePath = try c.decode(String.self, forKey: .guidePath)
    firstReferencePath = try c.decode(String.self, forKey: .firstReferencePath)
    sourcePath = try c.decode(String.self, forKey: .sourcePath)
    sourceSHA256 = try c.decode(String.self, forKey: .sourceSHA256)
    sourceStart = try c.decode(Double.self, forKey: .sourceStart)
    duration = try c.decode(Double.self, forKey: .duration)
    editorialFrames = try c.decode(Int.self, forKey: .editorialFrames)
    width = try c.decode(Int.self, forKey: .width)
    height = try c.decode(Int.self, forKey: .height)
    frames = try c.decode(Int.self, forKey: .frames)
    fps = try c.decode(Double.self, forKey: .fps)
    seed = try c.decode(UInt64.self, forKey: .seed)
    prompt = try c.decode(String.self, forKey: .prompt)
    referenceStrength = try c.decode(Float.self, forKey: .referenceStrength)
    anchors = try c.decode([Anchor].self, forKey: .anchors)
    audioPolicy = try c.decode(String.self, forKey: .audioPolicy)
    ffmpegPath = try c.decode(String.self, forKey: .ffmpegPath)
    outputDirectory = try c.decode(String.self, forKey: .outputDirectory)

    guard version == 1, engine == "ltx25", task == "ripple", prompt.utf8.count <= 65_536,
      !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      adapterStrength.isFinite, (0.001...3).contains(adapterStrength),
      referenceStrength.isFinite, (0...1).contains(referenceStrength),
      sourceStart.isFinite, sourceStart >= 0, duration.isFinite, (0.001...30).contains(duration),
      fps.isFinite, (1...60).contains(fps), (1...1800).contains(editorialFrames),
      audioPolicy == "preserve" || audioPolicy == "silent",
      sourceSHA256.utf8.count == 64,
      sourceSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      [gemmaRoot, transformerRoot, connectorCheckpoint, videoCheckpoint, audioCheckpoint,
       adapterPath, guidePath, firstReferencePath, sourcePath, ffmpegPath,
       outputDirectory].allSatisfy({
        $0.hasPrefix("/") && $0.utf8.count <= 4096 && !$0.utf8.contains(0)
      }) else { throw LTXError.invalid("Ripple requires bounded local inputs and its pinned single-stage settings.") }
    let geometry = try AVGeometry(width: width, height: height, frames: frames, fps: fps)
    guard width <= 1920, height <= 1920, frames <= 1501,
      editorialFrames == Int(ceil(duration * fps - 1e-7)), editorialFrames <= frames,
      frames == 1 + 8 * max(1, Int(ceil(Double(editorialFrames - 1) / 8)),
        Int(ceil(0.25 * fps / 8))),
      Double(frames - 1) / fps <= 30,
      anchors.count <= 8,
      anchors.allSatisfy({ (1..<editorialFrames).contains($0.frame) &&
        $0.strength.isFinite && (0...1).contains($0.strength) &&
        $0.path.hasPrefix("/") && $0.path.utf8.count <= 4096 && !$0.path.utf8.contains(0) }),
      zip(anchors, anchors.dropFirst()).allSatisfy({ $0.0.frame < $0.1.frame }) else {
      throw LTXError.invalid("Ripple editorial timing, guide padding, or timed anchors are inconsistent.")
    }
    _ = try MLXReferenceVideoLayout(geometry: geometry, strength: referenceStrength,
      anchors: anchors.map { RippleImageAnchor(frame: $0.frame, strength: $0.strength) })
  }

  public var geometry: AVGeometry {
    // Validated at decode; this cannot fail for an immutable request.
    try! AVGeometry(width: width, height: height, frames: frames, fps: fps)
  }

  public func validateGuide() throws {
    let fd = Darwin.open(guidePath, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
    guard fd >= 0 else { throw LTXError.invalid("Cannot open the Ripple RGB24 guide.") }
    defer { Darwin.close(fd) }
    var info = stat()
    let expected = Int64(frames) * Int64(width) * Int64(height) * 3
    guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
      info.st_size == expected else {
      throw LTXError.invalid("Ripple guide bytes differ from the admitted full-rate geometry.")
    }
  }

  public func validateSource() throws {
    try NativeMediaSource(path: sourcePath, sha256: sourceSHA256).verify()
  }

  public func plan(maximumActivationBytes: Int) throws -> MLXSingleStageRipple.Plan {
    try MLXSingleStageRipple.plan(geometry: geometry, strength: referenceStrength,
      anchors: anchors.map { RippleImageAnchor(frame: $0.frame, strength: $0.strength) },
      maximumActivationBytes: maximumActivationBytes)
  }
}
