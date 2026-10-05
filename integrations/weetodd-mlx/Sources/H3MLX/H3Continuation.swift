import CryptoKit
import Foundation

/// Native, process-independent H3 continuation. The payload contains normalized
/// sampler rows, never pixels or VAE posterior samples. This version is kept
/// separate from the Python v1 SafeTensors format until cross-runtime parity is
/// qualified.
public enum H3Continuation {
  public static let allowedContextFrames: Set<Int> = [5, 22, 39, 56]
  private static let maxPayloadBytes = 64 * 1024 * 1024

  public struct Plan: Sendable {
    public let contextFrames: Int
    public let generatedFrames: Int
    public let publishedFrames: Int
    public let overlapFrames: Int
    public let tailTrimFrames: Int
    public let saveContext: Bool
    public let sourceManifest: URL?
    public let sourceSHA256: String?

    public init(contextFrames: Int, requestedDuration: Double,
      sourceManifest: URL?, sourceSHA256: String?, saveContext: Bool) throws {
      guard allowedContextFrames.contains(contextFrames),
        requestedDuration.isFinite, (2.5...15).contains(requestedDuration),
        (sourceManifest == nil) == (sourceSHA256 == nil),
        sourceManifest != nil || saveContext else {
        throw H3CheckpointError.invalid("Invalid H3 continuation request.")
      }
      if let sourceManifest, let sourceSHA256 {
        guard sourceManifest.isFileURL, sourceManifest.path.hasPrefix("/"),
          H3Continuation.validSHA256(sourceSHA256) else {
          throw H3CheckpointError.invalid("Invalid H3 source context or digest.")
        }
      }
      let requested = Int((requestedDuration * 24).rounded(.toNearestOrEven))
      let overlap = sourceManifest == nil ? 0 : contextFrames
      var generated = requested + overlap
      while generated % 17 != 5 { generated += 1 }
      let published = sourceManifest == nil ? generated : requested
      let trim = generated - overlap - published
      guard generated <= 362, published > contextFrames,
        !(saveContext && trim > 0) else {
        throw H3CheckpointError.invalid("H3 continuation exceeds the sampling window or would save a trimmed tail.")
      }
      self.contextFrames = contextFrames
      self.generatedFrames = generated
      self.publishedFrames = published
      self.overlapFrames = overlap
      self.tailTrimFrames = trim
      self.saveContext = saveContext
      self.sourceManifest = sourceManifest
      self.sourceSHA256 = sourceSHA256
    }
  }

  public struct Rows: Sendable {
    public let video: [Float]
    public let audio: [Float]
  }

  private struct Manifest: Codable {
    let format: String
    let task: String?
    let contextFrames: Int
    let width: Int
    let height: Int
    let generatedFrames: Int
    let publishedFrames: Int
    let overlapFrames: Int
    let identity: String
    let payloadBytes: Int
    let payloadSHA256: String
  }

  private static func validSHA256(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy {
      (48...57).contains($0) || (97...102).contains($0)
    }
  }

  private static func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private static func readBounded(_ url: URL, limit: Int) throws -> Data {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    let data = try file.read(upToCount: limit + 1) ?? Data()
    guard data.count <= limit else {
      throw H3CheckpointError.invalid("H3 continuation artifact exceeds its size limit.")
    }
    return data
  }

  private static func shape(contextFrames: Int, width: Int, height: Int)
    throws -> (videoFrames: Int, videoRows: Int, audioFrames: Int, videoFloats: Int,
      audioFloats: Int) {
    guard allowedContextFrames.contains(contextFrames),
      (32...1920).contains(width), (32...1920).contains(height),
      width.isMultiple(of: 32), height.isMultiple(of: 32) else {
      throw H3CheckpointError.invalid("Invalid H3 continuation canvas or context.")
    }
    let frames = ((contextFrames - 5) / 17) * 5 + 2
    let rows = (height / 32) * (width / 32)
    let audio = Int((Double(contextFrames) / 24 * 40).rounded(.toNearestOrEven))
    return (frames, rows, audio, frames * rows * 96, 2 * audio * 32)
  }

  /// Retain only the synchronized latent tail. Audio rows are channel-major,
  /// so the tail must be extracted separately from each channel.
  public static func tail(video: [Float], audio: [Float], geometry: H3Geometry,
    contextFrames: Int) throws -> Rows {
    let tail = try shape(contextFrames: contextFrames,
      width: geometry.width, height: geometry.height)
    guard geometry.frames > contextFrames,
      video.count == geometry.videoRows * 96,
      audio.count == geometry.audioRows * 32,
      video.allSatisfy(\.isFinite), audio.allSatisfy(\.isFinite) else {
      throw H3CheckpointError.invalid("H3 continuation source latent rows are incomplete.")
    }
    let sourceChannelFloats = geometry.audioLatentFrames * 32
    let tailChannelFloats = tail.audioFrames * 32
    return Rows(video: Array(video.suffix(tail.videoFloats)),
      audio: Array(audio[(sourceChannelFloats - tailChannelFloats)..<sourceChannelFloats]) +
        Array(audio[(2 * sourceChannelFloats - tailChannelFloats)..<(2 * sourceChannelFloats)]))
  }

  /// Identity is a fingerprint of installed component metadata and effective
  /// generation controls. It is intentionally labeled as such; a future
  /// cross-runtime format must use full content hashes for model identity.
  public static func fingerprint(_ request: H3T2VARequest) throws -> String {
    guard request.vdn == nil else { throw H3CheckpointError.invalid("VDN continuation is not qualified.") }
    var records: [String] = ["h3-swift-continuation-v2", "24", "32000",
      String(request.geometry.width), String(request.geometry.height),
      String(request.requestedSteps)]
    if request.samplingMethod != .euler {
      records.append("sampling_method=" + request.samplingMethod.rawValue)
    }
    let components = [request.transformer, request.qwenPages, request.tokenizer,
      request.videoVAE, request.audioVAE] + request.loRAAdapters.map(\.url)
    for component in components {
      let root = component.resolvingSymlinksInPath().standardizedFileURL
      var isDirectory: ObjCBool = false
      guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory) else {
        throw H3CheckpointError.invalid("Missing H3 continuation component.")
      }
      let files: [URL]
      if isDirectory.boolValue {
        guard let iterator = FileManager.default.enumerator(at: root,
          includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey],
          options: []) else {
          throw H3CheckpointError.invalid("Cannot inventory H3 continuation component.")
        }
        var discovered: [URL] = []
        for case let file as URL in iterator {
          guard discovered.count < 10_000,
            try file.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true else {
            throw H3CheckpointError.invalid("Unsafe H3 continuation component tree.")
          }
          if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
            discovered.append(file)
          }
        }
        files = discovered.sorted { $0.path < $1.path }
      } else { files = [root] }
      guard !files.isEmpty else {
        throw H3CheckpointError.invalid("Empty H3 continuation component.")
      }
      records.append(root.path)
      for file in files {
        let values = try file.resourceValues(forKeys: [.fileSizeKey,
          .contentModificationDateKey, .creationDateKey])
        guard let size = values.fileSize, let modified = values.contentModificationDate,
          let created = values.creationDate else {
          throw H3CheckpointError.invalid("Cannot fingerprint H3 continuation component.")
        }
        records.append("\(file.path)|\(size)|\(modified.timeIntervalSince1970)|\(created.timeIntervalSince1970)")
      }
    }
    for adapter in request.loRAAdapters {
      records.append("lora|\(adapter.url.path)|\(adapter.strength)")
      if adapter.qkvLayout != .auto || adapter.profile != .auto || adapter.startAfterEvaluations != 0 {
        records.append("lora_controls|\(adapter.qkvLayout.rawValue)|\(adapter.profile.rawValue)|\(adapter.startAfterEvaluations)")
      }
    }
    return digest(Data(records.joined(separator: "\n").utf8))
  }

  /// FL identity includes the vision tower and task; it cannot alias a T2VA tail.
  public static func fingerprint(_ request: H3FL2VARequest) throws -> String {
    let root = request.vision.resolvingSymlinksInPath().standardizedFileURL
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory) else {
      throw H3CheckpointError.invalid("Missing FL2VA continuation vision component.")
    }
    var files: [URL] = []
    if isDirectory.boolValue {
      guard let iterator = FileManager.default.enumerator(at: root,
        includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]) else {
        throw H3CheckpointError.invalid("Cannot inventory FL2VA vision component.")
      }
      for case let file as URL in iterator {
        try Task.checkCancellation()
        guard files.count < 10_000,
          try file.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true else {
          throw H3CheckpointError.invalid("Unsafe FL2VA vision component tree.")
        }
        if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true { files.append(file) }
      }
    } else { files = [root] }
    guard !files.isEmpty else { throw H3CheckpointError.invalid("Empty FL2VA vision component.") }
    var records = ["h3-swift-continuation-v2|fl2va", try fingerprint(request.base), root.path]
    if let noise = request.referenceNoise {
      records.append("visual_condition_strength=" + String(noise.visual))
      records.append("audio_condition_strength=" + String(noise.audio))
    }
    for file in files.sorted(by: { $0.path < $1.path }) {
      let values = try file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey, .creationDateKey])
      guard let size = values.fileSize, let modified = values.contentModificationDate,
        let created = values.creationDate else { throw H3CheckpointError.invalid("Cannot fingerprint FL2VA vision component.") }
      records.append("\(file.path)|\(size)|\(modified.timeIntervalSince1970)|\(created.timeIntervalSince1970)")
    }
    return digest(Data(records.joined(separator: "\n").utf8))
  }

  public static func fingerprint(_ request: H3Ref2VAStillRequest) throws -> String {
    let base = try H3T2VARequest(prompt: request.prompt,
      width: request.geometry.width, height: request.geometry.height,
      durationSeconds: min(Double(request.geometry.frames) / 24, 15), seed: request.seed,
      requestedSteps: request.requestedSteps, transformer: request.transformer,
      qwenPages: request.qwenPages, tokenizer: request.tokenizer,
      videoVAE: request.videoVAE, audioVAE: request.audioVAE,
      turboLoRA: request.turboLoRA, turboLoRAStrength: request.turboLoRAStrength,
      additionalLoRAs: request.additionalLoRAs, loRAAdapters: request.loRAAdapters,
      videoDecodeMemoryMode: request.videoDecodeMemoryMode, samplingMethod: request.samplingMethod)
    let root = request.qwenVision.resolvingSymlinksInPath().standardizedFileURL
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory) else {
      throw H3CheckpointError.invalid("Missing Ref2VA continuation vision component.")
    }
    var files: [URL] = []
    if isDirectory.boolValue {
      guard let iterator = FileManager.default.enumerator(at: root,
        includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]) else {
        throw H3CheckpointError.invalid("Cannot inventory Ref2VA vision component.")
      }
      for case let file as URL in iterator {
        try Task.checkCancellation()
        guard files.count < 10_000,
          try file.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true else {
          throw H3CheckpointError.invalid("Unsafe Ref2VA vision component tree.")
        }
        if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true { files.append(file) }
      }
    } else { files = [root] }
    guard !files.isEmpty else { throw H3CheckpointError.invalid("Empty Ref2VA vision component.") }
    var records = ["h3-swift-continuation-v2|ref2va", try fingerprint(base), root.path]
    if let noise = request.referenceNoise {
      records.append("visual_condition_strength=" + String(noise.visual))
      records.append("audio_condition_strength=" + String(noise.audio))
    }
    for file in files.sorted(by: { $0.path < $1.path }) {
      let values = try file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey, .creationDateKey])
      guard let size = values.fileSize, let modified = values.contentModificationDate,
        let created = values.creationDate else { throw H3CheckpointError.invalid("Cannot fingerprint Ref2VA vision component.") }
      records.append("\(file.path)|\(size)|\(modified.timeIntervalSince1970)|\(created.timeIntervalSince1970)")
    }
    return digest(Data(records.joined(separator: "\n").utf8))
  }

  public static func save(_ rows: Rows, plan: Plan, width: Int, height: Int,
    identity: String, directory: URL, task: String = "t2va") throws -> (manifest: URL, sha256: String, payloadSHA256:String) {
    let expected = try shape(contextFrames: plan.contextFrames,
      width: width, height: height)
    guard ["t2va", "fl2va", "ref2va"].contains(task), plan.saveContext, plan.tailTrimFrames == 0,
      validSHA256(identity), rows.video.count == expected.videoFloats,
      rows.audio.count == expected.audioFloats,
      rows.video.allSatisfy(\.isFinite), rows.audio.allSatisfy(\.isFinite),
      directory.isFileURL, !FileManager.default.fileExists(atPath: directory.path) else {
      throw H3CheckpointError.invalid("Invalid H3 continuation artifact output.")
    }
    var payload = Data(capacity: (rows.video.count + rows.audio.count) * 4)
    for value in rows.video + rows.audio {
      var word = value.bitPattern.littleEndian
      withUnsafeBytes(of: &word) { payload.append(contentsOf: $0) }
    }
    guard payload.count <= maxPayloadBytes else {
      throw H3CheckpointError.invalid("H3 continuation payload exceeds 64 MiB.")
    }
    let manifest = Manifest(format: "weetodd-h3-swift-continuation-v2",
      task: task == "t2va" ? nil : task,
      contextFrames: plan.contextFrames, width: width, height: height,
      generatedFrames: plan.generatedFrames, publishedFrames: plan.publishedFrames,
      overlapFrames: plan.overlapFrames, identity: identity,
      payloadBytes: payload.count, payloadSHA256: digest(payload))
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let manifestData = try encoder.encode(manifest)
    guard manifestData.count <= 1024 * 1024 else {
      throw H3CheckpointError.invalid("H3 continuation manifest exceeds 1 MiB.")
    }
    try FileManager.default.createDirectory(at: directory,
      withIntermediateDirectories: false)
    do {
      try payload.write(to: directory.appendingPathComponent("latents.f32"), options: .atomic)
      let path = directory.appendingPathComponent("manifest.json")
      try manifestData.write(to: path, options: .atomic)
      return (path, digest(manifestData),manifest.payloadSHA256)
    } catch {
      try? FileManager.default.removeItem(at: directory)
      throw error
    }
  }

  public static func load(manifestURL: URL, expectedSHA256: String,
    contextFrames: Int, width: Int, height: Int, identity: String,
    loadRows: Bool = true, task: String = "t2va") throws -> Rows? {
    let manifestAttributes = try manifestURL.resourceValues(forKeys: [.fileSizeKey,
      .isRegularFileKey, .isSymbolicLinkKey])
    guard manifestURL.isFileURL, validSHA256(expectedSHA256),
      validSHA256(identity),
      manifestAttributes.isRegularFile == true,
      manifestAttributes.isSymbolicLink != true,
      let size = manifestAttributes.fileSize, size <= 1024 * 1024 else {
      throw H3CheckpointError.invalid("Invalid H3 continuation manifest.")
    }
    let manifestData = try readBounded(manifestURL, limit: 1024 * 1024)
    guard digest(manifestData) == expectedSHA256,
      let manifest = try? JSONDecoder().decode(Manifest.self, from: manifestData),
      manifest.format == "weetodd-h3-swift-continuation-v2",
      ["t2va", "fl2va", "ref2va"].contains(task), (manifest.task ?? "t2va") == task,
      manifest.contextFrames == contextFrames,
      manifest.width == width, manifest.height == height,
      manifest.identity == identity,
      manifest.generatedFrames - manifest.overlapFrames == manifest.publishedFrames,
      manifest.generatedFrames % 17 == 5 else {
      throw H3CheckpointError.invalid("H3 continuation manifest identity or timing changed.")
    }
    let shape = try shape(contextFrames: contextFrames, width: width, height: height)
    let expectedBytes = (shape.videoFloats + shape.audioFloats) * 4
    let payloadURL = manifestURL.deletingLastPathComponent().appendingPathComponent("latents.f32")
    let attributes = try payloadURL.resourceValues(forKeys: [.isRegularFileKey,
      .isSymbolicLinkKey, .fileSizeKey])
    guard attributes.isRegularFile == true, attributes.isSymbolicLink != true,
      manifest.payloadBytes == expectedBytes,
      attributes.fileSize == expectedBytes, expectedBytes <= maxPayloadBytes else {
      throw H3CheckpointError.invalid("H3 continuation payload size or type changed.")
    }
    let payload = try readBounded(payloadURL, limit: maxPayloadBytes)
    guard digest(payload) == manifest.payloadSHA256 else {
      throw H3CheckpointError.invalid("H3 continuation payload digest changed.")
    }
    guard loadRows else { return nil }
    let values = payload.withUnsafeBytes { buffer -> [Float] in
      let words = buffer.bindMemory(to: UInt32.self)
      return words.map { Float(bitPattern: UInt32(littleEndian: $0)) }
    }
    guard values.allSatisfy(\.isFinite) else {
      throw H3CheckpointError.invalid("H3 continuation payload contains nonfinite latents.")
    }
    return Rows(video: Array(values[..<shape.videoFloats]),
      audio: Array(values[shape.videoFloats...]))
  }
}
