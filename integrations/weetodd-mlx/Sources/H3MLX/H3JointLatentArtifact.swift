import CryptoKit
import Darwin
import Foundation

/// Complete normalized AV rows for explicit native spatial/motion refinement.
/// This is not a continuation tail and is never accepted by that artifact loader.
public enum H3JointLatentArtifact {
  public static let maximumPayloadBytes = 64 * 1024 * 1024
  public struct Rows: Sendable {
    public let video: [Float]
    public let audio: [Float]
    public init(video: [Float], audio: [Float]) { self.video = video; self.audio = audio }
    public func validate(geometry: H3Geometry) throws {
      try geometry.canvasAdmission.validate(width: geometry.width, height: geometry.height)
      try geometry.canvasAdmission.validatePackedRows(geometry.videoRows + geometry.audioRows + 1)
      guard geometry.width * geometry.height <= geometry.canvasAdmission.maximumPixels,
        video.count == geometry.videoRows * 96, audio.count == geometry.audioRows * 32,
        (video.count + audio.count) * 4 <= H3JointLatentArtifact.maximumPayloadBytes,
        video.allSatisfy(\.isFinite), audio.allSatisfy(\.isFinite) else {
        throw H3CheckpointError.invalid("H3 initialized sampling requires complete finite normalized AV rows.")
      }
    }
  }
  public struct Manifest: Codable, Sendable {
    public let format: String
    public let task: String
    public let width: Int
    public let height: Int
    public let generatedFrames: Int
    public let componentIdentity: String
    public let videoFloats: Int
    public let audioFloats: Int
    public let payloadBytes: Int
    public let payloadSHA256: String
    public var geometry: H3Geometry {
      get throws { try H3Geometry(width: width, height: height,
        durationSeconds: min(Double(generatedFrames) / 24, 15),
        canvasAdmission: format == "weetodd-h3-swift-joint-latents-v2-spatial" ? .spatialRefinement : .ordinary) }
    }
  }
  /// Component-only metadata identity: a refinement may change canvas, seed,
  /// sampling grid and ordinary adapters, but never silently substitute model
  /// partitions. Reuse the existing native component inventory implementation.
  public static func componentIdentity(base: H3T2VARequest, task: String,
    vision: URL? = nil) throws -> String {
    guard ["t2va", "fl2va", "ref2va"].contains(task),
      (task == "t2va") == (vision == nil) else {
      throw H3CheckpointError.invalid("H3 full latent component binding requires the exact task and vision components.")
    }
    let canonical = try H3T2VARequest(prompt: "H3 component identity",
      width: 32, height: 32, durationSeconds: 2.5, seed: 0, requestedSteps: 20,
      transformer: base.transformer, qwenPages: base.qwenPages, tokenizer: base.tokenizer,
      videoVAE: base.videoVAE, audioVAE: base.audioVAE)
    var value = try H3Continuation.fingerprint(canonical)
    if let vision {
      // FL's existing fingerprint includes only its explicit vision component
      // in addition to the base request; no images are decoded here.
      let dummy = H3StillReference(rgb8: Data(count: 32 * 32 * 3), width: 32, height: 32)
      let request = try H3FL2VARequest(base: canonical, vision: vision,
        images: [dummy], anchors: [.first])
      value = try H3Continuation.fingerprint(request)
    }
    return sha(Data(("weetodd-h3-joint-components-v1|" + task + "|" + value).utf8))
  }
  private static func sha(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
  private static func validSHA(_ string: String) -> Bool {
    string.utf8.count == 64 && string.utf8.allSatisfy {
      (48...57).contains($0) || (97...102).contains($0)
    }
  }
  private static func read(_ url: URL, limit: Int) throws -> Data {
    guard url.isFileURL, url.path.hasPrefix("/"), !url.path.utf8.contains(0) else {
      throw H3CheckpointError.invalid("H3 full latent artifacts need absolute local files.")
    }
    let fd = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
    guard fd >= 0 else { throw H3CheckpointError.invalid("Cannot open H3 full latent artifact.") }
    let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true); defer { try? file.close() }
    var before = stat(), after = stat()
    guard fstat(fd, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
      before.st_size > 0, before.st_size <= limit else {
      throw H3CheckpointError.invalid("H3 full latent artifact must be a bounded regular file.")
    }
    try Task.checkCancellation()
    let bytes = try file.read(upToCount: limit + 1) ?? Data()
    guard bytes.count == before.st_size, fstat(fd, &after) == 0,
      before.st_dev == after.st_dev, before.st_ino == after.st_ino,
      before.st_size == after.st_size,
      before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
      before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
      before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec,
      before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec else {
      throw H3CheckpointError.invalid("H3 full latent artifact changed while reading.")
    }
    try Task.checkCancellation()
    return bytes
  }
  public static func inspect(manifestURL: URL, expectedSHA256: String,
    expectedTask: String, expectedComponentIdentity: String) throws -> Manifest {
    guard validSHA(expectedSHA256), validSHA(expectedComponentIdentity),
      ["t2va", "fl2va", "ref2va"].contains(expectedTask) else {
      throw H3CheckpointError.invalid("Invalid H3 full latent source binding.")
    }
    let bytes = try read(manifestURL, limit: 1024 * 1024)
    let root = try JSONSerialization.jsonObject(with: bytes) as? [String: Any]
    guard sha(bytes) == expectedSHA256, let root,
      Set(root.keys) == ["format", "task", "width", "height", "generatedFrames",
        "componentIdentity", "videoFloats", "audioFloats", "payloadBytes", "payloadSHA256"] else {
      throw H3CheckpointError.invalid("H3 full latent manifest digest or fields changed.")
    }
    let manifest = try JSONDecoder().decode(Manifest.self, from: bytes)
    let geometry = try manifest.geometry
    guard ["weetodd-h3-swift-joint-latents-v1", "weetodd-h3-swift-joint-latents-v2-spatial"].contains(manifest.format),
      manifest.task == expectedTask, manifest.componentIdentity == expectedComponentIdentity,
      manifest.generatedFrames == geometry.frames,
      manifest.videoFloats == geometry.videoRows * 96,
      manifest.audioFloats == geometry.audioRows * 32,
      manifest.payloadBytes == (manifest.videoFloats + manifest.audioFloats) * 4,
      (1...maximumPayloadBytes).contains(manifest.payloadBytes),
      geometry.width * geometry.height <= geometry.canvasAdmission.maximumPixels,
      validSHA(manifest.payloadSHA256) else {
      throw H3CheckpointError.invalid("H3 full latent source task, component identity or geometry differs.")
    }
    try geometry.canvasAdmission.validate(width: geometry.width, height: geometry.height)
    try geometry.canvasAdmission.validatePackedRows(geometry.videoRows + geometry.audioRows + 1)
    return manifest
  }
  public static func verify(manifestURL: URL, expectedSHA256: String,
    expectedTask: String, expectedComponentIdentity: String) throws {
    let manifest = try inspect(manifestURL: manifestURL, expectedSHA256: expectedSHA256,
      expectedTask: expectedTask, expectedComponentIdentity: expectedComponentIdentity)
    let bytes = try read(manifestURL.deletingLastPathComponent()
      .appendingPathComponent("joint-latents.f32"), limit: maximumPayloadBytes)
    guard bytes.count == manifest.payloadBytes, sha(bytes) == manifest.payloadSHA256 else {
      throw H3CheckpointError.invalid("H3 full latent payload changed after source admission.")
    }
  }
  public static func load(manifestURL: URL, expectedSHA256: String,
    expectedTask: String, expectedComponentIdentity: String) throws -> (Manifest, Rows) {
    let manifest = try inspect(manifestURL: manifestURL, expectedSHA256: expectedSHA256,
      expectedTask: expectedTask, expectedComponentIdentity: expectedComponentIdentity)
    let bytes = try read(manifestURL.deletingLastPathComponent()
      .appendingPathComponent("joint-latents.f32"), limit: maximumPayloadBytes)
    guard bytes.count == manifest.payloadBytes, sha(bytes) == manifest.payloadSHA256 else {
      throw H3CheckpointError.invalid("H3 full latent payload digest or length changed.")
    }
    var values = [Float](); values.reserveCapacity(bytes.count / 4)
    bytes.withUnsafeBytes { raw in
      for offset in stride(from: 0, to: raw.count, by: 4) {
        values.append(Float(bitPattern: UInt32(littleEndian:
          raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))))
      }
    }
    let rows = Rows(video: Array(values.prefix(manifest.videoFloats)),
      audio: Array(values.suffix(manifest.audioFloats)))
    try rows.validate(geometry: manifest.geometry)
    // The pinned manifest must still be identical after reading its payload.
    guard sha(try read(manifestURL, limit: 1024 * 1024)) == expectedSHA256 else {
      throw H3CheckpointError.invalid("H3 full latent manifest changed while loading payload.")
    }
    return (manifest, rows)
  }
  /// Worker publication happens inside its private staging directory. Refuse
  /// existing files, so replay cannot overwrite source or previously saved rows.
  public static func save(rows: Rows, geometry: H3Geometry, task: String,
    componentIdentity: String, directory: URL) throws -> (manifestSHA256: String, payloadSHA256: String) {
    try rows.validate(geometry: geometry)
    guard ["t2va", "fl2va", "ref2va"].contains(task), validSHA(componentIdentity),
      directory.isFileURL, directory.path.hasPrefix("/"),
      !FileManager.default.fileExists(atPath: directory.path) else {
      throw H3CheckpointError.invalid("H3 full latent publication needs fresh staging and valid component identity.")
    }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    var published = false
    defer { if !published { try? FileManager.default.removeItem(at: directory) } }
    var payload = Data(capacity: (rows.video.count + rows.audio.count) * 4)
    for values in [rows.video, rows.audio] {
      for index in values.indices {
        if index % 65_536 == 0 { try Task.checkCancellation() }
        var bits = values[index].bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { payload.append(contentsOf: $0) }
      }
    }
    let digest = sha(payload)
    try payload.write(to: directory.appendingPathComponent("joint-latents.f32"), options: [.withoutOverwriting])
    let manifest = Manifest(format: geometry.canvasAdmission == .spatialRefinement
      ? "weetodd-h3-swift-joint-latents-v2-spatial" : "weetodd-h3-swift-joint-latents-v1", task: task,
      width: geometry.width, height: geometry.height, generatedFrames: geometry.frames,
      componentIdentity: componentIdentity, videoFloats: rows.video.count,
      audioFloats: rows.audio.count, payloadBytes: payload.count, payloadSHA256: digest)
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    let bytes = try encoder.encode(manifest)
    try bytes.write(to: directory.appendingPathComponent("manifest.json"), options: [.withoutOverwriting])
    try Task.checkCancellation(); published = true
    return (sha(bytes), digest)
  }
}
