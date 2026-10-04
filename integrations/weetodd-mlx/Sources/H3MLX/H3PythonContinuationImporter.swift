import CoreFoundation
import CryptoKit
import Darwin
import Foundation

/// Explicit, bounded import of the owned Python v1 format. This validates full
/// content identities once per import; it is not a per-step/model cache. The
/// current native v2 run route remains separate until its import bridge is used.
public enum H3PythonContinuationImporter {
  public struct Artifact: Sendable {
    public let rows: H3Continuation.Rows
    public let identityJSON: Data
    public let provenanceJSON: Data
    public let payloadSHA256: String
  }
  private static let manifestLimit = 1_048_576
  private static let payloadLimit = 64 * 1_048_576

  private static func fail(_ message: String) -> H3CheckpointError {
    .invalid("Python H3 context: " + message)
  }
  private static func object(_ value: Any?) throws -> [String: Any] {
    guard let value = value as? [String: Any] else { throw fail("missing object") }
    return value
  }
  private static func integer(_ value: Any?) throws -> Int {
    guard let number = value as? NSNumber,
      CFGetTypeID(number) != CFBooleanGetTypeID(),
      number.doubleValue.isFinite, number.doubleValue.rounded() == number.doubleValue,
      number.doubleValue >= Double(Int.min), number.doubleValue < Double(Int.max) else {
      throw fail("invalid integer")
    }
    return number.intValue
  }
  private static func hash(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
  private static func validHash(_ value: Any?) throws -> String {
    guard let value = value as? String, value.utf8.count == 64,
      value.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
      throw fail("invalid content digest")
    }
    return value
  }
  private static func regular(_ url: URL) throws -> [FileAttributeKey: Any] {
    guard url.isFileURL, url.path.hasPrefix("/"),
      try url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true else {
      throw fail("source must be a regular local file")
    }
    var status = stat()
    guard lstat(url.path, &status) == 0, status.st_mode & S_IFMT == S_IFREG else {
      throw fail("source must be a regular local file")
    }
    var value = try FileManager.default.attributesOfItem(atPath: url.path)
    value[FileAttributeKey(rawValue: "h3-ctime-seconds")] = NSNumber(value: status.st_ctimespec.tv_sec)
    value[FileAttributeKey(rawValue: "h3-ctime-nanoseconds")] = NSNumber(value: status.st_ctimespec.tv_nsec)
    value[FileAttributeKey(rawValue: "h3-mtime-nanoseconds")] = NSNumber(value: status.st_mtimespec.tv_nsec)
    guard value[.type] as? FileAttributeType == .typeRegular else {
      throw fail("source must be a regular local file")
    }
    return value
  }
  private static func sameStat(_ lhs: [FileAttributeKey: Any],
    _ rhs: [FileAttributeKey: Any]) -> Bool {
    [.size, .modificationDate, .creationDate, .systemNumber, .systemFileNumber,
      FileAttributeKey(rawValue: "h3-ctime-seconds"), FileAttributeKey(rawValue: "h3-ctime-nanoseconds"),
      FileAttributeKey(rawValue: "h3-mtime-nanoseconds")]
      .allSatisfy { key in
        guard let a = lhs[key] as? NSObject, let b = rhs[key] as? NSObject else { return false }
        return a.isEqual(b)
      }
  }
  private static func bounded(_ url: URL, limit: Int) throws -> Data {
    let before = try regular(url)
    guard let size = before[.size] as? NSNumber, size.intValue <= limit else {
      throw fail("source exceeds its size bound")
    }
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    try Task.checkCancellation()
    let data = try file.read(upToCount: limit + 1) ?? Data()
    guard data.count == size.intValue, data.count <= limit,
      sameStat(before, try regular(url)) else { throw fail("source changed while reading") }
    return data
  }
  /// Historical model-index publication uses a symlink to upstream metadata.
  /// This exception is only for that index: component weights remain regular
  /// canonical files. Bind the link text/inode/timestamps and resolved target.
  fileprivate struct ModelIndexAlias: Equatable {
    let url: URL, target: URL, linkText: String
    let identity: [String]
    init(_ url: URL) throws {
      var status = stat()
      guard lstat(url.path, &status) == 0, status.st_mode & S_IFMT == S_IFLNK else {
        throw fail("model-index alias changed type or disappeared")
      }
      self.url = url
      identity = [String(status.st_dev), String(status.st_ino), String(status.st_size),
        String(status.st_mtimespec.tv_sec), String(status.st_mtimespec.tv_nsec),
        String(status.st_ctimespec.tv_sec), String(status.st_ctimespec.tv_nsec)]
      linkText = try FileManager.default.destinationOfSymbolicLink(atPath: url.path)
      target = URL(fileURLWithPath: try canonicalPath(url))
      _ = try regular(target)
    }
  }
  private static func verifyModelIndex(_ url: URL, record: [String: Any], session: ImportSession?) throws {
    var status = stat()
    guard lstat(url.path, &status) == 0 else { throw fail("missing model index") }
    if status.st_mode & S_IFMT == S_IFREG {
      try verifyFile(url, record: record, session: session)
    } else {
      let before = try ModelIndexAlias(url)
      try session?.pin(before)
      try verifyFile(before.target, record: record, session: session)
      guard try ModelIndexAlias(url) == before else { throw fail("model-index alias changed while reading") }
    }
  }

  /// Explicit serialized import lease. A full content hash is reusable only
  /// within this session, for the identical canonical file and declaration.
  /// Every hit and every publication boundary rechecks inode/size/mtime/ctime;
  /// a changed source fails the batch rather than refreshing its cached hash.
  public final class ImportSession {
    private struct Verified {
      let url: URL, bytes: Int, sha256: String
      let identity: [FileAttributeKey: Any]
    }
    private var verified: [String: Verified] = [:]
    private var aliases: [String: ModelIndexAlias] = [:]
    fileprivate func pin(_ alias: ModelIndexAlias) throws {
      if let prior = aliases[alias.url.path], prior != alias {
        throw fail("model-index alias changed within the explicit import session")
      }
      aliases[alias.url.path] = alias
    }
    public private(set) var filesHashed = 0
    public private(set) var bytesHashed: UInt64 = 0
    public private(set) var hashReuses = 0
    public private(set) var bytesReuseValidated: UInt64 = 0
    public init() {}
    fileprivate func reuse(_ url: URL, bytes: Int, sha256: String,
      identity: [FileAttributeKey: Any]) throws -> Bool {
      let key = try canonicalPath(url)
      guard let value = verified[key] else { return false }
      guard value.bytes == bytes, value.sha256 == sha256,
        sameStat(value.identity, identity), sameStat(value.identity, try regular(url)) else {
        throw fail("verified component changed within the explicit import session")
      }
      hashReuses += 1; bytesReuseValidated += UInt64(bytes)
      return true
    }
    fileprivate func remember(_ url: URL, bytes: Int, sha256: String,
      identity: [FileAttributeKey: Any]) throws {
      verified[try canonicalPath(url)] = Verified(url: url, bytes: bytes, sha256: sha256, identity: identity)
      filesHashed += 1; bytesHashed += UInt64(bytes)
    }
    public func checkUnchanged() throws {
      try Task.checkCancellation()
      for alias in aliases.values {
        guard try ModelIndexAlias(alias.url) == alias else { throw fail("model-index alias changed at the explicit import boundary") }
      }
      for value in verified.values {
        try Task.checkCancellation()
        guard sameStat(value.identity, try regular(value.url)) else {
          throw fail("verified component changed at the explicit import boundary")
        }
      }
    }
  }
  private static func verifyFile(_ url: URL, record: [String: Any], session: ImportSession? = nil) throws {
    let expectedBytes = try integer(record["bytes"])
    let expectedHash = try validHash(record["sha256"])
    guard expectedBytes >= 0 else { throw fail("negative source size") }
    let before = try regular(url)
    guard (before[.size] as? NSNumber)?.intValue == expectedBytes else {
      throw fail("component size changed")
    }
    if try session?.reuse(url, bytes: expectedBytes, sha256: expectedHash, identity: before) == true { return }
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var digest = SHA256(), count = 0
    while true {
      try Task.checkCancellation()
      let bytes = try file.read(upToCount: 1_048_576) ?? Data()
      if bytes.isEmpty { break }
      count += bytes.count
      guard count <= expectedBytes else { throw fail("component grew during import") }
      digest.update(data: bytes)
    }
    let actual = digest.finalize().map { String(format: "%02x", $0) }.joined()
    guard count == expectedBytes, actual == expectedHash,
      sameStat(before, try regular(url)) else { throw fail("component content changed") }
    try session?.remember(url, bytes: count, sha256: actual, identity: before)
  }
  // Darwin realpath matches Python Path.resolve. Foundation's URL resolver
  // canonicalizes /private/var back to /var while directory enumeration emits
  // /private/var; mixing those spellings corrupts relative inventory names.
  private static func canonicalPath(_ url: URL) throws -> String {
    guard let resolved = realpath(url.path, nil) else { throw fail("missing identity source") }
    defer { free(resolved) }
    return String(cString: resolved)
  }
  private static func location(_ value: Any?) throws -> URL {
    guard let value = value as? String, value.hasPrefix("/"), !value.contains("://") else {
      throw fail("invalid identity path")
    }
    let url = URL(fileURLWithPath: value)
    guard try canonicalPath(url) == value else {
      throw fail("identity paths must already be canonical")
    }
    return url
  }
  private static func verifyComponent(_ record: [String: Any], session: ImportSession?) throws {
    let root = try location(record["path"])
    let files = try object(record["files"])
    guard !files.isEmpty, files.count <= 10_000 else { throw fail("component inventory bound") }
    var directory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: root.path, isDirectory: &directory) else {
      throw fail("missing component")
    }
    if directory.boolValue {
      guard let enumerator = FileManager.default.enumerator(at: root,
        includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey], options: []) else {
        throw fail("cannot enumerate component")
      }
      var actual = Set<String>(), visited = 0
      for case let url as URL in enumerator {
        visited += 1
        guard visited <= 20_000 else { throw fail("component traversal bound") }
        let attributes = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard attributes.isSymbolicLink != true else { throw fail("nested symlink") }
        if attributes.isRegularFile == true {
          let fullPath = try canonicalPath(url)
          guard fullPath.hasPrefix(root.path + "/") else { throw fail("component traversal escaped root") }
          actual.insert(String(fullPath.dropFirst(root.path.count + 1)))
        }
      }
      guard actual == Set(files.keys) else { throw fail("component inventory changed") }
    } else {
      // DT SQLite/sidecar context import is not a native H3 weight-layout claim.
      guard Set(files.keys) == [root.lastPathComponent] else {
        throw fail("unsupported multi-file component")
      }
    }
    for (name, value) in files {
      guard !name.hasPrefix("/"), !name.split(separator: "/").contains(".."),
        !name.split(separator: "/").contains("."), !name.isEmpty else {
        throw fail("unsafe component-relative path")
      }
      try verifyFile(directory.boolValue ? root.appendingPathComponent(name) : root,
        record: object(value), session: session)
    }
    if let references = record["referenced_files"] {
      guard try object(references).isEmpty else {
        throw fail("converted/DT external component references are not admitted")
      }
    }
  }

  private static func verifyIdentity(_ identity: [String: Any], request: H3T2VARequest,
    task: String, session: ImportSession?) throws {
    guard try integer(identity["version"]) == 1,
      identity["engine"] as? String == "h3", ["t2va", "fl2va"].contains(task),
      request.funControl == nil,
      identity["task"] as? String == task,
      try integer(identity["fps"]) == 24,
      try integer(identity["sample_rate"]) == 32_000,
      try integer(identity["audio_channels"]) == 2,
      try integer(identity["width"]) == request.geometry.width,
      try integer(identity["height"]) == request.geometry.height,
      identity["schedule"] as? String == "native_h3_shifted_flow_v1",
      identity["block_residency"] as? String == "checkpoint_default" else {
      throw fail("task, timing, canvas or residency differs")
    }
    let sampling = try object(identity["sampling"])
    let keys: Set<String> = ["width", "height", "steps", "drop_adaln", "memory_mode",
      "attention_chunk_size", "attention_head_chunk_size", "ffn_row_chunk_size",
      "projection_backend", "transformer_backend", "sampling_method",
      "inference_optimization", "paging_cache_gb"]
    // Real v1 artifacts predate the explicit backend field; at that time
    // the owned generator's backend was MLX. No other omitted key is defaulted.
    guard Set(sampling.keys) == keys || Set(sampling.keys) == keys.subtracting(["transformer_backend"]) else {
      throw fail("sampling dictionary has missing or unknown settings")
    }
    guard
      try integer(sampling["width"]) == request.geometry.width,
      try integer(sampling["height"]) == request.geometry.height,
      try integer(sampling["steps"]) == request.requestedSteps,
      (sampling["drop_adaln"] as? NSNumber).map({ CFGetTypeID($0) == CFBooleanGetTypeID() && $0.boolValue }) == true,
      sampling["memory_mode"] as? String == (request.videoDecodeMemoryMode?.rawValue ?? "normal"),
      sampling["attention_chunk_size"] as? String == "automatic",
      ["automatic", "disabled"].contains(sampling["attention_head_chunk_size"] as? String ?? ""),
      sampling["ffn_row_chunk_size"] as? String == "automatic",
      ["auto", "mlx"].contains(sampling["projection_backend"] as? String ?? ""),
      (sampling["transformer_backend"] == nil || sampling["transformer_backend"] as? String == "mlx"),
      sampling["sampling_method"] as? String == request.samplingMethod.rawValue,
      sampling["inference_optimization"] as? String == "off",
      try integer(sampling["paging_cache_gb"]) == 0 else {
      throw fail("sampling settings are incompatible with this native request")
    }
    let components = try object(identity["components"])
    guard Set(components.keys) == Set(["transformer", "text_encoder", "processor",
      "tokenizer", "video_vae", "audio_vae"]) else { throw fail("missing component inventory") }
    let mapped = ["transformer": request.transformer, "text_encoder": request.qwenPages,
      "video_vae": request.videoVAE, "audio_vae": request.audioVAE]
    for (name, url) in mapped {
      guard try location(object(components[name])["path"]).path ==
        canonicalPath(url) else {
        throw fail("native component path differs: " + name)
      }
    }
    let tokenizer = try object(components["tokenizer"])
    let tokenizerRoot = try location(tokenizer["path"])
    let tokenizerPath = try canonicalPath(request.tokenizer)
    let tokenizerFiles = try object(tokenizer["files"])
    let tokenizerMatches = tokenizerPath == tokenizerRoot.path ||
      (tokenizerPath.hasPrefix(tokenizerRoot.path + "/") &&
        tokenizerFiles[String(tokenizerPath.dropFirst(tokenizerRoot.path.count + 1))] != nil)
    guard tokenizerMatches else { throw fail("native tokenizer differs") }
    let adapters = identity["loras"] as? [[String: Any]]
    guard let adapters, adapters.count == request.loRAAdapters.count else { throw fail("adapter stack differs") }
    for (adapter, native) in zip(adapters, request.loRAAdapters) {
      let settings = try object(adapter["settings"])
      guard try location(settings["path"]).path == canonicalPath(native.url),
        (settings["strength"] as? NSNumber).map({ CFGetTypeID($0) != CFBooleanGetTypeID() && $0.floatValue == native.strength }) == true,
        ["auto", "standard", "turbo"].contains(settings["profile"] as? String ?? ""),
        (native.profile == .auto || settings["profile"] as? String == native.profile.rawValue),
        settings["adaln_input_grid"] is NSNull,
        adapter["adaln_input_grid"] is NSNull,
        settings["qkv_layout"] as? String == native.qkvLayout.rawValue,
        try integer(settings["start_after_evaluations"]) == native.startAfterEvaluations else {
        throw fail("adapter settings differ or are unsupported")
      }
      _ = try H3LoRAFile(url: native.url, strength: native.strength,
        requestedSteps: request.requestedSteps, samplingMethod: request.samplingMethod,
        qkvLayout: native.qkvLayout, profile: native.profile, startAfterEvaluations: native.startAfterEvaluations)
      if settings["profile"] as? String == "turbo" {
        guard request.samplingMethod == .euler, request.requestedSteps == 5 else {
          throw fail("explicit Turbo needs Euler/four evaluations")
        }
      }
      try verifyFile(native.url, record: object(adapter["content"]), session: session)
    }
    let checkpoint = try location(identity["checkpoint"])
    try verifyModelIndex(checkpoint.appendingPathComponent("model_index.json"),
      record: object(identity["model_index"]), session: session)
    if !(identity["text_architecture_config"] is NSNull) {
      let config = try object(identity["text_architecture_config"])
      try verifyFile(location(config["path"]), record: object(config["content"]), session: session)
    }
    for value in components.values { try verifyComponent(object(value), session: session) }
  }

  public static func load(manifestURL: URL, expectedSHA256: String,
    expectedIdentityJSON: Data, request: H3T2VARequest, task: String,
    contextFrames: Int, session: ImportSession? = nil) throws -> Artifact {
    guard H3Continuation.allowedContextFrames.contains(contextFrames),
      expectedIdentityJSON.count <= manifestLimit / 2 else { throw fail("invalid import bounds") }
    let bytes = try bounded(manifestURL, limit: manifestLimit)
    guard hash(bytes) == (try validHash(expectedSHA256)) else { throw fail("manifest digest changed") }
    let manifest = try object(JSONSerialization.jsonObject(with: bytes))
    let identity = try object(manifest["identity"])
    let expected = try object(JSONSerialization.jsonObject(with: expectedIdentityJSON))
    guard manifest["format"] as? String == "weetodd-h3-continuation-v1",
      try integer(manifest["context_frames"]) == contextFrames,
      NSDictionary(dictionary: identity).isEqual(to: expected) else {
      throw fail("manifest identity or context changed")
    }
    let provenance = try object(manifest["provenance"])
    let generated = try integer(provenance["generated_frames"])
    let published = try integer(provenance["published_frames"])
    let overlap = try integer(provenance["overlap_frames"] ?? 0)
    guard try integer(provenance["tail_trim_frames"]) == 0,
      published > contextFrames, (6...362).contains(generated), generated % 17 == 5,
      overlap == 0 || H3Continuation.allowedContextFrames.contains(overlap),
      generated - overlap == published else { throw fail("trimmed or invalid tail provenance") }
    let payload = try object(manifest["payload"])
    guard payload["file"] as? String == "latents.safetensors" else { throw fail("unsafe payload path") }
    let payloadData = try bounded(manifestURL.deletingLastPathComponent()
      .appendingPathComponent("latents.safetensors"), limit: payloadLimit)
    guard payloadData.count == (try integer(payload["bytes"])), payloadData.count > 8,
      hash(payloadData) == (try validHash(payload["sha256"])) else { throw fail("payload digest/size changed") }
    let headerBytes = payloadData.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
    guard (1...65_536).contains(headerBytes), headerBytes + 8 <= UInt64(payloadData.count) else { throw fail("invalid tensor header") }
    let base = 8 + Int(headerBytes)
    let header = try object(JSONSerialization.jsonObject(with: payloadData.subdata(in: 8..<base)))
    guard Set(header.keys).subtracting(["__metadata__"]) == ["video", "audio"] else { throw fail("unexpected payload tensors") }
    let f = (contextFrames - 5) / 17 * 5 + 2
    let h = request.geometry.height / 16, w = request.geometry.width / 16
    let a = Int((Double(contextFrames) / 24 * 40).rounded(.toNearestOrEven))
    let shapes = ["video": [1, 24, f, h, w], "audio": [2, 32, a]]
    let descriptors = try object(payload["tensors"])
    guard Set(descriptors.keys) == Set(shapes.keys) else { throw fail("manifest tensor inventory changed") }
    var offsets: [(Int, Int)] = [], info: [String: (Int, String)] = [:]
    for (name, shape) in shapes {
      let tensor = try object(header[name])
      let declared = try object(descriptors[name])
      guard NSDictionary(dictionary: tensor).isEqual(to: declared),
        let dimensions = tensor["shape"] as? [Any],
        try dimensions.map({ try integer($0) }) == shape,
        let dtype = tensor["dtype"] as? String, ["F16", "BF16", "F32"].contains(dtype),
        let range = tensor["data_offsets"] as? [Any], range.count == 2 else { throw fail("tensor shape/dtype changed") }
      let start = try integer(range[0]), stop = try integer(range[1])
      let count = shape.reduce(1, *)
      guard start >= 0, stop >= start, stop - start == count * (dtype == "F32" ? 4 : 2),
        stop <= payloadData.count - base else { throw fail("tensor offsets changed") }
      offsets.append((start, stop)); info[name] = (base + start, dtype)
    }
    offsets.sort { $0.0 < $1.0 }
    guard offsets[0].0 == 0, offsets[0].1 == offsets[1].0,
      offsets[1].1 == payloadData.count - base else { throw fail("tensor ranges do not cover payload") }
    // Full legacy identity verification is explicit and precedes allocation of
    // native rows. A caller must not replace it with untrusted SHA/stat JSON.
    try verifyIdentity(identity, request: request, task: task, session: session)
    let rows = try payloadData.withUnsafeBytes { buffer -> H3Continuation.Rows in
      func value(_ name: String, _ index: Int) throws -> Float {
        let (offset, dtype) = info[name]!
        let result: Float
        if dtype == "F32" {
          result = Float(bitPattern: UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: offset + index * 4, as: UInt32.self)))
        } else {
          let word = UInt16(littleEndian: buffer.loadUnaligned(fromByteOffset: offset + index * 2, as: UInt16.self))
          result = dtype == "BF16" ? Float(bitPattern: UInt32(word) << 16) : Float(Float16(bitPattern: word))
        }
        guard result.isFinite else { throw fail("nonfinite latent") }
        return result
      }
      var video = [Float](); video.reserveCapacity(24 * f * h * w)
      for frame in 0..<f {
        try Task.checkCancellation()
        for y in stride(from: 0, to: h, by: 2) {
          for x in stride(from: 0, to: w, by: 2) {
            for channel in 0..<24 { for dy in 0..<2 { for dx in 0..<2 {
              video.append(try value("video", ((channel * f + frame) * h + y + dy) * w + x + dx))
            } } }
          }
        }
      }
      var audio = [Float](); audio.reserveCapacity(2 * a * 32)
      for channel in 0..<2 { for time in 0..<a {
        try Task.checkCancellation()
        for latent in 0..<32 { audio.append(try value("audio", (channel * 32 + latent) * a + time)) }
      } }
      return H3Continuation.Rows(video: video, audio: audio)
    }
    return Artifact(rows: rows,
      identityJSON: try JSONSerialization.data(withJSONObject: identity, options: [.sortedKeys]),
      provenanceJSON: try JSONSerialization.data(withJSONObject: provenance, options: [.sortedKeys]),
      payloadSHA256: try validHash(payload["sha256"]))
  }
  public struct Publication: Sendable {
    public let manifestURL: URL
    public let manifestSHA256: String
    public let payloadSHA256: String
    public let originReceiptSHA256: String
  }

  /// Publish a fresh native artifact after explicit import. Legacy data stays
  /// intact. Native identity and legacy content identity remain separate and
  /// the SHA-bound origin receipt records the conversion's proof level.
  public static func importToNative(manifestURL: URL, expectedSHA256: String,
    expectedIdentityJSON: Data, request: H3T2VARequest, task: String,
    contextFrames: Int, output: URL, session: ImportSession? = nil) throws -> Publication {
    guard task == "t2va" else { throw fail("T2VA import needs a T2VA request") }
    return try publish(manifestURL: manifestURL, expectedSHA256: expectedSHA256,
      expectedIdentityJSON: expectedIdentityJSON, request: request, task: task,
      contextFrames: contextFrames, output: output, session: session,
      fingerprint: { try H3Continuation.fingerprint(request) })
  }

  /// FL legacy identity must already cover the exact native vision component.
  /// An unrelated compatible BF16 checkpoint cannot replace a legacy Q8 source.
  public static func importToNative(manifestURL: URL, expectedSHA256: String,
    expectedIdentityJSON: Data, request: H3FL2VARequest,
    contextFrames: Int, output: URL, session: ImportSession? = nil) throws -> Publication {
    let identity = try object(JSONSerialization.jsonObject(with: expectedIdentityJSON))
    let components = try object(identity["components"])
    let text = try object(components["text_encoder"])
    let root = try location(text["path"])
    let vision = try canonicalPath(request.vision)
    let files = try object(text["files"])
    guard vision == root.path || (vision.hasPrefix(root.path + "/") &&
      files.keys.contains(where: { key in
        let covered = root.appendingPathComponent(key).path
        return covered == vision || covered.hasPrefix(vision + "/")
      })) else { throw fail("legacy text inventory does not bind the native vision component") }
    return try publish(manifestURL: manifestURL, expectedSHA256: expectedSHA256,
      expectedIdentityJSON: expectedIdentityJSON, request: request.base, task: "fl2va",
      contextFrames: contextFrames, output: output, session: session,
      fingerprint: { try H3Continuation.fingerprint(request) })
  }

  private static func publish(manifestURL: URL, expectedSHA256: String,
    expectedIdentityJSON: Data, request: H3T2VARequest, task: String,
    contextFrames: Int, output: URL, session: ImportSession?, fingerprint: () throws -> String) throws -> Publication {
    guard ["t2va", "fl2va"].contains(task), output.isFileURL, output.path.hasPrefix("/"),
      !FileManager.default.fileExists(atPath: output.path) else {
      throw fail("native context publication requires a fresh destination")
    }
    let nativeIdentity = try fingerprint()
    try session?.checkUnchanged()
    let imported = try load(manifestURL: manifestURL, expectedSHA256: expectedSHA256,
      expectedIdentityJSON: expectedIdentityJSON, request: request, task: task,
      contextFrames: contextFrames, session: session)
    try session?.checkUnchanged()
    guard try fingerprint() == nativeIdentity else {
      throw fail("native components changed during import")
    }
    let provenance = try object(JSONSerialization.jsonObject(with: imported.provenanceJSON))
    var payload = Data(capacity: (imported.rows.video.count + imported.rows.audio.count) * 4)
    for values in [imported.rows.video, imported.rows.audio] {
      for (index, value) in values.enumerated() {
        if index.isMultiple(of: 65_536) { try Task.checkCancellation() }
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { payload.append(contentsOf: $0) }
      }
    }
    guard payload.count <= payloadLimit else { throw fail("native payload bound") }
    let payloadSHA = hash(payload)
    let origin: [String: Any] = ["format": "weetodd-h3-python-context-import-v1",
      "legacyManifest": manifestURL.path, "legacyManifestSHA256": expectedSHA256,
      "legacyPayloadSHA256": imported.payloadSHA256,
      "legacyIdentity": try object(JSONSerialization.jsonObject(with: imported.identityJSON)),
      "legacyProvenance": provenance, "nativeIdentity": nativeIdentity,
      "normalizationApplied": false, "latentValueConversionLossless": true,
      "crossRuntimeGenerationParityQualified": false]
    let originData = try JSONSerialization.data(withJSONObject: origin, options: [.sortedKeys])
    let originSHA = hash(originData)
    var manifest: [String: Any] = ["format": "weetodd-h3-swift-continuation-v2",
      "contextFrames": contextFrames, "width": request.geometry.width,
      "height": request.geometry.height,
      "generatedFrames": try integer(provenance["generated_frames"]),
      "publishedFrames": try integer(provenance["published_frames"]),
      "overlapFrames": try integer(provenance["overlap_frames"] ?? 0),
      "identity": nativeIdentity, "payloadBytes": payload.count,
      "payloadSHA256": payloadSHA, "pythonV1OriginReceiptSHA256": originSHA]
    if task != "t2va" { manifest["task"] = task }
    let manifestData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
    let manifestSHA = hash(manifestData)
    let staging = output.deletingLastPathComponent()
      .appendingPathComponent(".h3-context-import-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: staging) }
    try payload.write(to: staging.appendingPathComponent("latents.f32"), options: .withoutOverwriting)
    try originData.write(to: staging.appendingPathComponent("python-v1-origin.json"), options: .withoutOverwriting)
    let stagedManifest = staging.appendingPathComponent("manifest.json")
    try manifestData.write(to: stagedManifest, options: .withoutOverwriting)
    let checked = try H3Continuation.load(manifestURL: stagedManifest,
      expectedSHA256: manifestSHA, contextFrames: contextFrames,
      width: request.geometry.width, height: request.geometry.height, identity: nativeIdentity, task: task)
    guard checked?.video.map(\.bitPattern) == imported.rows.video.map(\.bitPattern),
      checked?.audio.map(\.bitPattern) == imported.rows.audio.map(\.bitPattern) else {
      throw fail("native context loader verification failed")
    }
    try Task.checkCancellation()
    try session?.checkUnchanged()
    guard try fingerprint() == nativeIdentity else {
      throw fail("native components changed before publication")
    }
    try FileManager.default.moveItem(at: staging, to: output)
    return Publication(manifestURL: output.appendingPathComponent("manifest.json"),
      manifestSHA256: manifestSHA, payloadSHA256: payloadSHA, originReceiptSHA256: originSHA)
  }

}
