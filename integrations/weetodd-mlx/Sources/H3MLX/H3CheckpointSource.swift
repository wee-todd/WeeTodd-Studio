import CoreFoundation
import Darwin
import Foundation
import TensorIO

/// Header-only routing for owned mixed-affine FL and pinned FastH3 pages. No weights are
/// copied or retained; each existing weighted operation opens its active page.
enum H3CheckpointSource {
  private struct Identity: Equatable {
    let device: dev_t, inode: ino_t, bytes: off_t
    let mtime: Int, mtimeNS: Int, ctime: Int, ctimeNS: Int
    init(_ url: URL, directory: Bool = false) throws {
      var s = stat()
      guard lstat(url.path, &s) == 0, s.st_mode & S_IFMT == (directory ? S_IFDIR : S_IFREG) else {
        throw H3CheckpointError.invalid("A paged H3 source is missing, nonregular or a symlink.")
      }
      device = s.st_dev; inode = s.st_ino; bytes = s.st_size
      mtime = s.st_mtimespec.tv_sec; mtimeNS = s.st_mtimespec.tv_nsec
      ctime = s.st_ctimespec.tv_sec; ctimeNS = s.st_ctimespec.tv_nsec
    }
  }
  private struct Page: Decodable {
    let file: String, sha256: String
    let tensor_count: Int
    let tensor_bytes: UInt64
  }
  private struct Manifest: Decodable {
    let format: String, source: String
    let num_blocks: Int
    let source_tensor_bytes: UInt64
    let fixed: Page
    let blocks: [Page]
    let source_revision: String?
    let attention: String?
    let sampling: [String: Int]?
  }
  final class Paged {
    let root: URL
    let layout: H3CheckpointLayout
    let fixed: URL
    let blocks: [URL]
    private let identities: [(URL, Identity)]
    private let directoryIdentities: [(URL, Identity)]

    fileprivate init(root: URL) throws {
      try Task.checkCancellation()
      self.root = root
      let directories = [root, root.appendingPathComponent("pages")]
      directoryIdentities = try directories.map { ($0, try Identity($0, directory: true)) }
      var pinned: [(URL, Identity)] = []
      func json(_ name: String) throws -> Data {
        let url = root.appendingPathComponent(name), before = try Identity(url)
        guard (1...1_048_576).contains(before.bytes) else {
          throw H3CheckpointError.invalid("Paged H3 metadata exceeds its bounded size.")
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: 1_048_577) ?? Data()
        guard data.count == before.bytes, try Identity(url) == before else {
          throw H3CheckpointError.invalid("Paged H3 metadata changed while reading.")
        }
        pinned.append((url, before)); return data
      }
      let manifest = try JSONDecoder().decode(Manifest.self, from: json("paged_manifest.json"))
      let config = try JSONSerialization.jsonObject(with: json("config.json")) as? [String: Any]
      let quant = try JSONSerialization.jsonObject(with: json("quant_config.json")) as? [String: Any]
      func number(_ object: [String: Any]?, _ key: String, equals value: Double) -> Bool {
        guard let n = object?[key] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return false }
        return n.doubleValue == value
      }
      func falseBoolean(_ object: [String: Any]?, _ key: String) -> Bool {
        guard let n = object?[key] as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() else { return false }
        return !n.boolValue
      }
      let fast: H3FastVariant?
      switch manifest.source {
      case "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree": fast = .denseV1
      case "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree": fast = .vsaV1
      default: fast = nil
      }
      var expected = Set<String>()
      if let fast {
        let vsa = fast == .vsaV1
        guard manifest.format == "weetodd-h3-paged-v1", manifest.num_blocks == 50,
          manifest.blocks.count == 50,
          manifest.source_revision == (vsa ? "b65818d41939b5085451074fe8ca8b799f8d4921" : "f624f08c6c279ab43534c003e556fc5b295b6558"),
          manifest.source_tensor_bytes == (vsa ? 70_099_582_760 : 66_246_059_536),
          manifest.attention == (vsa ? "vsa_h3_64_90" : "dense"),
          manifest.sampling == ["schedule_points":5,"transformer_evaluations":4],
          Set(quant?.keys.map { $0 } ?? []) == ["bits","group_size","quantize_core","quantize_adaln","adaln_bits","overrides"],
          (quant?["overrides"] as? [String:Any])?.isEmpty == true,
          number(quant,"bits",equals:8), number(quant,"group_size",equals:64), number(quant,"adaln_bits",equals:8),
          (quant?["quantize_core"] as? NSNumber).map({ CFGetTypeID($0) == CFBooleanGetTypeID() && $0.boolValue }) == true,
          (quant?["quantize_adaln"] as? NSNumber).map({ CFGetTypeID($0) == CFBooleanGetTypeID() && $0.boolValue }) == true,
          number(config,"hidden_size",equals:5376),number(config,"num_layers",equals:50),
          number(config,"token_refiner_num_layers",equals:2),number(config,"num_attention_heads",equals:56),
          number(config,"attention_head_dim",equals:128),number(config,"ffn_hidden_size",equals:14336),
          number(config,"latents_dim",equals:24),number(config,"audio_latents_dim",equals:32),
          (config?["patch_size"] as? [Int]) == [1,2,2],number(config,"text_dim",equals:5120),
          number(config,"timestep_input_dim",equals:256),number(config,"time_embed_hidden_size",equals:5376),
          number(config,"time_embed_dim",equals:2688),number(config,"rope_inv_freq_len",equals:16),
          number(config,"rope_theta",equals:10000),number(config,"norm_eps",equals:1e-5),
          number(config,"qk_norm_eps",equals:1e-5),number(config,"final_norm_eps",equals:1e-5),
          vsa ? (config?["vsa_gate"] as? Bool) == true : config?["vsa_gate"] == nil else {
          throw H3CheckpointError.invalid("Unsupported FastH3 release, architecture or affine recipe.")
        }
        for index in 0..<50 {
          for suffix in ["attn.qkv_proj","attn.out_proj","mlp.fc1","mlp.fc2","adaln_proj.linear"] {
            expected.insert("blocks.\(index).\(suffix)")
          }
        }
        for index in 0..<2 {
          for suffix in ["attn.qkv_proj","attn.out_proj","mlp.fc1","mlp.fc2"] {
            expected.insert("token_refiner.blocks.\(index).\(suffix)")
          }
        }
      } else {
      guard manifest.format == "weetodd-h3-paged-v1", manifest.num_blocks == 50,
        manifest.blocks.count == 50, manifest.source_tensor_bytes <= 64 * 1024 * 1024 * 1024, ["q8_extended", "q8_conservative"].contains(manifest.source),
        quant?["format"] as? String == "minimax-h3-mlx-mixed-quant",
        number(quant, "format_version", equals: 1), number(quant, "bits", equals: 8),
        number(quant, "group_size", equals: 64),
        falseBoolean(quant, "quantize_core"), falseBoolean(quant, "quantize_adaln"),
        quant?["profile"] as? String == manifest.source,
        number(config, "hidden_size", equals: 5376), number(config, "num_layers", equals: 50),
        number(config, "num_attention_heads", equals: 56), number(config, "attention_head_dim", equals: 128),
        number(config, "ffn_hidden_size", equals: 14336), number(config, "time_embed_dim", equals: 64),
        number(config, "adaln_curve_grid", equals: 1001), number(config, "rope_inv_freq_len", equals: 16),
        number(config, "rope_theta", equals: 10_000),
        let overrides = quant?["overrides"] as? [String: Any] else {
        throw H3CheckpointError.invalid("Unsupported paged H3 architecture or quantization recipe.")
      }
      for index in 0..<50 {
        for suffix in ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"] {
          if index >= 38 || (manifest.source == "q8_extended" && index >= 21 && suffix.hasPrefix("mlp.")) {
            expected.insert("blocks.\(index).\(suffix)")
          }
        }
      }
      guard Set(overrides.keys) == expected, overrides.keys.allSatisfy({ number(overrides, $0, equals: 8) }) else {
        throw H3CheckpointError.invalid("Paged H3 quantization overrides do not match its named profile.")
      }
      }
      var headers: [String: H3TensorInfo] = [:], bytes: UInt64 = 0
      func page(_ record: Page, name: String, index: Int?) throws -> URL {
        try Task.checkCancellation()
        guard record.file == name, record.sha256.utf8.count == 64,
          record.sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
          throw H3CheckpointError.invalid("Paged H3 has an invalid page identity or slot.")
        }
        let url = root.appendingPathComponent(name), before = try Identity(url)
        let file = try SafeTensorFile(url: url, maximumHeaderBytes: 1_048_576)
        let tensorBytes = file.tensors.values.reduce(UInt64(0)) { $0 + $1.byteCount }
        let expectedCount: Int
        if let index {
          let affine = expected.filter { $0.hasPrefix("blocks.\(index).") }.count
          expectedCount = fast == nil ? 10 + 2 * affine : (fast == .vsaV1 ? 21 : 20)
        } else { expectedCount = fast == nil ? 31 : 50 }
        guard file.tensors.count == expectedCount, record.tensor_count == expectedCount,
          tensorBytes == record.tensor_bytes else {
          throw H3CheckpointError.invalid("Paged H3 tensor count or payload bytes differ from its manifest.")
        }
        for (name, tensor) in file.tensors {
          guard headers[name] == nil,
            index.map({ name.hasPrefix("blocks.\($0).") }) ?? !name.hasPrefix("blocks.") else {
            throw H3CheckpointError.invalid("A paged H3 tensor occupies the wrong slot or is duplicated.")
          }
          headers[name] = H3TensorInfo(dtype: tensor.dtype, shape: tensor.shape)
        }
        try file.checkUnchanged(at: url)
        guard try Identity(url) == before else { throw H3CheckpointError.invalid("Paged H3 page changed during admission.") }
        pinned.append((url, before)); bytes += tensorBytes
        return url
      }
      fixed = try page(manifest.fixed, name: "pages/fixed.safetensors", index: nil)
      blocks = try manifest.blocks.enumerated().map {
        try page($0.element, name: String(format: "pages/block-%03d.safetensors", $0.offset), index: $0.offset)
      }
      guard (fast == nil ? bytes == manifest.source_tensor_bytes : bytes == (fast == .vsaV1 ? 39_120_840_192 : 35_267_323_392)),
        Set(headers.filter { $0.value.dtype == "U32" }.keys) == Set(expected.map { $0 + ".weight" }) else {
        throw H3CheckpointError.invalid("Paged H3 affine payloads differ from the quantization recipe.")
      }
      layout = try H3CheckpointLayout(tensors: headers, pagedAffine: true, computedRotary: true, fastVariant: fast)
      identities = pinned
      try checkUnchanged()
    }
    func checkUnchanged() throws {
      try Task.checkCancellation()
      for (url, identity) in directoryIdentities {
        guard try Identity(url, directory: true) == identity else { throw H3CheckpointError.invalid("Paged H3 directory changed after admission.") }
      }
      for (url, identity) in identities {
        guard try Identity(url) == identity else { throw H3CheckpointError.invalid("Paged H3 source changed after admission.") }
      }
    }
  }
  private final class Cache: @unchecked Sendable {
    let lock = NSRecursiveLock()
    var sources: [String: Paged] = [:]
  }
  private static let cache = Cache()
  static func isPaged(_ url: URL) -> Bool {
    // An admitted directory cannot silently become a single-file checkpoint
    // during a stage, even if the replacement file has compatible shapes.
    cache.lock.lock()
    let known = cache.sources[url.standardizedFileURL.path] != nil
    cache.lock.unlock()
    if known { return true }
    var directory: ObjCBool = false
    return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) && directory.boolValue
  }
  static func inspect(_ root: URL) throws -> Paged {
    cache.lock.lock(); defer { cache.lock.unlock() }
    if let source = cache.sources[root.standardizedFileURL.path] {
      try source.checkUnchanged(); return source
    }
    let source = try Paged(root: root)
    // Metadata only, bounded to the process's most recent component set.
    if cache.sources.count >= 4 { cache.sources.removeAll() }
    cache.sources[root.standardizedFileURL.path] = source
    return source
  }
  static func fileURL(_ root: URL, block: Int? = nil) throws -> URL {
    guard isPaged(root) else { return root }
    let source = try inspect(root)
    if let block {
      guard (0..<50).contains(block) else { throw H3CheckpointError.invalid("Invalid paged H3 block slot.") }
      return source.blocks[block]
    }
    return source.fixed
  }
  static func checkUnchanged(_ root: URL) throws {
    if isPaged(root) { try inspect(root).checkUnchanged() }
  }
}
