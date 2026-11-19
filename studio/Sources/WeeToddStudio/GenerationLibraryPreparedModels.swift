import CryptoKit
import Foundation
import StudioCore

enum GenerationLibraryPreparedModels {
  /// Mirrors the existing prepared Qwen contract without loading tensors or invoking Python.
  static func qwenFiles(_ manifest: URL) throws -> [String] {
    let raw = try GenerationLibraryModelImport.json(manifest), root = manifest.deletingLastPathComponent().resolvingSymlinksInPath()
    guard raw["format"] as? String == "weetodd-qwen-image21-v1", raw["modelID"] as? String == "Qwen/Qwen-Image-2.1",
      raw["sourceRevision"] as? String == "b3179ad355be050328e483a9dfdd9e60cd62adfa",
      raw["bits"] as? Int == 8, raw["groupSize"] as? Int == 64,
      let components = raw["components"] as? [String: [String: Any]], Set(components.keys) == ["transformer", "text_encoder", "vae"] else {
      throw StudioError.invalid("Choose the qualified prepared Qwen Image 2.1 8-bit, group-64 package.")
    }
    var files = [manifest.path]
    func item(_ value: [String: Any], folder: URL, direct: Bool, verifyHash: Bool) throws {
      guard let name = value["name"] as? String, !name.hasPrefix("/"), let bytes = value["bytes"] as? Int, bytes > 0,
        let sha = value["sha256"] as? String, sha.count == 64 else { throw StudioError.invalid("Invalid prepared Qwen component identity.") }
      let file = folder.appendingPathComponent(name).standardizedFileURL.resolvingSymlinksInPath()
      guard file.path.hasPrefix(root.path + "/"), !direct || file.deletingLastPathComponent() == folder,
        let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]), values.isRegularFile == true, values.fileSize == bytes else {
        throw StudioError.invalid("Missing or changed Qwen component: " + name)
      }
      if verifyHash {
        guard bytes <= 16 * 1024 * 1024, SHA256.hash(data: try Data(contentsOf: file)).map({ String(format: "%02x", $0) }).joined() == sha else {
          throw StudioError.invalid("Changed Qwen model metadata: " + name)
        }
      }
      files.append(file.path)
    }
    for name in components.keys.sorted() {
      let folder = root.appendingPathComponent(name).resolvingSymlinksInPath()
      guard folder.deletingLastPathComponent() == root, let shards = components[name]?["files"] as? [[String: Any]], !shards.isEmpty, shards.count <= 1024 else {
        throw StudioError.invalid("Missing prepared Qwen checkpoint shards.")
      }
      for shard in shards { try item(shard, folder: folder, direct: true, verifyHash: false) }
      let config = folder.appendingPathComponent("config.json"), content = try GenerationLibraryModelImport.json(config)
      let required = name == "transformer" ? ["num_layers": 32, "in_channels": 64, "context_in_dim": 4096] : name == "vae" ? ["in_channels": 4, "out_channels": 4, "z_dim": 64] : [:]
      guard required.allSatisfy({ content[$0.key] as? Int == $0.value }) else { throw StudioError.invalid("Incompatible prepared Qwen architecture.") }
      files.append(config.path)
    }
    for name in ["processor/tokenizer.json", "processor/preprocessor_config.json", "scheduler/scheduler_config.json"] {
      let file = root.appendingPathComponent(name)
      guard FileManager.default.isReadableFile(atPath: file.path) else { throw StudioError.invalid("Missing Qwen processor or scheduler.") }; files.append(file.path)
    }
    guard let metadata = raw["metadata"] as? [[String: Any]], metadata.count <= 1024 else { throw StudioError.invalid("Missing bounded Qwen metadata identities.") }
    for value in metadata { try item(value, folder: root, direct: false, verifyHash: true) }
    if let preview = raw["preview"] as? [String: Any] { try item(preview, folder: root, direct: true, verifyHash: true) }
    return files
  }
}
