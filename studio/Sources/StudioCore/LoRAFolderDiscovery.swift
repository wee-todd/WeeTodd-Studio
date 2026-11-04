import Foundation

/// Finds already linked libraries without walking model payloads or searching drives.
public enum LoRAFolderDiscovery {
  public struct Result {
    public var folders: [LoRAFolder]
    public var automaticFolders: [LoRAFolder]
    public var modelHints: [String: LoRAModel]
    public var warnings: [String]
  }
  public static func discover(folders: [LoRAFolder], profilesDirectory: String,
    assets: [MediaAsset], groups: [LoRAGroup], additionalPaths: [String] = [], enabled: Bool = true) throws -> Result {
    var result = Result(folders: folders, automaticFolders: [], modelHints: [:], warnings: [])
    guard enabled else { return result }
    let fm = FileManager.default
    func canonical(_ path: String) -> String {
      URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL.resolvingSymlinksInPath().path
    }
    var hints: [String: Set<LoRAModel>] = [:], roots = Set<String>()
    func link(_ path: String, model: LoRAModel?) {
      guard path.hasPrefix("/"), !path.utf8.contains(0), path.utf8.count <= 4096,
        URL(fileURLWithPath: path).pathExtension.lowercased() == "safetensors" else { return }
      let source = canonical(path)
      guard (try? URL(fileURLWithPath: source).resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { return }
      roots.insert(URL(fileURLWithPath: source).deletingLastPathComponent().path)
      if let model { hints[source, default: []].insert(model) }
    }
    for asset in assets where asset.kind == .lora { link(asset.path, model: asset.loraModel) }
    for group in groups { for member in group.members { link(member.asset.path, model: member.asset.loraModel) } }
    for path in additionalPaths { link(path, model: nil) }
    if !profilesDirectory.isEmpty, let files = try? fm.contentsOfDirectory(
      at: URL(fileURLWithPath: profilesDirectory), includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) {
      let profiles = files.filter { $0.pathExtension.lowercased() == "json" }.sorted { $0.path < $1.path }
      if profiles.count > 1000 { result.warnings.append("Linked-model discovery is limited to 1,000 setup files.") }
      for file in profiles.prefix(1000) {
        try Task.checkCancellation()
        guard let info = try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
          info.isRegularFile == true, let size = info.fileSize, (1...1024 * 1024).contains(size),
          let bytes = try? Data(contentsOf: file),
          let recipe = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any],
          recipe["format"] as? String == "weetodd-headless-v2",
          let model = (recipe["engine"] as? String).flatMap(LoRAModel.init(rawValue:)) else { continue }
        for section in [recipe["components"], recipe["config"]] {
          guard let values = section as? [String: Any] else { continue }
          for (key, value) in values where key == "loras" || key == "ic_loras" || key.hasSuffix("lora_path") {
            if let path = value as? String { link(path, model: model) }
            else if let pairs = value as? [[Any]] {
              for pair in pairs { if let path = pair.first as? String { link(path, model: model) } }
            } else if let adapters = value as? [[String: Any]] {
              for adapter in adapters { if let path = adapter["path"] as? String { link(path, model: model) } }
            }
          }
        }
        if let stack = (recipe["loras"] as? [String: Any])?["adapters"] as? [[String: Any]] {
          for adapter in stack { if let path = adapter["path"] as? String { link(path, model: model) } }
        }
      }
    }
    for path in hints.keys.sorted() {
      guard let models = hints[path] else { continue }
      guard result.modelHints.count < 1000 else {
        result.warnings.append("Linked-model hints are limited to 1,000 adapter files."); break
      }
      if models.count == 1 { result.modelHints[path] = models.first }
      else if models.isSubset(of: [.ltx23, .ltx25]) { result.modelHints[path] = .ltx23 }
    }
    let configured = folders.map { (canonical($0.path), $0) }
    for root in roots.sorted() {
      // Explicit exclusions and nonrecursive parents remain authoritative.
      if configured.contains(where: { parent, _ in root == parent || root.hasPrefix(parent + "/") }) { continue }
      // A previously added automatic parent already covers this child.
      if result.automaticFolders.contains(where: { root.hasPrefix($0.path + "/") }) { continue }
      guard result.folders.count < 64 else {
        result.warnings.append("Automatic LoRA discovery reached the 64-folder limit."); break
      }
      let folder = LoRAFolder(path: root)
      result.folders.append(folder); result.automaticFolders.append(folder)
    }
    return result
  }
}
