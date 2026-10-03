import Foundation

/// A bounded in-place index; every generation revalidates enabled adapters.
public enum NativeLoRAFolderScan {
  public static func scan(_ folders: [LoRAFolder], maximumFiles: Int = 1000,
    maximumVisits: Int = 20_000) throws -> [String: Any] {
    guard folders.count <= 64, (1...1000).contains(maximumFiles), (1...20_000).contains(maximumVisits) else {
      throw StudioError.invalid("Choose up to 64 LoRA folders within the bounded scan limits.")
    }
    let fm = FileManager.default
    var entries: [[String: Any]] = [], warnings: [String] = [], seen = Set<String>()
    var visits = 0, skippedDrawThings = 0, limited = false
    for folder in folders where folder.enabled {
      if limited { break }
      guard !folder.path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
      let expanded = (folder.path as NSString).expandingTildeInPath
      let root = URL(fileURLWithPath: expanded).standardizedFileURL.resolvingSymlinksInPath()
      var isDirectory: ObjCBool = false
      guard fm.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        warnings.append("Cannot read LoRA folder: \(root.path)."); continue
      }
      var options: FileManager.DirectoryEnumerationOptions = [.skipsHiddenFiles]
      if !folder.recursive { options.insert(.skipsSubdirectoryDescendants) }
      guard let iterator = fm.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey],
        options: options, errorHandler: { url, error in
          if warnings.count < 100 { warnings.append("Cannot read \(url.path): \(error.localizedDescription)") }; return true
        }) else { warnings.append("Cannot read LoRA folder: \(root.path)."); continue }
      while let url = iterator.nextObject() as? URL {
        visits += 1
        if visits > maximumVisits || entries.count >= maximumFiles { limited = true; break }
        let properties = try? url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey])
        let source = url.standardizedFileURL.resolvingSymlinksInPath()
        var targetDirectory: ObjCBool = false
        guard fm.fileExists(atPath: source.path, isDirectory: &targetDirectory) else { continue }
        if targetDirectory.boolValue {
          if properties?.isSymbolicLink == true { iterator.skipDescendants() }
          continue
        }
        if url.pathExtension.lowercased() == "ckpt" { skippedDrawThings += 1; continue }
        guard url.pathExtension.lowercased() == "safetensors", seen.insert(source.path).inserted else { continue }
        var entry: [String: Any] = ["name": source.deletingPathExtension().lastPathComponent,
          "path": source.path, "sourceFolder": root.path]
        do {
          let info = try NativeLoRAInspection.inspect(source, modelHint: folder.modelHint)
          entry.merge(info) { _, new in new }
          entry["status"] = info["loraModel"] == nil ? "needsModel" : "ready"
        } catch {
          let message = String(error.localizedDescription.prefix(500))
          entry["status"] = message.contains("recipe") ? "specialized" : "unsupported"
          entry["detail"] = message
        }
        entries.append(entry)
      }
    }
    if limited { warnings.append("Scan limit reached. Choose smaller LoRA folders or disable subfolders.") }
    if skippedDrawThings > 0 {
      warnings.append("Skipped \(skippedDrawThings) .ckpt files. Browse installed Draw Things LoRAs through its connection; native generation needs compatible SafeTensors adapters.")
    }
    entries.sort { ($0["path"] as? String ?? "").lowercased() < ($1["path"] as? String ?? "").lowercased() }
    return ["entries": entries, "warnings": Array(warnings.prefix(100))]
  }
}
