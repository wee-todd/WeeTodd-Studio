import CryptoKit
import Foundation

/// Bind checked image settings to the installed component files. As in native
/// video preflight, component stamps describe size, inode and modification time;
/// inference independently validates checkpoint headers and unchanged reads.
public enum H3ImageIdentity {
  public static func fingerprint(recipe: [String: Any], components: [URL]) throws -> String {
    var recipe = recipe
    var fields = recipe["still_image"] as? [String: Any] ?? [:]
    fields.removeValue(forKey: "expected_fingerprint")
    recipe["still_image"] = fields
    var files = Set<URL>()
    let manager = FileManager.default
    for source in components {
      let source = source.standardizedFileURL.resolvingSymlinksInPath()
      let values = try source.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey])
      if values.isRegularFile == true { files.insert(source) }
      else if values.isDirectory == true,
        let entries = manager.enumerator(at: source, includingPropertiesForKeys: [.isRegularFileKey],
          options: [.skipsHiddenFiles]) {
        var count = 0
        for case let entry as URL in entries {
          count += 1
          guard count <= 4000 else { throw H3CheckpointError.invalid("H3 component directory exceeds the bounded image preflight.") }
          if try entry.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true { files.insert(entry) }
        }
      } else { throw H3CheckpointError.invalid("Relink the selected H3 image components.") }
    }
    let records = try files.sorted { $0.path < $1.path }.map { url -> [String: Any] in
      let values = try manager.attributesOfItem(atPath: url.path)
      guard let size = values[.size] as? NSNumber,
        let modified = values[.modificationDate] as? Date,
        let inode = values[.systemFileNumber] as? NSNumber else {
        throw H3CheckpointError.invalid("Cannot bind an H3 image component to preflight.")
      }
      return ["path": url.path, "bytes": size.uint64Value,
        "modified": modified.timeIntervalSince1970, "inode": inode.uint64Value]
    }
    let data = try JSONSerialization.data(withJSONObject: ["recipe": recipe, "components": records], options: [.sortedKeys])
    return "sha256:" + SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
}
