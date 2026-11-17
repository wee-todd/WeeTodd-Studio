import Foundation

/// Locates shared official components without copying or loading model weights.
public enum MOSSModelSetup {
  public static func discover(root: URL, variant: String) -> MOSSImageSettings? {
    guard ["base", "sft"].contains(variant) else { return nil }
    let fm = FileManager.default
    func exists(_ url: URL, _ name: String) -> Bool { fm.fileExists(atPath: url.appendingPathComponent(name).path) }
    for family in [root, root.appendingPathComponent("WeeTodd/MOSS-Image"), root.appendingPathComponent("MOSS-Image")] {
      let shared = [family.appendingPathComponent("Shared"), family.appendingPathComponent("moss-image-shared")]
        .first { exists($0, "text_encoder/config.json") && exists($0, "tokenizer/tokenizer.json") && exists($0, "vae/config.json") }
      let transformer = [family.appendingPathComponent(variant == "sft" ? "SFT/transformer" : "Base/transformer"),
        family.appendingPathComponent("moss-image-" + variant + "/transformer")].first { exists($0, "config.json") }
      if let shared, let transformer {
        var result = MOSSImageSettings()
        result.libraryDirectory = family.path; result.sharedDirectory = shared.path
        result.transformerDirectory = transformer.path; result.variant = variant
        return result
      }
    }
    return nil
  }

  public static func prepare(root: URL, variant: String, catalog: URL,
    progress: @escaping NativeModelDownloads.Progress) async throws -> MOSSImageSettings {
    guard ["base", "sft"].contains(variant) else { throw StudioError.invalid("Choose MOSS Base or SFT.") }
    let packages = try NativeModelDownloads.catalog(at: catalog)
    let ids = ["moss-image-shared", "moss-image-" + variant]
    // Verify existing complete installations before using them. Native preflight
    // separately checks tensor shapes and binds their file identities.
    if let found = discover(root: root, variant: variant) {
      var complete = true
      for id in ids {
        guard let package = packages.first(where: { $0.descriptor.id == id }) else { throw StudioError.invalid("MOSS download catalog is missing.") }
        let folder = URL(fileURLWithPath: id == ids[0] ? found.sharedDirectory : found.transformerDirectory).deletingLastPathComponent()
        for file in package.files {
          let candidate = id == ids[0] ? URL(fileURLWithPath: found.sharedDirectory).appendingPathComponent(file.target) : folder.appendingPathComponent(file.target)
          if try !NativeModelDownloads.verified(candidate, file: file) { complete = false; break }
        }
      }
      if complete { progress("Verified installed MOSS components", 1); return found }
    }
    for (index, id) in ids.enumerated() {
      let final = root.appendingPathComponent(id)
      if FileManager.default.fileExists(atPath: final.path) {
        guard let package = packages.first(where: { $0.descriptor.id == id }) else { throw StudioError.invalid("MOSS download catalog is missing.") }
        for file in package.files {
          guard try NativeModelDownloads.verified(final.appendingPathComponent(file.target), file: file) else {
            throw StudioError.invalid("The existing MOSS package is incomplete or changed. Choose a new family folder; setup can reuse verified files from this one.")
          }
        }
      } else {
        _ = try await NativeModelDownloads.prepare(id: id, catalog: catalog, destination: root,
          existingRoots: [root.path], progress: { message, fraction in progress(message, (Double(index) + fraction) / 2) })
      }
    }
    guard let result = discover(root: root, variant: variant) else { throw StudioError.invalid("MOSS setup did not produce complete component folders.") }
    return result
  }
}
