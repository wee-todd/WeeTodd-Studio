import AppKit
import Foundation
import StudioCore

@MainActor extension StudioStore {
  func loadH3ImageModels() {
    let file = dataDirectory.appendingPathComponent("h3-image-models.json")
    guard FileManager.default.fileExists(atPath:file.path) else { return }
    do {
      let values = try JSONDecoder().decode([String:String].self,from:Data(contentsOf:file))
      h3ImageVAEPath = values["imageVAEPath"] ?? ""
    } catch { self.error = "Could not read H3 image model links: \(error.localizedDescription)" }
  }
  func saveH3ImageVAEPath(_ path: String, applyToDraft: Bool) throws {
    guard path.hasPrefix("/"), !path.utf8.contains(0) else { throw StudioError.invalid("Choose an absolute image VAE path.") }
    try FileManager.default.createDirectory(at:dataDirectory,withIntermediateDirectories:true)
    try JSONEncoder().encode(["imageVAEPath":path]).write(
      to:dataDirectory.appendingPathComponent("h3-image-models.json"),options:.atomic)
    h3ImageVAEPath = path
    if applyToDraft, imageDraft?.executionProvider == .nativeH3 {
      if imageDraft?.h3Image == nil { imageDraft?.h3Image = H3ImageSettings() }
      imageDraft?.h3Image?.imageVAEPath = path; imageEstimate = nil
    }
  }
  func prepareH3ImageVAE() async {
    guard !operationBusy, !preparingImageRequest else { return }
    let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
    panel.prompt = "Prepare here"
    panel.message = "Choose AI model storage for the H3 image VAE. Verified existing files are reused; allow about 9 GiB for first setup."
    if !h3ImageVAEPath.isEmpty { panel.directoryURL = URL(fileURLWithPath:h3ImageVAEPath).deletingLastPathComponent() }
    guard panel.runModal() == .OK, let folder = panel.url else { return }
    let session = documentSessionID, key = imageDraft?.storageKey
    do {
      let result = try await bridge.invoke("image-h3-vae-prepare",runtime:runtime,payload:["directory":folder.path])
      guard let path = result["imageVAEPath"] as? String else { throw StudioError.invalid("H3 image VAE preparation returned no checkpoint.") }
      try saveH3ImageVAEPath(path,applyToDraft:session == documentSessionID && key == imageDraft?.storageKey)
      notice = "H3 image-only VAE prepared at \(path)."
    } catch { self.error = error.localizedDescription }
  }
}
