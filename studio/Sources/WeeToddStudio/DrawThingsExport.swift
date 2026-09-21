import AppKit
import StudioCore

@MainActor extension StudioStore {
  func exportDrawThingsImageJob() {
    guard let draft = imageDraft else { return }
    let connection = drawThingsConnections.first(where: { $0.id == draft.profileID })
    let panel = NSSavePanel()
    panel.title = "Export Image Headless Job"
    panel.nameFieldStringValue = draft.name + ".weetodd-job.json"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task {
      do {
        var imageProject = StudioProject()
        imageProject.name = draft.name
        let payload = try await imagePayload(draft, connection: connection)
        var body: [String: Any] = ["project": try imageProject.object(), "generateIDs": [], "globalAssets": []]
        var entry: [String: Any] = ["id": UUID().uuidString, "kind": "image", "dependsOn": []]
        if draft.executionProvider == .nativeMLX {
          entry["request"] = payload["nativeImageRequest"]; body["nativeImageJobs"] = [entry]
        } else {
          entry["request"] = payload["drawThingsRequest"]; entry["connection"] = payload["connection"]
          body["drawThingsImageJobs"] = [entry]
        }
        _ = try await bridge.invoke("export-job", runtime: runtime, payload: body, output: url)
        notice = "Exported an image job for WeeToddCLI. You can close Studio before running it."
        NSWorkspace.shared.activateFileViewerSelecting([url])
      } catch { self.error = error.localizedDescription }
    }
  }
}
