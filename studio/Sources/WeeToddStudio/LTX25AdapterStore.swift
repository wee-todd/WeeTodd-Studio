import Foundation
import StudioCore

@MainActor extension StudioStore {
  @discardableResult func selectLTX25Adapter(_ entry: LoRAFolderEntry) async -> Bool {
    guard !operationBusy, !ltx25AdapterSelectionBusy, let clip = selectedClip,
      clip.engine == .ltx25, runtime.usesNativeLTX25, let kind = entry.adapterKind else {
      error = "Select a native LTX 2.5 clip to use this adapter."; return false
    }
    let session = documentSessionID
    ltx25AdapterSelectionBusy = true
    defer { ltx25AdapterSelectionBusy = false }
    do {
      let path = entry.path
      let binding = try await Task.detached(priority: .userInitiated) {
        try NativeModelSetup.adapterBinding(at: URL(fileURLWithPath: path), kind: kind)
      }.value
      guard !Task.isCancelled, documentSessionID == session, selectedClip == clip,
        runtime.usesNativeLTX25, !operationBusy else {
        throw StudioError.invalid("The selected clip changed. Select the adapter again for the current shot.")
      }
      var updated = clip
      try updated.applyLTX25Config(.adapter(binding), in: project)
      editClip { $0 = updated }
      select(updated.id)
      generationInputsFocusRequest = UUID()
      error = nil
      notice = "Selected \(kind.label). Add its required inputs in the clip inspector."
      return true
    } catch { self.error = error.localizedDescription; return false }
  }
}
