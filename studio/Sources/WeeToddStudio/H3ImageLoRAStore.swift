import Foundation
import StudioCore

@MainActor extension StudioStore {
  func selectImageProvider(_ provider: ImageExecutionProvider) {
    guard imageDraft != nil else { return }
    imageDraft?.selectProvider(provider)
    if provider == .nativeH3 {
      if imageDraft?.modelID.isEmpty == true, let profile = h3ImageProfiles.first {
        selectH3ImageProfile(profile.id)
      }
      if imageDraft?.h3Image?.imageVAEPath.isEmpty == true {
        imageDraft?.h3Image?.imageVAEPath = h3ImageVAEPath
      }
    }
    imageEstimate = nil
  }
  var h3ImageProfiles: [ModelProfile] {
    profiles.filter { $0.engine == "h3" && ["t2v", "fflf", "ref2va"].contains($0.task)
      && $0.generation?.vdn != true && $0.generation?.fasth3 != true && $0.generation?.fasth3Omni != true }
  }
  func selectH3ImageProfile(_ id: String) {
    imageDraft?.modelID = id
    if let evaluations = h3ImageProfiles.first(where: { $0.id == id })?.generation?.controls.evaluations {
      imageDraft?.steps = evaluations
    }
    imageEstimate = nil
  }
  func applyH3ImageLoRAMembers(_ members: [LoRAMember], preservingSnapshots: Bool = false,
    mode: LoRAGroupApplicationMode = .add) {
    guard var draft = imageDraft, draft.executionProvider == .nativeH3 else { return }
    do {
      let incoming = resolvedLoRAMembers(members, preservingSnapshots: preservingSnapshots)
      for member in incoming { try member.validate(for: .h3) }
      let current = draft.h3Image?.loras ?? []
      let combined = mode == .replace ? incoming : current + incoming
      guard combined.count <= 8, Set(combined.map(\.fileKey)).count == combined.count else {
        throw StudioError.invalid("Use at most eight distinct H3 LoRAs. Remove duplicates or replace the stack.")
      }
      try insertLoRAPromptGuidance(incoming.filter(\.isEnabled).compactMap { $0.asset.loraPrompt }, into: &draft)
      if draft.h3Image == nil { draft.h3Image = H3ImageSettings() }
      draft.h3Image?.loras = combined
      imageDraft = draft; imageEstimate = nil
    } catch { self.error = error.localizedDescription }
  }
  func setH3ImageLoRAEnabled(_ id: UUID, enabled: Bool) {
    guard var draft = imageDraft, draft.executionProvider == .nativeH3,
      let index = draft.h3Image?.loras.firstIndex(where: { $0.id == id }) else { return }
    do {
      if enabled, let member = draft.h3Image?.loras[index] {
        let resolved = resolvedLoRAMembers([member], preservingSnapshots: true)[0]
        try insertLoRAPromptGuidance(resolved.asset.loraPrompt.map { [$0] } ?? [], into: &draft)
        draft.h3Image?.loras[index] = resolved
      }
      draft.h3Image?.loras[index].enabled = enabled
      imageDraft = draft; imageEstimate = nil
    } catch { self.error = error.localizedDescription }
  }
}
