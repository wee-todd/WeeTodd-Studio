import Foundation
import StudioCore

extension StudioStore {
  /// Apply catalog defaults only to the active sheet, never to the ordinary image workspace.
  func configureCharacterSheetAdapter(_ draft: inout DrawThingsImageDraft) {
    guard draft.referenceSheet?.template == .characterSheet,
      draft.executionProvider == .drawThings, draft.characterSheetLoRAID == nil else { return }
    if draft.characterSheetModelPending == true,
      let catalog = drawThingsCatalogs[draft.profileID] {
      let installed = (catalog["loras"] as? [[String: Any]] ?? []).filter {
        let name = ($0["name"] as? String ?? "").lowercased().filter { $0.isLetter || $0.isNumber }
        return name.contains("krea2characterdesign4viewv1")
      }
      if installed.count == 1, let compatible = installed[0]["compatibleModelIDs"] as? [String] {
        let models = drawThingsModels(draft.profileID, operation: "image").filter {
          compatible.contains($0.id) && $0.name.lowercased().filter { $0.isLetter || $0.isNumber }.contains("krea2turbo")
        }
        if models.count == 1 {
          draft.modelID = models[0].id; draft.steps = 8; draft.guidance = 1
          draft.sampler = nil; draft.shift = nil
        }
      }
      // An existing model remains a valid explicit starting point when the preferred pair is absent.
      if !draft.modelID.isEmpty { draft.characterSheetModelPending = false }
    }
    guard !draft.modelID.isEmpty else { return }
    let candidates = drawThingsLoRAs(profileID: draft.profileID, modelID: draft.modelID).filter {
      ($0.name + " " + $0.id).range(of: #"(?i)(?:4|four)[\s_-]*(?:panel|view)"#,
        options: .regularExpression) != nil
    }
    let preferred = candidates.filter {
      $0.name.lowercased().filter { $0.isLetter || $0.isNumber }.contains("krea2characterdesign4viewv1")
    }
    let matches = preferred.count == 1 ? preferred : candidates
    guard matches.count == 1, let candidate = matches.first else { return }
    draft.characterSheetLoRAID = candidate.id
    if !draft.loras.contains(where: { $0.modelID == candidate.id }) {
      draft.loras.append(DrawThingsLoRA(modelID: candidate.id))
    }
  }
  func configureCharacterSheetAdapter() {
    guard var draft = imageDraft else { return }
    configureCharacterSheetAdapter(&draft)
    if draft != imageDraft { imageDraft = draft; imageEstimate = nil }
  }
  func selectCharacterSheetAdapter(_ id: String) {
    guard var draft = imageDraft, draft.referenceSheet?.template == .characterSheet,
      drawThingsLoRAs(profileID: draft.profileID, modelID: draft.modelID).contains(where: { $0.id == id }) else { return }
    if let previous = draft.characterSheetLoRAID, previous != id {
      draft.loras.removeAll { $0.modelID == previous }
    }
    draft.characterSheetLoRAID = id
    if let index = draft.loras.firstIndex(where: { $0.modelID == id }) {
      draft.loras[index].enabled = true
      if draft.loras[index].weight == 0 { draft.loras[index].weight = 1 }
    } else { draft.loras.append(DrawThingsLoRA(modelID: id)) }
    imageDraft = draft; imageEstimate = nil
  }
  func characterSheetIssue(_ draft: DrawThingsImageDraft) -> String? {
    guard draft.referenceSheet?.template == .characterSheet else { return nil }
    if let issue = draft.characterSheetRequestIssue { return issue }
    guard let id = draft.characterSheetLoRAID,
      drawThingsLoRAs(profileID: draft.profileID, modelID: draft.modelID).contains(where: { $0.id == id }) else {
      return "Refresh the Draw Things catalog and choose a compatible four-panel LoRA for this model."
    }
    return nil
  }
}
