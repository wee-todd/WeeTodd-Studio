import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class LoRAPromptMetadataInteractionTests: XCTestCase {
  private func directory() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    return root
  }
  private func seed(_ root: URL, key: String) throws {
    let values = [key: LoRAPromptMetadata(triggers: ["BUNNY"], notes: "Keep the contact readable.")]
    try JSONEncoder().encode(values).write(to: root.appendingPathComponent("lora-prompt-metadata.json"))
  }
  @MainActor func testNativeLibraryLoadsTriggersAndAppliesWithUndo() throws {
    let root = try directory()
    try seed(root, key: LoRAPromptKey.native("/tmp/weapon.safetensors"))
    let store = StudioStore(dataDirectory: root)
    var asset = MediaAsset(name: "Weapon", kind: .lora, path: "/tmp/weapon.safetensors"); asset.loraModel = .h3
    store.globalAssets = [asset]
    var clip = Clip(engine: .h3); clip.prompt = "A robot practices."
    store.project.clips = [clip]; store.selectedClipID = clip.id
    let before = store.project
    let candidate = try XCTUnwrap(store.compatibleLoRAs(for: .h3).first)
    XCTAssertEqual(candidate.loraPrompt?.notes, "Keep the contact readable.")
    store.applyLoRAMembers([LoRAMember(asset: candidate)])
    XCTAssertEqual(store.selectedClip?.prompt, "BUNNY\nA robot practices.")
    XCTAssertEqual(store.project.assets.last?.loraPrompt?.triggers, ["BUNNY"])
    store.undo(); XCTAssertEqual(store.project, before)
    store.redo(); XCTAssertEqual(store.selectedClip?.prompt, "BUNNY\nA robot practices.")
  }
  @MainActor func testDrawThingsLibraryTriggersApplyToClipAndImageDraft() async throws {
    let root = try directory()
    try seed(root, key: LoRAPromptKey.drawThings(profileID: "server", modelID: "style.ckpt"))
    let store = StudioStore(dataDirectory: root)
    var clip = Clip(engine: .drawThings); clip.prompt = "A robot practices."
    clip.drawThings = DrawThingsSelection(profileID: "server", modelID: "model", modelFamily: "h3")
    store.project.clips = [clip]; store.selectedClipID = clip.id
    await store.drawThingsDiscovery.load(DrawThingsConnection(id: "server")) {
      ["models": [["id": "model", "name": "Model", "family": "h3"]], "capabilities": [:],
        "loras": [["id": "style.ckpt", "name": "Style", "family": "h3", "compatibleModelIDs": ["model"]]]]
    }
    let before = store.project
    store.addDrawThingsLoRA(profileID: "server", modelID: "style.ckpt")
    XCTAssertEqual(store.selectedClip?.prompt, "BUNNY\nA robot practices.")
    XCTAssertEqual(store.selectedClip?.drawThings?.loras.first?.promptMetadata?.notes, "Keep the contact readable.")
    store.undo(); XCTAssertEqual(store.project, before)
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    draft.profileID = "server"; draft.modelID = "model"; draft.prompt = "A wooden robot."
    store.imageDraft = draft
    store.addDrawThingsLoRA(profileID: "server", modelID: "style.ckpt", imageWorkspace: true)
    XCTAssertEqual(store.imageDraft?.prompt, "BUNNY\nA wooden robot.")
    XCTAssertEqual(store.project, before)
    let wire = try store.imageDraft!.request(id: "image-test")
    let stack = try XCTUnwrap(wire["loras"] as? [[String: Any]])
    XCTAssertEqual(Set(stack[0].keys), ["modelID", "weight"])
  }
}

extension LoRAPromptMetadataInteractionTests {
  @MainActor func testLibraryEditsAffectNewApplicationsButSavedGroupsKeepTheirSnapshot() throws {
    let root = try directory(), store = StudioStore(dataDirectory: root, restoreSession: false)
    let key = LoRAPromptKey.native("/tmp/style.safetensors")
    XCTAssertTrue(store.saveLoRAPromptMetadata(.init(triggers: ["first"], notes: "First notes"), key: key))
    var asset = MediaAsset(name: "Style", kind: .lora, path: "/tmp/style.safetensors")
    asset.loraModel = .h3; asset.loraPrompt = .init(triggers: ["first"], notes: "First notes")
    var first = Clip(engine: .h3); first.prompt = "A robot."
    store.project.clips = [first]; store.selectedClipID = first.id
    store.applyLoRAMembers([LoRAMember(asset: asset)])
    let frozen = try XCTUnwrap(store.currentLoRAGroup())
    let applied = store.project
    XCTAssertTrue(store.saveLoRAPromptMetadata(.init(triggers: ["second"], notes: "Second notes"), key: key))
    XCTAssertEqual(store.project, applied)
    var next = Clip(engine: .h3); next.prompt = "A robot."
    store.project.clips.append(next); store.selectedClipID = next.id
    store.applyLoRAMembers([LoRAMember(asset: asset)])
    XCTAssertEqual(store.selectedClip?.prompt, "second\nA robot.")
    var third = Clip(engine: .h3); third.prompt = "A robot."
    store.project.clips.append(third); store.selectedClipID = third.id
    store.applyLoRAGroup(frozen)
    XCTAssertEqual(store.selectedClip?.prompt, "first\nA robot.")
    XCTAssertEqual(store.project.assets.last?.loraPrompt?.notes, "First notes")
    let restored = StudioStore(dataDirectory: root, restoreSession: false)
    restored.loadLoRAPromptMetadata()
    XCTAssertEqual(restored.loRAPromptMetadata(key: key).triggers, ["second"])
    XCTAssertTrue(restored.saveLoRAPromptMetadata(.init(), key: key))
    restored.loadLoRAPromptMetadata()
    XCTAssertTrue(restored.loRAPromptMetadata(key: key, fallback: asset.loraPrompt).triggers.isEmpty)
  }
  @MainActor func testDisabledNativeGroupInsertsOnEnableAndRemovalLeavesAuthoredText() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    var asset = MediaAsset(name: "Style", kind: .lora, path: "/tmp/style.safetensors")
    asset.loraModel = .h3; asset.loraPrompt = .init(triggers: ["BUNNY"])
    var clip = Clip(engine: .h3); clip.prompt = "A robot."
    store.project.clips = [clip]; store.selectedClipID = clip.id
    store.applyLoRAGroup(.init(name: "Muted", engine: .h3, members: [.init(asset: asset, enabled: false)]))
    XCTAssertEqual(store.selectedClip?.prompt, "A robot.")
    let id = try XCTUnwrap(store.selectedClip?.attachments.first?.id)
    store.setNativeLoRAEnabled(id, enabled: true)
    XCTAssertEqual(store.selectedClip?.prompt, "BUNNY\nA robot.")
    store.undo(); XCTAssertEqual(store.selectedClip?.prompt, "A robot.")
    store.redo(); store.setNativeLoRAEnabled(id, enabled: false)
    XCTAssertEqual(store.selectedClip?.prompt, "BUNNY\nA robot.")
    store.setNativeLoRAEnabled(id, enabled: true)
    XCTAssertEqual(store.selectedClip?.prompt, "BUNNY\nA robot.")
  }
  @MainActor func testManagedCharacterDraftRejectsPromptChangingLoRAAtomically() async throws {
    let root = try directory(), store = StudioStore(dataDirectory: root, restoreSession: false)
    let key = LoRAPromptKey.drawThings(profileID: "server", modelID: "style.ckpt")
    XCTAssertTrue(store.saveLoRAPromptMetadata(.init(triggers: ["BUNNY"]), key: key))
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    draft.profileID = "server"; draft.modelID = "model"; draft.prompt = "Reviewed character description."
    var sheet = ReferenceSheetContext(subjectKey: "character", name: "Character", kind: .character, description: "A robot.")
    sheet.template = .characterSheet; draft.referenceSheet = sheet; store.imageDraft = draft
    await store.drawThingsDiscovery.load(DrawThingsConnection(id: "server")) {
      ["models": [], "capabilities": [:], "loras": [["id": "style.ckpt", "name": "Style", "family": "h3", "compatibleModelIDs": ["model"]]]]
    }
    store.addDrawThingsLoRA(profileID: "server", modelID: "style.ckpt", imageWorkspace: true)
    XCTAssertEqual(store.imageDraft, draft)
    XCTAssertNotNil(store.error)
  }
}

extension LoRAPromptMetadataInteractionTests {
  @MainActor func testDrawThingsGroupsFreezeMetadataAndDuplicateFailureKeepsPrompt() async throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    let key = LoRAPromptKey.drawThings(profileID: "server", modelID: "style.ckpt")
    XCTAssertTrue(store.saveLoRAPromptMetadata(.init(triggers: ["first"]), key: key))
    var clip = Clip(engine: .drawThings); clip.prompt = "A robot."
    clip.drawThings = DrawThingsSelection(profileID: "server", modelID: "model", modelFamily: "h3")
    store.project.clips = [clip]; store.selectedClipID = clip.id
    await store.drawThingsDiscovery.load(DrawThingsConnection(id: "server")) {
      ["models": [], "capabilities": [:], "loras": [["id": "style.ckpt", "name": "Style", "family": "h3", "compatibleModelIDs": ["model"]]]]
    }
    store.addDrawThingsLoRA(profileID: "server", modelID: "style.ckpt")
    XCTAssertTrue(store.saveCurrentDrawThingsLoRAGroup(name: "Saved"))
    let group = try XCTUnwrap(store.drawThingsLoRAGroups.first)
    XCTAssertTrue(store.saveLoRAPromptMetadata(.init(triggers: ["second"]), key: key))
    let before = store.project
    store.applyDrawThingsLoRAGroup(group, mode: .add)
    XCTAssertEqual(store.project, before)
    var next = clip; next.id = UUID(); next.prompt = "A robot."
    store.project.clips.append(next); store.selectedClipID = next.id
    store.applyDrawThingsLoRAGroup(group, mode: .add)
    XCTAssertEqual(store.selectedClip?.prompt, "first\nA robot.")
    store.undo(); XCTAssertEqual(store.selectedClip?.prompt, "A robot.")
    store.redo(); XCTAssertEqual(store.selectedClip?.drawThings?.loras.first?.promptMetadata?.triggers, ["first"])
  }

  @MainActor func testFailedMetadataSaveDoesNotReplaceExistingLibraryState() throws {
    let root = try directory(), store = StudioStore(dataDirectory: root, restoreSession: false)
    let key = LoRAPromptKey.native("/tmp/style.safetensors")
    store.loraPromptLibrary[key] = .init(triggers: ["old"])
    try FileManager.default.createDirectory(at: root.appendingPathComponent("lora-prompt-metadata.json"), withIntermediateDirectories: true)
    XCTAssertFalse(store.saveLoRAPromptMetadata(.init(triggers: ["new"]), key: key))
    XCTAssertEqual(store.loraPromptLibrary[key]?.triggers, ["old"])
    XCTAssertNotNil(store.error)
  }
}

extension LoRAPromptMetadataInteractionTests {
  @MainActor func testInstalledH3PromptMetadataWhenRequested() async throws {
    let env = ProcessInfo.processInfo.environment
    guard let folderPath = env["WEETODD_H3_LORA_METADATA_DIR"], let fixturePath = env["WEETODD_H3_LORA_METADATA_FIXTURE"] else {
      throw XCTSkip("Opt-in installed H3 metadata/header qualification")
    }
    let root = try directory()
    try FileManager.default.copyItem(at: URL(fileURLWithPath: fixturePath), to: root.appendingPathComponent("lora-prompt-metadata.json"))
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.loadLoRAPromptMetadata()
    var folder = LoRAFolder(path: folderPath); folder.modelHint = .h3
    store.runtime.loraFolders = [folder]
    store.runtime.nativeH3Enabled = true; store.runtime.nativeLTX25Enabled = true
    store.addClip(); store.project.clips[0].engine = .h3
    let before = store.project
    await store.refreshLoRAFolders()
    XCTAssertEqual(store.folderLoRAEntries.count, 13, store.loraFolderWarnings.joined(separator: "\n"))
    for entry in store.folderLoRAEntries {
      XCTAssertFalse(store.loRAPromptMetadata(key: LoRAPromptKey.native(entry.path)).notes.isEmpty, entry.name)
    }
    let weapon = try XCTUnwrap(store.compatibleLoRAs(for: .h3).first { $0.name == "Bunny_weapon_combatV1" })
    XCTAssertEqual(weapon.loraPrompt?.triggers, ["BUNNY"])
    XCTAssertEqual(store.project, before)
    print("H3 metadata qualification:", store.folderLoRAEntries.count, "headers;", store.compatibleLoRAs(for: .h3).count, "library candidates")
  }
}

extension LoRAPromptMetadataInteractionTests {
  @MainActor func testDrawThingsPrefixIsDefaultAndAnExplicitEmptyOverrideWins() async throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    var clip = Clip(engine: .drawThings); clip.prompt = "A robot."
    clip.drawThings = DrawThingsSelection(profileID: "server", modelID: "model", modelFamily: "h3")
    store.project.clips = [clip]; store.selectedClipID = clip.id
    await store.drawThingsDiscovery.load(DrawThingsConnection(id: "server")) {
      ["models": [], "capabilities": [:], "loras": [["id": "style.ckpt", "name": "Style", "family": "h3", "compatibleModelIDs": ["model"], "promptPrefix": "cinematic glow"]]]
    }
    store.addDrawThingsLoRA(profileID: "server", modelID: "style.ckpt")
    XCTAssertEqual(store.selectedClip?.prompt, "cinematic glow\nA robot.")
    store.undo()
    let key = LoRAPromptKey.drawThings(profileID: "server", modelID: "style.ckpt")
    XCTAssertTrue(store.saveLoRAPromptMetadata(.init(), key: key))
    store.addDrawThingsLoRA(profileID: "server", modelID: "style.ckpt")
    XCTAssertEqual(store.selectedClip?.prompt, "A robot.")
  }
}

extension LoRAPromptMetadataInteractionTests {
  @MainActor func testExplicitEnableResolvesAndFreezesLegacyNativeMetadata() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    var asset = MediaAsset(name: "Weapon", kind: .lora, path: "/tmp/weapon.safetensors")
    asset.loraModel = .h3; asset.scope = .global; store.globalAssets = [asset]
    var clip = Clip(engine: .h3); clip.prompt = "A robot."
    var attachment = Attachment(assetID: asset.id, role: .lora); attachment.enabled = false
    clip.attachments = [attachment]; store.project.clips = [clip]; store.selectedClipID = clip.id
    let key = LoRAPromptKey.native(asset.path)
    XCTAssertTrue(store.saveLoRAPromptMetadata(.init(triggers: ["BUNNY"]), key: key))
    let before = store.project
    store.setNativeLoRAEnabled(attachment.id, enabled: true)
    XCTAssertEqual(store.selectedClip?.prompt, "BUNNY\nA robot.")
    let linked = try XCTUnwrap(store.allAssets.first { $0.id == store.selectedClip?.attachments.first?.assetID })
    XCTAssertEqual(linked.loraPrompt?.triggers, ["BUNNY"])
    XCTAssertNil(store.globalAssets[0].loraPrompt)
    store.undo(); XCTAssertEqual(store.project, before)
    store.redo()
    XCTAssertTrue(store.saveLoRAPromptMetadata(.init(triggers: ["changed"]), key: key))
    store.setNativeLoRAEnabled(attachment.id, enabled: false)
    store.setNativeLoRAEnabled(attachment.id, enabled: true)
    XCTAssertEqual(store.selectedClip?.prompt, "BUNNY\nA robot.")
  }
  @MainActor func testLegacyDrawThingsEnableResolvesMetadataAndPreservesExplicitEmpty() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    let key = LoRAPromptKey.drawThings(profileID: "server", modelID: "style.ckpt")
    XCTAssertTrue(store.saveLoRAPromptMetadata(.init(triggers: ["BUNNY"]), key: key))
    var clip = Clip(engine: .drawThings); clip.prompt = "A robot."
    clip.drawThings = .init(profileID: "server", modelID: "model", modelFamily: "h3",
      loras: [.init(modelID: "style.ckpt", enabled: false)])
    store.project.clips = [clip]; store.selectedClipID = clip.id
    store.setDrawThingsLoRAEnabled("style.ckpt", enabled: true)
    XCTAssertEqual(store.selectedClip?.prompt, "BUNNY\nA robot.")
    XCTAssertEqual(store.selectedClip?.drawThings?.loras[0].promptMetadata?.triggers, ["BUNNY"])
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    draft.profileID = "server"; draft.modelID = "model"; draft.prompt = "A robot."
    draft.loras = [.init(modelID: "style.ckpt", enabled: false)]; store.imageDraft = draft
    store.setDrawThingsLoRAEnabled("style.ckpt", enabled: true, imageWorkspace: true)
    XCTAssertEqual(store.imageDraft?.prompt, "BUNNY\nA robot.")
    draft.loras[0].promptMetadata = .init(); store.imageDraft = draft
    store.setDrawThingsLoRAEnabled("style.ckpt", enabled: true, imageWorkspace: true)
    XCTAssertEqual(store.imageDraft?.prompt, "A robot.")
    var sheet = ReferenceSheetContext(subjectKey: "character", name: "Character", kind: .character, description: "A robot.")
    sheet.template = .characterSheet; draft.referenceSheet = sheet
    draft.loras[0].promptMetadata = nil; store.imageDraft = draft
    store.setDrawThingsLoRAEnabled("style.ckpt", enabled: true, imageWorkspace: true)
    XCTAssertEqual(store.imageDraft, draft)
    XCTAssertNotNil(store.error)
  }
}
