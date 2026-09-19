import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class LoRAFolderTests: XCTestCase {
  private func directory() throws -> URL {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
    return folder
  }

  @MainActor func testFolderPreferencesDoNotInvalidateGenerationInputs() throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    let clip = Clip(engine: .ltx25)
    store.project.clips = [clip]
    store.selectedClipID = clip.id
    let before = store.generationRequestKey(for: clip)
    store.runtime.loraFolders = [LoRAFolder(path: "/tmp/new-library")]
    XCTAssertEqual(store.generationRequestKey(for: clip), before)
    store.runtime.pythonPath = "/tmp/different-python"
    XCTAssertNotEqual(store.generationRequestKey(for: clip), before)
  }

  @MainActor func testLegacySettingsReceiveDefaultFolderAndExplicitEmptyListIsPreserved() throws {
    let root = try directory()
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    XCTAssertEqual(store.loraFolders.map(\.path), [root.appendingPathComponent("Models/LoRAs").path])
    var runtime = store.runtime
    runtime.loraFolders = []
    let restored = RuntimeSettings.restoring(try JSONEncoder().encode(runtime), defaults: store.runtime)
    XCTAssertEqual(restored.loraFolders, [])
  }

  @MainActor func testFolderRefreshCreatesLibraryCandidatesWithoutMutatingProject() async throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false,
      invocation: { command, _, _, _ in
        XCTAssertEqual(command, "lora-scan")
        return ["entries": [["name": "Style", "path": "/tmp/style.safetensors",
          "sourceFolder": "/tmp", "status": "ready", "loraModel": "ltx25"]], "warnings": []]
      })
    store.project.clips = [Clip(engine: .ltx25)]
    store.selectedClipID = store.project.clips[0].id
    let before = store.project
    await store.refreshLoRAFolders()
    XCTAssertEqual(store.project, before)
    XCTAssertTrue(store.globalAssets.isEmpty)
    let asset = try XCTUnwrap(store.compatibleLoRAs(for: .ltx25).first)
    store.applyLoRAMembers([LoRAMember(asset: asset, strength: 0.4)])
    XCTAssertEqual(store.selectedClip?.attachments.first?.strength, 0.4)
    store.runtime.loraFolders = []
    XCTAssertTrue(store.folderLoRAEntries.isEmpty)
    XCTAssertEqual(store.project.assets.first?.path, "/tmp/style.safetensors")
    store.undo()
    XCTAssertEqual(store.project, before)
  }

  @MainActor func testLateFolderScanCannotRestoreRemovedFolder() async throws {
    var store: StudioStore!
    store = StudioStore(dataDirectory: try directory(), restoreSession: false,
      invocation: { _, _, _, _ in
        store.runtime.loraFolders = []
        return ["entries": [["name": "Stale", "path": "/tmp/stale.safetensors",
          "sourceFolder": "/tmp", "status": "ready", "loraModel": "h3"]], "warnings": []]
      })
    await store.refreshLoRAFolders()
    XCTAssertTrue(store.folderLoRAEntries.isEmpty)
  }

  @MainActor func testDrawThingsAdditionValidatesCatalogAndSupportsUndo() async throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    var clip = Clip(engine: .drawThings)
    clip.drawThings = DrawThingsSelection(profileID: "server", modelID: "model", modelFamily: "h3")
    store.project.clips = [clip]
    store.selectedClipID = clip.id
    await store.drawThingsDiscovery.load(DrawThingsConnection(id: "server")) {
      ["models": [], "capabilities": [:], "loras": [["id": "style.ckpt", "name": "Style",
        "family": "h3", "compatibleModelIDs": ["model"]]]]
    }
    let original = store.project
    store.addDrawThingsLoRA(profileID: "wrong", modelID: "style.ckpt")
    XCTAssertEqual(store.project, original)
    store.addDrawThingsLoRA(profileID: "server", modelID: "style.ckpt")
    XCTAssertEqual(store.selectedClip?.drawThings?.loras.map(\.modelID), ["style.ckpt"])
    store.addDrawThingsLoRA(profileID: "server", modelID: "style.ckpt")
    XCTAssertEqual(store.selectedClip?.drawThings?.loras.count, 1)
    store.undo()
    XCTAssertEqual(store.project, original)
  }

  @MainActor func testDrawThingsGroupStorageUsesIsolatedLibraryAndPreservesClip() async throws {
    let root = try directory()
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var clip = Clip(engine: .drawThings)
    clip.drawThings = DrawThingsSelection(profileID: "server", modelID: "model", modelFamily: "h3",
      loras: [DrawThingsLoRA(modelID: "style.ckpt", weight: 0.4, enabled: false)])
    store.project.clips = [clip]
    store.selectedClipID = clip.id
    await store.drawThingsDiscovery.load(DrawThingsConnection(id: "server")) {
      ["models": [], "capabilities": [:], "loras": [["id": "style.ckpt", "name": "Style",
        "family": "h3", "compatibleModelIDs": ["model"]]]]
    }
    let original = store.project
    XCTAssertTrue(store.saveCurrentDrawThingsLoRAGroup(name: "Muted style"))
    let file = root.appendingPathComponent("drawthings-lora-groups.json")
    let saved = try JSONDecoder().decode([DrawThingsLoRAGroup].self, from: Data(contentsOf: file))
    XCTAssertEqual(saved.first?.members.first?.weight, 0.4)
    XCTAssertEqual(saved.first?.members.first?.isEnabled, false)
    XCTAssertEqual(store.project, original)
    let restored = StudioStore(dataDirectory: root, restoreSession: false)
    restored.loadDrawThingsLoRAGroups()
    XCTAssertEqual(restored.drawThingsLoRAGroups, saved)
    restored.deleteDrawThingsLoRAGroup(try XCTUnwrap(saved.first?.id))
    XCTAssertTrue(restored.drawThingsLoRAGroups.isEmpty)
    XCTAssertEqual(store.project, original)
  }

  @MainActor func testImageLibraryTargetsDraftWithoutChangingTheUnderlyingClip() async throws {
    let store = StudioStore(dataDirectory: try directory(), restoreSession: false)
    let clip = Clip(engine: .h3)
    store.project.clips = [clip]
    store.selectedClipID = clip.id
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(
      scope: .project, projectID: store.project.id, owner: nil))
    draft.profileID = "server"; draft.modelID = "image"
    store.imageDraft = draft
    store.imageEstimate = ["eligibility": "allowed"]
    await store.drawThingsDiscovery.load(DrawThingsConnection(id: "server")) {
      ["models": [["id": "image", "name": "Image model", "family": "zImage"]],
       "capabilities": [:], "loras": [["id": "style.ckpt", "name": "Style",
        "family": "zImage", "compatibleModelIDs": ["image", "other-compatible-image"]]]]
    }
    let original = store.project
    store.addDrawThingsLoRA(profileID: "server", modelID: "style.ckpt", imageWorkspace: true)
    XCTAssertEqual(store.imageDraft?.loras.map(\.modelID), ["style.ckpt"])
    XCTAssertNil(store.imageEstimate)
    XCTAssertEqual(store.project, original)
    XCTAssertTrue(store.saveCurrentDrawThingsLoRAGroup(name: "Image style", imageWorkspace: true))
    XCTAssertEqual(store.drawThingsLoRAGroups.first?.family, "zImage")
    XCTAssertEqual(store.drawThingsLoRAGroups.first?.compatibleModelIDs, ["image", "other-compatible-image"])
    XCTAssertEqual(store.project, original)
  }
}
