import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class LoRAInteractionTests: XCTestCase {
  private func temporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
    return directory
  }

  @MainActor func testNativeGroupReplaceAndToggleSupportUndoWithoutChangingAcceptedTake() throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false)
    var clip = Clip(engine: .h3)
    clip.sourcePath = "/tmp/accepted.mp4"
    clip.versions = [RenderVersion(path: clip.sourcePath, seed: 7, prompt: "accepted", recipePath: "/tmp/recipe.json")]
    store.project.clips = [clip]; store.selectedClipID = clip.id
    var asset = MediaAsset(name: "Style", kind: .lora, path: "/tmp/style.safetensors")
    asset.loraModel = .h3
    store.applyLoRAMembers([LoRAMember(asset: asset, strength: 0.7)])
    let before = store.project
    let group = LoRAGroup(name: "Muted style", engine: .h3,
      members: [LoRAMember(asset: asset, strength: 0.4, enabled: false)])
    store.applyLoRAGroup(group, mode: .replace)
    XCTAssertFalse(try XCTUnwrap(store.selectedClip?.attachments.first).isEnabled)
    XCTAssertEqual(store.selectedClip?.versions, clip.versions)
    store.undo()
    XCTAssertEqual(store.project, before)
    store.redo()
    XCTAssertEqual(store.selectedClip?.attachments.first?.strength, 0.4)
    store.editClip { $0.attachments[0].enabled = true }
    XCTAssertEqual(store.selectedClip?.attachments.first?.strength, 0.4)
    store.undo()
    XCTAssertFalse(try XCTUnwrap(store.selectedClip?.attachments.first).isEnabled)
    XCTAssertEqual(store.selectedClip?.sourcePath, "/tmp/accepted.mp4")
  }

  @MainActor func testDuplicateGroupFailureCreatesNoUndoOrPartialAssets() throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false)
    let clip = Clip(engine: .h3)
    store.project.clips = [clip]; store.selectedClipID = clip.id
    var asset = MediaAsset(name: "Style", kind: .lora, path: "/tmp/style.safetensors")
    asset.loraModel = .h3
    let member = LoRAMember(asset: asset)
    let before = store.project
    store.applyLoRAMembers([member, member], mode: .replace)
    XCTAssertEqual(store.project, before)
    XCTAssertFalse(store.canUndo)
    XCTAssertNotNil(store.error)
  }

  @MainActor func testDisabledNativeMissingLoRADoesNotBlockReadiness() throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false)
    var clip = Clip(engine: .h3)
    var attachment = Attachment(assetID: UUID(), role: .lora)
    attachment.enabled = false; clip.attachments = [attachment]
    XCTAssertFalse(store.issues(for: clip).contains("Relink a missing attachment"))
    clip.attachments[0].enabled = true
    XCTAssertTrue(store.issues(for: clip).contains("Relink a missing attachment"))
  }

  @MainActor func testNativeImportPassesExplicitProfileAndUsesDeclaredLayout() async throws {
    let directory = try temporaryDirectory()
    var inspection: [String: Any] = [:]
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { command, _, payload, _ in
      XCTAssertEqual(command, "inspect")
      inspection = payload
      return ["kind": "lora", "loraModel": "h3", "loraProfile": "turbo", "loraLayout": "contiguous_qkv"]
    })
    await store.importURLs([URL(fileURLWithPath: "/tmp/turbo.safetensors")], scope: .global,
      loraModel: .h3, loraProfile: "turbo", loraLayout: "auto", loraAdalnInputGrid: "/tmp/grid.safetensors")
    XCTAssertEqual(inspection["loraProfile"] as? String, "turbo")
    XCTAssertEqual(inspection["loraLayout"] as? String, "auto")
    XCTAssertEqual(inspection["loraAdalnInputGrid"] as? String, "/tmp/grid.safetensors")
    let imported = try XCTUnwrap(store.globalAssets.first)
    XCTAssertEqual(imported.loraProfile, "turbo")
    XCTAssertEqual(imported.loraLayout, "contiguous_qkv")
    XCTAssertEqual(imported.loraAdalnInputGrid, "/tmp/grid.safetensors")
    let saved = try JSONDecoder().decode([MediaAsset].self,
      from: Data(contentsOf: directory.appendingPathComponent("global-assets.json")))
    XCTAssertEqual(saved, [imported])
  }

  @MainActor func testLTXImportOmitsH3SamplingFieldsFromRequestAndSavedAsset() async throws {
    for model in [LoRAModel.ltx23, .ltx25] {
      let directory = try temporaryDirectory()
      var inspection: [String: Any] = [:]
      let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { _, _, payload, _ in
        inspection = payload
        return ["kind": "lora", "loraModel": model.rawValue]
      })
      await store.importURLs([URL(fileURLWithPath: "/tmp/style.safetensors")], scope: .global,
        loraModel: model, loraProfile: "standard", loraLayout: "auto", loraAdalnInputGrid: "/tmp/grid.safetensors")
      XCTAssertEqual(inspection["loraModel"] as? String, model.rawValue)
      for field in ["loraProfile", "loraLayout", "loraAdalnInputGrid"] { XCTAssertNil(inspection[field]) }
      let imported = try XCTUnwrap(store.globalAssets.first)
      XCTAssertEqual(imported.loraModel, model)
      XCTAssertNil(imported.loraProfile)
      XCTAssertNil(imported.loraLayout)
      XCTAssertNil(imported.loraAdalnInputGrid)
      let saved = try JSONDecoder().decode([MediaAsset].self,
        from: Data(contentsOf: directory.appendingPathComponent("global-assets.json")))
      XCTAssertEqual(saved, [imported])
    }
  }

  @MainActor func testLTXImportWithDefaultArgumentsLeavesH3FieldsUnset() async throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: { _, _, payload, _ in
      for field in ["loraProfile", "loraLayout", "loraAdalnInputGrid"] { XCTAssertNil(payload[field]) }
      return ["kind": "lora", "loraModel": "ltx25"]
    })
    await store.importURLs([URL(fileURLWithPath: "/tmp/style.safetensors")], scope: .global, loraModel: .ltx25)
    let imported = try XCTUnwrap(store.globalAssets.first)
    XCTAssertNil(imported.loraProfile)
    XCTAssertNil(imported.loraLayout)
    XCTAssertNil(imported.loraAdalnInputGrid)
  }

  @MainActor func testDeclaredLTXModelDoesNotInheritSelectedH3SamplingFields() async throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: { _, _, _, _ in
      return ["kind": "lora", "loraModel": "ltx23"]
    })
    await store.importURLs([URL(fileURLWithPath: "/tmp/style.safetensors")], scope: .global,
      loraModel: .h3, loraProfile: "turbo", loraLayout: "auto", loraAdalnInputGrid: "/tmp/grid.safetensors")
    let imported = try XCTUnwrap(store.globalAssets.first)
    XCTAssertEqual(imported.loraModel, .ltx23)
    XCTAssertNil(imported.loraProfile)
    XCTAssertNil(imported.loraLayout)
    XCTAssertNil(imported.loraAdalnInputGrid)
  }

  @MainActor func testNewestGlobalImportSuppliesPickerMetadataWithoutChangingAppliedClip() async throws {
    let store = StudioStore(dataDirectory: try temporaryDirectory(), restoreSession: false, invocation: { _, _, _, _ in
      return ["kind": "lora", "loraModel": "h3"]
    })
    let clip = Clip(engine: .h3)
    store.project.clips = [clip]; store.selectedClipID = clip.id
    await store.importURLs([URL(fileURLWithPath: "/tmp/style.safetensors")], scope: .global,
      loraModel: .h3, loraProfile: "standard", loraLayout: "auto")
    let original = try XCTUnwrap(store.globalAssets.first)
    store.applyLoRAMembers([LoRAMember(asset: original, strength: 0.6)])
    let applied = store.project
    await store.importURLs([URL(fileURLWithPath: "/tmp/./style.safetensors")], scope: .global,
      loraModel: .h3, loraProfile: "turbo", loraLayout: "contiguous_qkv", loraAdalnInputGrid: "/tmp/grid.safetensors")
    let candidates = store.compatibleLoRAs(for: .h3)
    XCTAssertEqual(candidates.count, 1)
    XCTAssertEqual(candidates.first?.id, store.globalAssets.last?.id)
    XCTAssertEqual(candidates.first?.loraProfile, "turbo")
    XCTAssertEqual(candidates.first?.loraLayout, "contiguous_qkv")
    XCTAssertEqual(candidates.first?.loraAdalnInputGrid, "/tmp/grid.safetensors")
    XCTAssertEqual(store.globalAssets.first, original)
    XCTAssertEqual(store.project, applied)
    XCTAssertEqual(store.project.assets.first?.loraProfile, "standard")
    store.globalAssets = []
    XCTAssertEqual(store.compatibleLoRAs(for: .h3).first?.loraProfile, "standard")
  }

  @MainActor func testLoRAGroupsUseInjectedDirectoryForReadAndWrite() throws {
    let directory = try temporaryDirectory()
    let file = directory.appendingPathComponent("lora-groups.json")
    var asset = MediaAsset(name: "Style", kind: .lora, path: "/tmp/style.safetensors")
    asset.loraModel = .h3
    var group = LoRAGroup(name: "Temporary group", engine: .h3, members: [LoRAMember(asset: asset)])
    try JSONEncoder().encode([group]).write(to: file, options: .atomic)
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.loadLoRAGroups()
    XCTAssertEqual(store.loraGroups, [group])
    // Do not attempt a write until reads are proven to use the isolated directory.
    guard store.loraGroups == [group] else { return }
    group.members[0].strength = 0.42
    group.members[0].enabled = false
    XCTAssertTrue(store.saveLoRAGroup(group))
    let saved = try JSONDecoder().decode([LoRAGroup].self, from: Data(contentsOf: file))
    XCTAssertEqual(saved, [group])
    let restored = StudioStore(dataDirectory: directory, restoreSession: false)
    restored.loadLoRAGroups()
    XCTAssertEqual(restored.loraGroups, [group])
    restored.deleteLoRAGroup(group.id)
    XCTAssertEqual(try JSONDecoder().decode([LoRAGroup].self, from: Data(contentsOf: file)), [])
  }
}
