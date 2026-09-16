import XCTest
@testable import StudioCore

final class LoRAStackParityTests: XCTestCase {
  private func object<T: Encodable>(_ value: T) throws -> [String: Any] {
    try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
  }
  private func withFields<T: Codable>(_ value: T, _ fields: [String: Any]) throws -> T {
    var object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
    object.merge(fields) { _, new in new }
    return try JSONDecoder().decode(T.self, from: JSONSerialization.data(withJSONObject: object))
  }

  func testDisabledNativeAttachmentAndMemberRetainStateOnRoundTrip() throws {
    var asset = MediaAsset(name: "Style", kind: .lora, path: "/tmp/style.safetensors")
    asset.loraModel = .h3
    var attachment = Attachment(assetID: asset.id, role: .lora)
    attachment.strength = 0.67
    let disabledAttachment = try withFields(attachment, ["enabled": false])
    let disabledMember = try withFields(LoRAMember(asset: asset, strength: 0.67), ["enabled": false])
    for object in [try object(disabledAttachment), try object(disabledMember)] {
      XCTAssertEqual(object["enabled"] as? Bool, false)
      XCTAssertEqual(object["strength"] as? Double, 0.67)
    }
  }

  func testDisabledDrawThingsLoRAIsSavedButExcludedFromImageExecution() throws {
    let disabled = try withFields(DrawThingsLoRA(modelID: "disabled-style", weight: 0.63), ["enabled": false])
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .global, projectID: UUID()))
    draft.loras = [disabled, DrawThingsLoRA(modelID: "active-style", weight: 0.82)]
    let saved = try JSONDecoder().decode(DrawThingsImageDraft.self, from: JSONEncoder().encode(draft))
    XCTAssertEqual(try object(saved.loras[0])["enabled"] as? Bool, false)
    XCTAssertEqual(saved.loras[0].weight, 0.63)
    let execution = try XCTUnwrap(saved.request(id: "test")["loras"] as? [[String: Any]])
    XCTAssertEqual(execution.compactMap { $0["modelID"] as? String }, ["active-style"])
  }

  func testAdapterMetadataSurvivesAssetRoundTrip() throws {
    let asset = try withFields(MediaAsset(name: "Turbo", kind: .lora), [
      "loraProfile": "turbo", "loraLayout": "contiguous_qkv", "loraAdalnInputGrid": "/tmp/grid.safetensors"
    ])
    let object = try object(asset)
    XCTAssertEqual(object["loraProfile"] as? String, "turbo")
    XCTAssertEqual(object["loraLayout"] as? String, "contiguous_qkv")
    XCTAssertEqual(object["loraAdalnInputGrid"] as? String, "/tmp/grid.safetensors")
  }

  func testLegacyLoRAsExecuteByDefaultAndReenableRetainsStrength() throws {
    let data = Data(#"{"modelID":"style","weight":0.73}"#.utf8)
    var lora = try JSONDecoder().decode(DrawThingsLoRA.self, from: data)
    XCTAssertTrue(lora.isEnabled)
    lora.enabled = false
    lora = try JSONDecoder().decode(DrawThingsLoRA.self, from: JSONEncoder().encode(lora))
    lora.enabled = true
    XCTAssertEqual(lora.weight, 0.73)
    let asset = MediaAsset(name: "Style", kind: .lora)
    XCTAssertTrue(try JSONDecoder().decode(Attachment.self, from: JSONEncoder().encode(Attachment(assetID: asset.id, role: .lora))).isEnabled)
    XCTAssertTrue(try JSONDecoder().decode(LoRAMember.self, from: JSONEncoder().encode(LoRAMember(asset: asset))).isEnabled)
  }

  func testNativeReplaceKeepsFramesAndTakeAndSnapshotsDisabledTurboMetadata() throws {
    var project = StudioProject()
    var clip = Clip(engine: .h3)
    clip.sourcePath = "/tmp/accepted.mp4"
    clip.renderedSignature = "accepted-signature"
    let frame = MediaAsset(name: "First", kind: .image, path: "/tmp/first.png")
    clip.attachments = [Attachment(assetID: frame.id, role: .first)]
    project.clips = [clip]; project.assets = [frame]
    var style = MediaAsset(name: "Style", kind: .lora, path: "/tmp/style.safetensors")
    style.loraModel = .h3
    try project.applyLoRAs([LoRAMember(asset: style)], to: clip.id)
    var turbo = MediaAsset(name: "Turbo", kind: .lora, path: "/tmp/turbo.safetensors")
    turbo.loraModel = .h3; turbo.loraProfile = "turbo"; turbo.loraLayout = "contiguous_qkv"
    turbo.loraAdalnInputGrid = "/tmp/grid.safetensors"
    try project.applyLoRAs([LoRAMember(asset: turbo, strength: 0.8, enabled: false)], to: clip.id, groupName: "Turbo group", mode: .replace)
    XCTAssertEqual(project.clips[0].attachments.map(\.role), [.first, .lora])
    XCTAssertEqual(project.clips[0].sourcePath, "/tmp/accepted.mp4")
    XCTAssertEqual(project.clips[0].renderedSignature, "accepted-signature")
    let snapshot = try project.loraGroupSnapshot(for: clip.id, name: "Saved stack")
    XCTAssertFalse(snapshot.members[0].isEnabled)
    XCTAssertEqual(snapshot.members[0].strength, 0.8)
    XCTAssertEqual(snapshot.members[0].asset.loraProfile, "turbo")
    XCTAssertEqual(snapshot.members[0].asset.loraAdalnInputGrid, "/tmp/grid.safetensors")
    project.clips[0].attachments[1].strength = 1.2
    XCTAssertEqual(snapshot.members[0].strength, 0.8)
  }

  func testNativeAddAndInvalidReplaceAreAtomic() throws {
    var project = StudioProject()
    let clip = Clip(engine: .h3); project.clips = [clip]
    var asset = MediaAsset(name: "Style", kind: .lora, path: "/tmp/style.safetensors")
    asset.loraModel = .h3
    let member = LoRAMember(asset: asset)
    try project.applyLoRAs([member], to: clip.id)
    let before = project
    XCTAssertThrowsError(try project.applyLoRAs([member], to: clip.id, mode: .add))
    XCTAssertEqual(project, before)
    XCTAssertThrowsError(try project.applyLoRAs([member, member], to: clip.id, mode: .replace))
    XCTAssertEqual(project, before)
  }

  func testDrawThingsExplicitAddAndReplaceAreAtomicAndCopyState() throws {
    var selection = DrawThingsSelection(profileID: "local", modelID: "h3", modelFamily: "minimaxh3",
      loras: [DrawThingsLoRA(modelID: "existing", weight: 0.6)])
    var group = DrawThingsLoRAGroup(name: "Look", profileID: "local", family: "minimaxh3", compatibleModelIDs: ["h3"],
      members: [DrawThingsLoRA(modelID: "new", weight: 0.8, enabled: false)])
    try selection.apply(group, mode: .add)
    XCTAssertEqual(selection.loras.map(\.modelID), ["existing", "new"])
    let before = selection
    XCTAssertThrowsError(try selection.apply(group, mode: .add))
    XCTAssertEqual(selection, before)
    try selection.apply(group, mode: .replace)
    group.members[0].weight = 1.5
    XCTAssertEqual(selection.loras.count, 1)
    XCTAssertEqual(selection.loras[0].weight, 0.8)
    XCTAssertFalse(selection.loras[0].isEnabled)
    group.members.append(group.members[0])
    let beforeInvalid = selection
    XCTAssertThrowsError(try selection.apply(group, mode: .replace))
    XCTAssertEqual(selection, beforeInvalid)
  }

  func testAdapterAuxiliaryPathUsesProjectPathMapping() {
    var project = StudioProject()
    var asset = MediaAsset(name: "Turbo", kind: .lora, path: "/tmp/turbo.safetensors")
    asset.loraAdalnInputGrid = "/tmp/grid.safetensors"; project.assets = [asset]
    ProjectStorage.mapPaths(&project) { $0.replacingOccurrences(of: "/tmp/", with: "assets/") }
    XCTAssertEqual(project.assets[0].loraAdalnInputGrid, "assets/grid.safetensors")
  }

  func testDisabledNativeLoRADoesNotBlockDrawThingsConditioningAfterBackendChange() {
    var clip = Clip(engine: .drawThings)
    clip.drawThings = DrawThingsSelection(profileID: "local", modelID: "h3", modelFamily: "minimaxh3")
    var attachment = Attachment(assetID: UUID(), role: .lora)
    attachment.enabled = false; clip.attachments = [attachment]
    XCTAssertEqual(clip.drawThingsConditioningIssues(assets: []), [])
    clip.attachments[0].enabled = true
    XCTAssertEqual(clip.drawThingsConditioningIssues(assets: []).count, 1)
  }
}
