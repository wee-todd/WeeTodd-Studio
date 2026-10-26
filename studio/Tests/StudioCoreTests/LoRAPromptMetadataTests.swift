import Foundation
import XCTest
@testable import StudioCore

final class LoRAPromptMetadataTests: XCTestCase {
  func testApplyingLoRAInsertsStoredTriggerAndKeepsNotesOutOfPrompt() throws {
    var asset = MediaAsset(name: "Weapon", kind: .lora, path: "/tmp/weapon.safetensors")
    asset.loraModel = .h3
    var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(asset)) as? [String: Any])
    json["loraPrompt"] = ["triggers": ["BUNNY"], "notes": "Describe contact and recovery."]
    let imported = try JSONDecoder().decode(MediaAsset.self, from: JSONSerialization.data(withJSONObject: json))
    var project = StudioProject()
    var clip = Clip(engine: .h3); clip.prompt = "A robot practices with a wooden sword."
    project.clips = [clip]
    try project.applyLoRAs([LoRAMember(asset: imported)], to: clip.id)
    XCTAssertEqual(project.clips[0].prompt, "BUNNY\nA robot practices with a wooden sword.")
    XCTAssertFalse(project.clips[0].prompt.contains("Describe contact"))
  }
}

extension LoRAPromptMetadataTests {
  func testStructuredH3InsertionPreservesReferenceAndAudioSections() {
    let prompt = "Picture 1 is fully referenced at 0.00 seconds.\n\nintegrated_multimodal_description: [Shot 1] A robot waves.\n\noverall_soundscape: Quiet footsteps.\n\nnon_diegetic_music: N/A"
    let expected = "Picture 1 is fully referenced at 0.00 seconds.\n\nintegrated_multimodal_description: [Shot 1] BUNNY A robot waves.\n\noverall_soundscape: Quiet footsteps.\n\nnon_diegetic_music: N/A"
    XCTAssertEqual(LoRAPromptInsertion.applying([.init(triggers: ["BUNNY"])], to: prompt), expected)
    let reference = "subject_definitions:\n<Subject 1> is a robot.\ndetailed_description:\nA cinematic scene.\n[Shot 1]: The robot walks.\noverall_soundscape:\nFootsteps."
    let referenceExpected = "subject_definitions:\n<Subject 1> is a robot.\ndetailed_description:\nA cinematic scene.\n[Shot 1]: r3tr0sks The robot walks.\noverall_soundscape:\nFootsteps."
    XCTAssertEqual(LoRAPromptInsertion.applying([.init(triggers: ["r3tr0sks"])], to: reference), referenceExpected)
  }

  func testDeduplicatesPhrasesButDoesNotMatchInsideIdentifiers() {
    let metadata = LoRAPromptMetadata(triggers: ["BUNNY", "cinematic lighting", "bunny", " "])
    let prompt = "bunny_crisp_motion, CINEMATIC\n lighting. A robot waves."
    let result = LoRAPromptInsertion.applying([metadata], to: prompt)
    XCTAssertEqual(result, "BUNNY\nbunny_crisp_motion, CINEMATIC\n lighting. A robot waves.")
    XCTAssertEqual(LoRAPromptInsertion.applying([metadata], to: result), result)
    XCTAssertEqual(LoRAPromptInsertion.applying([.init(triggers: ["fox"])], to: "A foxglove grows."), "fox\nA foxglove grows.")
    XCTAssertEqual(LoRAPromptInsertion.applying([.init(triggers: [" BUNNY "])], to: ""), "BUNNY")
  }

  func testDisabledGroupsKeepPromptAndMetadataSnapshotsRoundTrip() throws {
    var asset = MediaAsset(name: "Style", kind: .lora, path: "/tmp/style.safetensors")
    asset.loraModel = .h3
    asset.loraPrompt = .init(triggers: ["BUNNY"], notes: "Optional guidance", sourceURL: "https://example.com/model")
    var project = StudioProject(); var clip = Clip(engine: .h3); clip.prompt = "A robot walks."
    project.clips = [clip]
    try project.applyLoRAs([LoRAMember(asset: asset, enabled: false)], to: clip.id)
    XCTAssertEqual(project.clips[0].prompt, "A robot walks.")
    asset.loraPrompt?.notes = "Changed in the library"
    let snapshot = try project.loraGroupSnapshot(for: clip.id, name: "Snapshot")
    XCTAssertEqual(snapshot.members[0].asset.loraPrompt?.notes, "Optional guidance")
    XCTAssertEqual(try JSONDecoder().decode(LoRAGroup.self, from: JSONEncoder().encode(snapshot)), snapshot)
    XCTAssertEqual(try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(project)), project)
  }

  func testLegacyAssetsAndDrawThingsMembersDecodeWithoutPromptMetadata() throws {
    let legacy = Data("{\"modelID\":\"style.ckpt\",\"weight\":0.7}".utf8)
    XCTAssertNil(try JSONDecoder().decode(DrawThingsLoRA.self, from: legacy).promptMetadata)
    var member = DrawThingsLoRA(modelID: "style.ckpt")
    member.promptMetadata = .init(triggers: ["style"], notes: "Private guidance")
    XCTAssertEqual(try JSONDecoder().decode(DrawThingsLoRA.self, from: JSONEncoder().encode(member)), member)
  }
}

extension LoRAPromptMetadataTests {
  func testSeparateTriggerFragmentsCannotSuppressARequiredPhrase() {
    let metadata = LoRAPromptMetadata(triggers: ["red", "fox", "red fox"])
    let result = LoRAPromptInsertion.applying([metadata], to: "")
    XCTAssertEqual(result, "red, fox, red fox")
    XCTAssertEqual(LoRAPromptInsertion.applying([metadata], to: result), result)
  }
}
