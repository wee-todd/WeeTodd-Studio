import XCTest
@testable import StudioCore

final class CharacterPanelRecipeTests: XCTestCase {
  func testDefaultSeparatePassesUseEightStepsAndExclusiveAdapters() throws {
    var settings = CharacterRefinementSettings()
    XCTAssertTrue(settings.twoPass)
    XCTAssertEqual(settings.steps, 8)
    settings.modelID = "klein9"; settings.detailLoRAID = "detail"; settings.headLoRAID = "head"
    settings.replaceFaces = true
    let combined = try CharacterPanelRecipe.makeDraft(role: .closeUp, definition: definition(), settings: settings,
      profileID: "local", panelPath: "/panel-2x.png", headPath: "/head.png", width: 960, height: 2176,
      documentID: UUID(), seed: 42)
    let head = try CharacterPanelRecipe.headOnlyDraft(from: combined)
    XCTAssertEqual(head.loras.map(\.modelID), ["head"])
    XCTAssertEqual(head.moodboard.map(\.path), ["/panel-2x.png", "/head.png"])
    XCTAssertEqual(head.steps, 8)
    XCTAssertNil(head.managedCharacterPromptIssue)
    XCTAssertTrue(head.prompt.hasPrefix("head_swap: replace the head with the reference head."))
    settings.replaceFaces = false
    let detail = try CharacterPanelRecipe.makeDraft(role: .closeUp, definition: definition(), settings: settings,
      profileID: "local", panelPath: "/head-pass-output.png", headPath: nil, width: 960, height: 2176,
      documentID: UUID(), seed: 1042, preservesInputHead: true)
    XCTAssertEqual(detail.loras.map(\.modelID), ["detail"])
    XCTAssertEqual(detail.moodboard.map(\.path), ["/head-pass-output.png"])
    XCTAssertEqual(detail.steps, 8)
    XCTAssertEqual(detail.width, head.width)
    XCTAssertEqual(detail.height, head.height)
    XCTAssertTrue(detail.characterPanel?.preservesInputHead == true)
    XCTAssertTrue(detail.prompt.hasPrefix("high quality."))
    XCTAssertFalse(detail.prompt.contains("head_swap:"))
    XCTAssertNil(detail.managedCharacterPromptIssue)
    XCTAssertThrowsError(try CharacterPanelRecipe.headOnlyDraft(from: detail))
    var legacy = settings; legacy.twoPass = false; legacy.steps = 4
    XCTAssertEqual(try JSONDecoder().decode(CharacterRefinementSettings.self,
      from: JSONEncoder().encode(legacy)), legacy)
  }
  func testPhotographicHeadReplacementPreservesReferenceTextureWithoutCanonicalSkinLeak() {
    var value = definition()
    value.appearance.setText("face.skinTexture", "canonical head has painted porcelain cracks")
    for role in CharacterPanelRole.allCases {
      let prompt = CharacterPanelPromptContext(role: role, definition: value,
        replacesHead: true, detailLoRAID: "detail", headLoRAID: "head").prompt
      XCTAssertTrue(prompt.contains("Preserve the reference images' visible surface texture and tonal variation"))
      XCTAssertTrue(prompt.contains("no beauty retouching, airbrushing or waxy smoothing"))
      XCTAssertFalse(prompt.contains("painted porcelain cracks"))
    }
    value.settings.stylePresetID = "clay"
    let clay = CharacterPanelPromptContext(role: .front, definition: value,
      replacesHead: true, detailLoRAID: "detail", headLoRAID: "head").prompt
    XCTAssertFalse(clay.contains("no beauty retouching"))
  }
  func testCloseUpKeepsExactInputFramingWithoutFullBodyOrOutfitInstructions() {
    var value = definition()
    value.appearance.setText("body.build", "broad muscular physique")
    value.appearance.setText("eyes.color", "amber")
    value.appearance.setText("hair.style", "short auburn curls")
    func record(_ fields: [String: String]) -> CharacterRepeatableRecord {
      .init(order: 0, fields: fields.mapValues { .value(.text($0)) })
    }
    value.appearance.garments = [record(["type": "gray trousers", "bodyRegion": "legs"]),
      record(["type": "brown ankle boots", "bodyRegion": "feet"])]
    value.appearance.features = [record(["type": "small eyebrow scar", "placement": "left eyebrow"]),
      record(["type": "large calf tattoo", "placement": "right calf"])]
    for replacesHead in [false, true] {
      let context = CharacterPanelPromptContext(role: .closeUp, definition: value,
        replacesHead: replacesHead, detailLoRAID: "detail", headLoRAID: replacesHead ? "head" : nil)
      let prompt = context.prompt
      XCTAssertTrue(prompt.contains("Preserve the exact tight face/head crop and subject scale of Image 1"))
      XCTAssertTrue(prompt.contains("Do not zoom out or reveal torso, legs or feet"))
      for omitted in ["broad muscular physique", "gray trousers", "brown ankle boots", "large calf tattoo",
        "body anatomy", "outfit construction", "head/body junction"] {
        XCTAssertFalse(prompt.contains(omitted), omitted)
      }
      if replacesHead {
        XCTAssertTrue(prompt.hasPrefix("head_swap: replace the head with the reference head. high quality."))
        XCTAssertTrue(prompt.contains("Image 2 supplies the reference head"))
        XCTAssertFalse(prompt.contains("short auburn curls"))
        XCTAssertFalse(prompt.contains("small eyebrow scar"))
      } else {
        XCTAssertTrue(prompt.contains("short auburn curls"))
        XCTAssertTrue(prompt.contains("small eyebrow scar"))
      }
      XCTAssertEqual(context.definition, value)
      for role in [CharacterPanelRole.front, .side, .back] {
        let full = CharacterPanelPromptContext(role: role, definition: value,
          replacesHead: replacesHead, detailLoRAID: "detail", headLoRAID: replacesHead ? "head" : nil).prompt
        XCTAssertTrue(full.contains("gray trousers"))
        XCTAssertTrue(full.contains("brown ankle boots"))
        XCTAssertTrue(full.contains("broad muscular physique"))
        XCTAssertTrue(full.contains("body anatomy, outfit construction"))
        XCTAssertFalse(full.contains("Do not zoom out or reveal torso"))
      }
    }
  }

  private func definition() -> CharacterSheetDefinition {
    var value = CharacterSheetDefinition.newDraft()
    value.appearance.setText("identity.species", "human")
    value.appearance.setText("identity.type", "human")
    return value
  }
  func testCombinedRecipeKeepsHeadSecondAndDoesNotRequestFourViews() throws {
    var settings = CharacterRefinementSettings()
    settings.modelID = "klein9"; settings.detailLoRAID = "detail"; settings.headLoRAID = "head"
    settings.replaceFaces = true
    let draft = try CharacterPanelRecipe.makeDraft(role: .back, definition: definition(), settings: settings,
      profileID: "local", panelPath: "/panel.png", headPath: "/head.png", width: 960, height: 2176,
      documentID: UUID(), seed: 42)
    XCTAssertEqual(draft.moodboard.map(\.path), ["/panel.png", "/head.png"])
    XCTAssertNil(draft.canvas)
    XCTAssertEqual(draft.loras.map(\.modelID), ["detail", "head"])
    XCTAssertTrue(draft.prompt.hasPrefix("head_swap: replace the head with the reference head. high quality."))
    XCTAssertTrue(draft.prompt.contains("rear"))
    XCTAssertFalse(draft.prompt.contains("4-view turnaround"))
    XCTAssertEqual(draft.seed, 42)
    var changed = draft; changed.prompt = "override"
    XCTAssertNotNil(changed.managedCharacterPromptIssue)
  }
  func testMissingHeadFailsBeforePreparingAReplacementRequest() {
    var settings = CharacterRefinementSettings(); settings.replaceFaces = true
    settings.modelID = "klein9"; settings.detailLoRAID = "detail"; settings.headLoRAID = "head"
    XCTAssertThrowsError(try CharacterPanelRecipe.makeDraft(role: .front, definition: definition(), settings: settings,
      profileID: "local", panelPath: "/panel.png", headPath: nil, width: 960, height: 2176, documentID: UUID(), seed: 1))
  }
  func testUnresolvedFieldsBlockPanelCreationAndRestoredRequests() throws {
    var settings = CharacterRefinementSettings(); settings.modelID = "klein9"; settings.detailLoRAID = "detail"
    XCTAssertThrowsError(try CharacterPanelRecipe.makeDraft(role: .front, definition: .newDraft(), settings: settings,
      profileID: "local", panelPath: "/panel.png", headPath: nil, width: 960, height: 2176, documentID: UUID(), seed: 1))
    var draft = try CharacterPanelRecipe.makeDraft(role: .front, definition: definition(), settings: settings,
      profileID: "local", panelPath: "/panel.png", headPath: nil, width: 960, height: 2176, documentID: UUID(), seed: 1)
    XCTAssertNil(draft.managedCharacterPromptIssue)
    draft.characterPanel?.definition.appearance.setState("identity.species", .unspecified)
    draft.prompt = draft.characterPanel!.prompt
    XCTAssertNotNil(draft.managedCharacterPromptIssue)
    XCTAssertThrowsError(try draft.request(id: "restored"))
    draft.characterPanel?.definition = definition()
    draft.characterPanel?.definition.appearance.setText("body.build", "<lora:extra:1>")
    draft.prompt = draft.characterPanel!.prompt
    XCTAssertNotNil(draft.managedCharacterPromptIssue)
  }

  func testExecutionDigestIgnoresInputUUIDButIncludesImageContent() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let image = root.appendingPathComponent("panel.png")
    try Data([1, 2, 3]).write(to: image)
    var settings = CharacterRefinementSettings(); settings.modelID = "klein9"; settings.detailLoRAID = "detail"
    let draft = try CharacterPanelRecipe.makeDraft(role: .front, definition: definition(), settings: settings,
      profileID: "local", panelPath: image.path, headPath: nil, width: 960, height: 2176, documentID: UUID(), seed: 1)
    let original = try draft.characterExecutionDigest()
    var resumed = draft; resumed.moodboard[0].id = UUID()
    XCTAssertEqual(try resumed.characterExecutionDigest(), original)
    try Data([3, 2, 1]).write(to: image)
    XCTAssertNotEqual(try resumed.characterExecutionDigest(), original)
  }

  func testReferenceHeadOverridesHeadRecordsButPreservesLocatedBodyRecords() throws {
    var value = definition()
    func record(_ fields: [String: String]) -> CharacterRepeatableRecord {
      .init(order: 0, fields: fields.mapValues { .value(.text($0)) })
    }
    let bodyAccessory = record(["type": "copper bracelet", "placement": "left wrist"])
    let headAccessory = record(["type": "silver earring", "placement": "left ear"])
    let jacket = record(["type": "navy jacket", "bodyRegion": "torso"])
    let helmet = record(["type": "brass helmet", "bodyRegion": "head"])
    value.appearance.garments = [jacket, helmet]
    value.appearance.accessories = [bodyAccessory, headAccessory, record(["type": "unplaced jewelry"])]
    value.appearance.features = [record(["type": "pale scar", "placement": "left eyebrow"]),
      record(["type": "sun tattoo", "placement": "right forearm"]), record(["type": "unplaced scar"])]
    value.appearance.surfaces = [record(["target": "scalp", "texture": "head-only texture"]),
      record(["target": bodyAccessory.id.uuidString, "finish": "brushed copper"]),
      record(["target": headAccessory.id.uuidString, "finish": "polished silver"]),
      record(["target": jacket.id.uuidString, "texture": "woven wool"]),
      record(["target": helmet.id.uuidString, "texture": "hammered brass"]),
      record(["texture": "unplaced texture"])]
    var context = CharacterPanelPromptContext(role: .back, definition: value, replacesHead: true,
      detailLoRAID: "detail", headLoRAID: "head")
    let original = value
    for preserve in [false, true] {
      context.replacesHead = !preserve; context.preservesInputHead = preserve
      let prompt = context.prompt
      for omitted in ["silver earring", "unplaced jewelry", "pale scar", "unplaced scar", "head-only texture", "polished silver", "hammered brass", "unplaced texture", "brass helmet"] {
        XCTAssertFalse(prompt.contains(omitted), omitted)
      }
      for retained in ["copper bracelet", "sun tattoo", "brushed copper", "woven wool", "navy jacket"] {
        XCTAssertTrue(prompt.contains(retained), retained)
      }
      XCTAssertEqual(context.definition, original)
    }
    context.replacesHead = false; context.preservesInputHead = false
    XCTAssertTrue(context.prompt.contains("pale scar"))
    XCTAssertTrue(context.prompt.contains("silver earring"))
  }
}
