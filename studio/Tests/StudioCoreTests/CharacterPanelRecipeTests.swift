import XCTest
@testable import StudioCore

final class CharacterPanelRecipeTests: XCTestCase {
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
