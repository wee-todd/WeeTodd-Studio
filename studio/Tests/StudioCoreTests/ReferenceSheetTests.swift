import XCTest
@testable import StudioCore

final class ReferenceSheetTests: XCTestCase {
  func testRestoredLegacyFourViewSheetIsReadOnlyAndRequiresMappingBeforeGeneration() throws {
    var draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: UUID()))
    ReferenceSheetContext(subjectKey: "legacy", name: "Legacy", kind: .character,
      description: "Original authored appearance").apply(to: &draft)
    draft.characterSheetLoRAID = "four-view"; draft.loras = [.init(modelID: "four-view")]
    let restored = try JSONDecoder().decode(DrawThingsImageDraft.self, from: JSONEncoder().encode(draft))
    XCTAssertNil(restored.referenceSheet?.characterDefinition)
    XCTAssertTrue(restored.hasManagedCharacterPrompt)
    XCTAssertNotNil(restored.managedCharacterPromptIssue)
    XCTAssertNotNil(restored.characterSheetRequestIssue)
    XCTAssertThrowsError(try restored.request(id: "regenerate"))
    XCTAssertTrue(restored.prompt.contains("Original authored appearance"))
  }
  func testLinkedContextPreservesConditionalPlacementAndMultipleRolesInBothPaths() throws {
    var ring = PlanningSubject(name: "Arm-ring", kind: .prop)
    ring.details = "Gold ring worn by the old king before the handover."
    var wiglaf = PlanningSubject(name: "Wiglaf", kind: .character)
    wiglaf.relationships = [
      ObjectRelationship(targetID: ring.id, role: .holds, placement: "Receives only after the dragon falls."),
      ObjectRelationship(targetID: ring.id, role: .wears, placement: "Only in a later epilogue.")
    ]
    let workflowRing = try JSONDecoder().decode(WorkflowSubjectProposal.self, from: JSONSerialization.data(withJSONObject: [
      "id": ring.id.uuidString, "name": ring.name, "kind": "prop", "description": ring.details,
      "aliases": [], "evidence": [], "suggestions": []
    ]))
    let links = wiglaf.relationships!.map {
      WorkflowObjectRelationship(id: $0.id.uuidString, targetID: $0.targetID.uuidString, role: $0.role, placement: $0.placement)
    }
    let manual = ReferenceSheetLinks.definitions(subject: wiglaf, inventory: [wiglaf, ring])
    let guided = ReferenceSheetLinks.definitions(relationships: links, inventory: [workflowRing])
    XCTAssertEqual(manual, guided)
    XCTAssertTrue(guided.contains("Role: holds")); XCTAssertTrue(guided.contains("Role: wears"))
    XCTAssertTrue(guided.contains("Receives only after the dragon falls."))
    XCTAssertTrue(guided.contains("Only in a later epilogue."))
    XCTAssertTrue(guided.contains(ring.details))
    let context = ReferenceSheetContext(subjectKey: "wiglaf", name: "Wiglaf", kind: .character,
      description: "A young warrior with ordinary mail and a sheathed sword.", linkedDefinitions: guided)
    XCTAssertTrue(context.prompt.contains("ordinary mail and a sheathed sword"))
    XCTAssertTrue(context.prompt.contains("Receives only after the dragon falls."))
    XCTAssertTrue(context.prompt.contains("authoritative baseline"))
    XCTAssertTrue(context.prompt.contains("unless explicitly requested"))
  }

  func testEnvironmentFallbackDoesNotDuplicateExplicitLocatedInOrInferMentionedObjects() {
    var environment = PlanningSubject(name: "Hall", kind: .environment); environment.details = "Timber hall."
    var set = PlanningSubject(name: "Dais", kind: .set)
    set.environmentID = environment.id; set.details = "Hall beyond the dais."
    let fallback = ReferenceSheetLinks.definitions(subject: set, inventory: [set, environment])
    XCTAssertTrue(fallback.contains("Role: located_in")); XCTAssertTrue(fallback.contains("Timber hall."))
    set.relationships = [ObjectRelationship(targetID: environment.id, role: .located_in, placement: "At the east end.")]
    let explicit = ReferenceSheetLinks.definitions(subject: set, inventory: [set, environment])
    XCTAssertEqual(explicit.components(separatedBy: "Role: located_in").count, 2)
    XCTAssertTrue(explicit.contains("At the east end."))
    set.relationships = nil; set.environmentID = nil
    XCTAssertEqual(ReferenceSheetLinks.definitions(subject: set, inventory: [set, environment]), "")
  }

  func testMissingLinkedDefinitionRetainsItsRoleAndPlacement() {
    let links = [WorkflowObjectRelationship(id: "link", targetID: "missing", role: .uses, placement: "Only after nightfall.")]
    let text = ReferenceSheetLinks.definitions(relationships: links, inventory: [])
    XCTAssertTrue(text.contains("missing")); XCTAssertTrue(text.contains("Role: uses"))
    XCTAssertTrue(text.contains("Only after nightfall.")); XCTAssertTrue(text.contains("Definition unavailable"))
  }

  func testOrdinaryEquipmentRemainsAvailableWithoutInferringAnEventState() {
    var sword = PlanningSubject(name: "Sword", kind: .prop); sword.details = "An intact iron sword."
    var warrior = PlanningSubject(name: "Warrior", kind: .character)
    warrior.relationships = [ObjectRelationship(targetID: sword.id, role: .wears, placement: "Sheathed at his left hip.")]
    let linked = ReferenceSheetLinks.definitions(subject: warrior, inventory: [warrior, sword])
    let context = ReferenceSheetContext(subjectKey: "warrior", name: warrior.name, kind: warrior.kind,
      description: "A warrior with his usual sword.", linkedDefinitions: linked)
    XCTAssertTrue(context.prompt.contains("Preserve ordinary authored clothing and equipment"))
    XCTAssertTrue(context.prompt.contains("Role: wears"))
    XCTAssertTrue(context.prompt.contains("Sheathed at his left hip."))
    XCTAssertTrue(context.prompt.contains("An intact iron sword."))
  }

  func testTemplatesKeepObjectDefinitionAndDoNotGrantApproval() throws {
    for kind in PlanningSubjectKind.allCases {
      let context = ReferenceSheetContext(subjectKey: "workflow:object", name: "Sample", kind: kind,
        description: "Exact approved description.", linkedDefinitions: "prop_123: Silver key.")
      var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: UUID()))
      draft.modelID = "custom-user-model"; draft.steps = 8; context.apply(to: &draft)
      XCTAssertTrue(draft.prompt.contains("Exact approved description."))
      XCTAssertTrue(draft.prompt.contains("prop_123"))
      XCTAssertEqual(draft.modelID, "custom-user-model"); XCTAssertEqual(draft.steps, 8)
      XCTAssertEqual(draft.width % 64, 0); XCTAssertEqual(draft.height % 64, 0)
      XCTAssertEqual(try JSONDecoder().decode(DrawThingsImageDraft.self, from: JSONEncoder().encode(draft)), draft)
    }
  }
  func testLeavingSheetOnNativeBackendDoesNotRestoreAdapterLater() {
    var context = ReferenceSheetContext(subjectKey: "one", name: "Character", kind: .character, description: "A blue robot.")
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: UUID()))
    context.apply(to: &draft)
    draft.characterSheetLoRAID = "four-panel"
    draft.loras = [DrawThingsLoRA(modelID: "four-panel"), DrawThingsLoRA(modelID: "style", weight: 0.3)]
    draft.selectProvider(.nativeMLX)
    context.template = .portrait; context.apply(to: &draft)
    draft.selectProvider(.drawThings)
    XCTAssertEqual(draft.loras.map(\.modelID), ["style"])
    XCTAssertEqual(draft.loras.first?.weight, 0.3)
  }

  func testNewCharacterSheetPromptStartsWithExactFourViewTriggerAndKeepsDetails() {
    let details = "Mira is a tall woman with warm brown skin, green eyes, short silver curls, a scar over her left eyebrow, a navy coat and brass boots."
    let context = ReferenceSheetContext(subjectKey: "mira", name: "Mira", kind: .character, description: details)
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: UUID()))
    context.apply(to: &draft)
    XCTAssertTrue(draft.prompt.hasPrefix("4-view turnaround of a character, front view, side view, back view, facial close-up, plain solid white background"))
    XCTAssertTrue(draft.prompt.contains(details))
    XCTAssertEqual(context.template.rawValue, "characterSheet")
    XCTAssertFalse(draft.prompt.contains("Three clearly separated views"))
    XCTAssertNotEqual(draft.storageKey, draft.destination.storageKey + ":reference:mira")
  }

  func testCharacterTemplatePairsFullBodyViewsWithFaceCloseUp() {
    var context = ReferenceSheetContext(subjectKey: "one", name: "Character", kind: .character,
      description: "Keep this exact face and wardrobe.")
    context.template = .character // Preserve the existing saved three-view template.
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: UUID()))
    context.apply(to: &draft)
    XCTAssertEqual(context.template.label, "Character · front, close-up, profile")
    XCTAssertTrue(draft.prompt.contains("left full-body front view"))
    XCTAssertTrue(draft.prompt.contains("center head-and-shoulders face close-up"))
    XCTAssertTrue(draft.prompt.contains("right full-body side profile at 90 degrees"))
    XCTAssertTrue(draft.prompt.contains("body views at equal scale, entire body and feet visible"))
    XCTAssertTrue(draft.prompt.contains("identity, anatomy, proportions, clothing and accessories identical"))
    XCTAssertFalse(draft.prompt.contains("three-quarter"))
    XCTAssertFalse(draft.negativePrompt.contains("cropped subject"))
    XCTAssertTrue(draft.negativePrompt.contains("cropped full-body views"))
    XCTAssertTrue(ReferenceSheetTemplate.prop.layout.contains("front, three-quarter and side"))
    XCTAssertEqual(ReferenceSheetTemplate.prop.label, "Prop · three views")
  }

  func testExplicitReferenceBindingsSurviveWithoutAnAgentReview() throws {
    let data = Data(#"{"id":"cat","kind":"character","name":"Cat","description":"Black cat","aliases":[],"evidence":[],"suggestions":[],"referenceAssets":["generated:one"]}"#.utf8)
    let subject = try JSONDecoder().decode(WorkflowSubjectProposal.self, from: data)
    XCTAssertEqual(subject.referenceAssetKeys, ["generated:one"])
    XCTAssertNil(subject.descriptionReview)
    let decoded = try JSONDecoder().decode(WorkflowSubjectProposal.self, from: JSONEncoder().encode(subject))
    XCTAssertEqual(decoded.referenceAssetKeys, subject.referenceAssetKeys)
  }

  func testReferenceDraftDoesNotReplaceOrdinaryWorkspace() {
    let destination = ImageAssetDestination(scope: .project, projectID: UUID())
    var normal = DrawThingsImageDraft(destination: destination); normal.prompt = "Keep my work"
    var sheet = normal
    ReferenceSheetContext(subjectKey: "one", name: "Corgi", kind: .character, description: "Short legs").apply(to: &sheet)
    var library = ImageWorkspaceLibrary(); library.record(normal, preview: nil); library.record(sheet, preview: nil)
    XCTAssertEqual(library.sessions.count, 2)
    XCTAssertEqual(library.sessions[destination.storageKey]?.draft.prompt, "Keep my work")
    XCTAssertTrue(sheet.prompt.contains("quadruped stays on four legs"))
  }
}
