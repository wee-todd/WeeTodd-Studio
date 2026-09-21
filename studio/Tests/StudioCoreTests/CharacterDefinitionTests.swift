import XCTest
@testable import StudioCore

final class CharacterDefinitionTests: XCTestCase {
  func testStyleExtractionOnlyIncludesRenderingCameraAndLighting() {
    let fields = CharacterFieldCatalog.shared.fields.filter { $0.extractionRoles.contains("style") }
    XCTAssertFalse(fields.isEmpty)
    XCTAssertTrue(fields.allSatisfy { (9...11).contains($0.section) })
    XCTAssertFalse(fields.contains { ["body.pose", "face.expression"].contains($0.key) })
    XCTAssertEqual(CharacterFieldCatalog.shared.field(for: "body.pose")?.extractionRoles, ["text"])
  }
  func testNewDraftDoesNotInventIdentityAndRoundTrips() throws {
    let sheet = CharacterSheetDefinition.newDraft()
    XCTAssertEqual(sheet.settings.stylePresetID, "photograph")
    XCTAssertEqual(sheet.appearance.fields["identity.species"]?.state, .unspecified)
    XCTAssertTrue(sheet.settings.requiredFieldPaths.contains("identity.species"))
    let restored = try JSONDecoder().decode(CharacterSheetDefinition.self,
      from: JSONEncoder().encode(sheet))
    XCTAssertEqual(restored, sheet)
  }

  func testTypedValuesPreserveCustomColorStatesAndClearedDefaults() throws {
    var sheet = CharacterSheetDefinition.newDraft()
    sheet.appearance.setColor("eyes.color", red: 0.1, green: 0.2, blue: 0.3, alpha: 1,
      displayValue: "storm-glass teal")
    sheet.appearance.setState("hair.facialHair", .explicitlyAbsent)
    sheet.appearance.setState("hair.length", .notApplicable)
    sheet.settings.setState("face.expression", .unspecified)
    let restored = try JSONDecoder().decode(CharacterSheetDefinition.self,
      from: JSONEncoder().encode(sheet))
    XCTAssertEqual(restored.appearance.fields["eyes.color"]?.value,
      .color(.init(red: 0.1, green: 0.2, blue: 0.3, alpha: 1, displayValue: "storm-glass teal")))
    XCTAssertEqual(restored.appearance.fields["hair.facialHair"]?.state, .explicitlyAbsent)
    XCTAssertEqual(restored.appearance.fields["hair.length"]?.state, .notApplicable)
    XCTAssertEqual(restored.settings.fields["face.expression"]?.state, .unspecified)
    XCTAssertEqual(restored.appearance.entry(at: "eyes.color")?.value?.displayString, "storm-glass teal")
  }

  func testCatalogRejectsInvalidUnitOversizedTextAndControlTokens() throws {
    let catalog = CharacterFieldCatalog.shared
    XCTAssertFalse(catalog.validate(.value(.measurement(value: 180, unit: "pixels")),
      at: "body.height", appearance: .init()).isEmpty)
    XCTAssertFalse(catalog.validate(.value(.measurement(value: 0, unit: "cm")),
      at: "body.height", appearance: .init()).isEmpty)
    XCTAssertFalse(catalog.validate(.value(.color(.init(red: .infinity, green: 0, blue: 0, displayValue: "bad"))),
      at: "eyes.color", appearance: .init()).isEmpty)
    XCTAssertFalse(catalog.validate(.value(.text(String(repeating: "x", count: 241))),
      at: "face.shape", appearance: .init()).isEmpty)
    XCTAssertFalse(catalog.validate(.value(.text("normal <lora:bad:1>")),
      at: "face.shape", appearance: .init()).isEmpty)
  }

  func testFullPathSetterCreatesStableRepeatableRecord() {
    let id = UUID(uuidString: "00000000-0000-0000-0000-000000000010")!
    var appearance = CharacterAppearance()
    appearance.setEntry(.value(.text("jacket")), at: "garments[\(id.uuidString)].type")
    appearance.setEntry(.value(.text("outer")), at: "garments[\(id.uuidString)].layer")
    XCTAssertEqual(appearance.garments.count, 1)
    XCTAssertEqual(appearance.garments[0].id, id)
    XCTAssertEqual(appearance.entry(at: "garments[\(id.uuidString)].layer")?.displayString, "outer")
  }

  func testMalformedRepeatablePathsAreRejectedWithoutTrapping() {
    let catalog = CharacterFieldCatalog.shared
    for path in ["garments[", "garments]id[.type", "garments[00000000-0000-0000-0000-000000000001]", "garments[bad].type", "garments[00000000-0000-0000-0000-000000000001]."] {
      XCTAssertNil(catalog.field(for: path), "unexpectedly accepted \(path)")
      XCTAssertNil(CharacterAppearance().entry(at: path))
    }
    XCTAssertNotNil(catalog.field(for: "garments[].type"))
    XCTAssertNil(CharacterAppearance().entry(at: "garments[].type"))
  }

  func testSettingsEncodeRequiredPathsInCanonicalOrder() throws {
    let first = CharacterSheetSettings(stylePresetID: "photograph",
      requiredFieldPaths: Set(["identity.type", "identity.species", "style.presetID"]))
    var secondPaths = Set<String>()
    secondPaths.insert("style.presetID"); secondPaths.insert("identity.species"); secondPaths.insert("identity.type")
    let second = CharacterSheetSettings(stylePresetID: "photograph", requiredFieldPaths: secondPaths)
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    XCTAssertEqual(try encoder.encode(first), try encoder.encode(second))
    let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(first)) as? [String: Any])
    XCTAssertEqual(object["requiredFieldPaths"] as? [String], ["identity.species", "identity.type", "style.presetID"])
  }

  func testCatalogAllowsAuthoredTextExtractionForEveryAppearanceField() {
    let appearanceFields = CharacterFieldCatalog.shared.fields.filter { $0.ownership == .appearance }
    XCTAssertFalse(appearanceFields.isEmpty)
    XCTAssertTrue(appearanceFields.allSatisfy { $0.extractionRoles.contains("text") })
    XCTAssertTrue(CharacterFieldCatalog.shared.field(for: "identity.authoredAge")?.extractionRoles.contains("text") == true)
  }

  func testGarmentOrderingIsStableAndCatalogResolvesRecordFields() throws {
    let outer = CharacterRepeatableRecord(id: UUID(uuidString: "00000000-0000-0000-0000-000000000002")!, order: 2,
      fields: ["type": .value(.text("coat")), "layer": .value(.choice(id: "outer", displayValue: "outer"))])
    let inner = CharacterRepeatableRecord(id: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!, order: 1,
      fields: ["type": .value(.text("shirt")), "layer": .value(.choice(id: "base", displayValue: "base"))])
    let appearance = CharacterAppearance(garments: [outer, inner])
    XCTAssertEqual(appearance.orderedGarments.map(\.id), [inner.id, outer.id])
    let path = "garments[\(inner.id.uuidString)].type"
    XCTAssertEqual(CharacterFieldCatalog.shared.resolvedField(path: path, appearance: appearance)?.catalog.key,
      "garments[].type")
  }
}
