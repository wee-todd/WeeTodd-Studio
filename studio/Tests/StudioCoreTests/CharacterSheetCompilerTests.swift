import XCTest
@testable import StudioCore

final class CharacterSheetCompilerTests: XCTestCase {
  func testIncompletePreviewKeepsExactPrefixWithoutInventingData() {
    let result = CharacterSheetCompiler.compile(.newDraft())
    XCTAssertTrue(result.prompt.hasPrefix(ReferenceSheetTemplate.characterSheetPrefix))
    XCTAssertFalse(result.canGenerate)
    XCTAssertTrue(result.diagnostics.contains { $0.fieldPath == "identity.species" })
    XCTAssertFalse(result.prompt.contains("unknown"))
    XCTAssertFalse(result.prompt.contains("heroic"))
  }

  func testUnsupportedSchemaCannotGenerate() {
    var sheet = CharacterSheetDefinition.newDraft()
    sheet.schemaVersion = 99
    let result = CharacterSheetCompiler.compile(sheet)
    XCTAssertFalse(result.canGenerate)
    XCTAssertTrue(result.diagnostics.contains { $0.code == "schema.unsupported" })
  }

  func testCompilerWalksThirteenSectionsInOrderWithExactRanges() {
    var sheet = CharacterSheetDefinition.newDraft()
    sheet.appearance.setText("identity.species", "gryphon")
    sheet.appearance.setText("identity.type", "fantasy creature")
    sheet.appearance.setChoice("body.plan", id: "quadruped", displayValue: "quadruped")
    sheet.appearance.setText("face.shape", "wedge-shaped head")
    sheet.appearance.setText("hair.style", "feathered crest")
    sheet.appearance.garments = [.init(order: 0, fields: ["type": .value(.text("red harness"))])]
    sheet.appearance.features = [.init(order: 0, fields: ["type": .value(.text("scar")), "placement": .value(.text("above the character's left eye"))])]
    sheet.appearance.surfaces = [.init(order: 0, fields: ["target": .value(.text("red harness")), "texture": .value(.text("creased"))])]
    let result = CharacterSheetCompiler.compile(sheet)
    XCTAssertTrue(result.canGenerate)
    XCTAssertEqual(result.sections.map(\.index), Array(1...13))
    for section in result.sections {
      XCTAssertEqual(String(result.prompt[section.range]), section.text)
    }
    XCTAssertTrue(result.prompt.contains("quadruped"))
    XCTAssertTrue(result.prompt.contains("stays on four legs"))
    XCTAssertTrue(result.prompt.contains("one row of four distinct panels"))
  }

  func testDictionaryOrderCannotChangePromptOrDigest() {
    let a: [String: CharacterFieldEntry] = [
      "identity.species": .value(.text("human")), "identity.type": .value(.text("person")),
      "face.shape": .value(.text("oval")), "body.build": .value(.text("broad"))
    ]
    let b = Dictionary(uniqueKeysWithValues: a.reversed())
    let first = CharacterSheetCompiler.compile(.init(appearance: .init(fields: a), settings: .newDraft()))
    let second = CharacterSheetCompiler.compile(.init(appearance: .init(fields: b), settings: .newDraft()))
    XCTAssertEqual(first.prompt, second.prompt)
    XCTAssertEqual(first.inputDigest, second.inputDigest)
  }

  func testRequiredPathInsertionOrderCannotChangeDigest() {
    var first = CharacterSheetDefinition.newDraft()
    first.appearance.setText("identity.species", "human")
    first.appearance.setText("identity.type", "person")
    var second = first
    first.settings.requiredFieldPaths = Set(["identity.species", "identity.type", "style.presetID"])
    var reordered = Set<String>(); reordered.insert("style.presetID"); reordered.insert("identity.type"); reordered.insert("identity.species")
    second.settings.requiredFieldPaths = reordered
    XCTAssertEqual(CharacterSheetCompiler.compile(first).inputDigest,
      CharacterSheetCompiler.compile(second).inputDigest)
  }

  func testAbsenceApplicabilityAndTypedStyleConflictStayOutOfPrompt() {
    var sheet = CharacterSheetDefinition.newDraft()
    sheet.appearance.setText("identity.species", "human")
    sheet.appearance.setText("identity.type", "person")
    sheet.appearance.setState("hair.facialHair", .explicitlyAbsent)
    sheet.appearance.setState("covering.pattern", .notApplicable)
    sheet.settings.setChoice("style.edgeTreatment", id: "anime", displayValue: "anime linework")
    let result = CharacterSheetCompiler.compile(sheet)
    XCTAssertTrue(result.prompt.contains("no facial hair"))
    XCTAssertFalse(result.prompt.contains("not applicable"))
    XCTAssertFalse(result.prompt.contains("anime linework"))
    XCTAssertTrue(result.diagnostics.contains { $0.code == "style.conflict" })
    XCTAssertFalse(result.canGenerate)
  }

  func testAuthoredAnimePrintIsNotMistakenForStyleConflictAndNoMeasurementsAreInvented() {
    var sheet = CharacterSheetDefinition.newDraft()
    sheet.appearance.setText("identity.species", "human")
    sheet.appearance.setText("identity.type", "person")
    sheet.appearance.garments = [.init(order: 0, fields: ["type": .value(.text("T-shirt")),
      "trim": .value(.text("authored anime print"))])]
    let result = CharacterSheetCompiler.compile(sheet)
    XCTAssertTrue(result.canGenerate)
    XCTAssertTrue(result.prompt.contains("authored anime print"))
    XCTAssertFalse(result.prompt.contains(" cm"))
    XCTAssertFalse(result.prompt.contains("heroic"))
  }

  func testSurfaceUUIDTargetRendersNamedGarmentOnce() {
    let garmentID = UUID(uuidString: "00000000-0000-0000-0000-000000000021")!
    var sheet = CharacterSheetDefinition.newDraft()
    sheet.appearance.setText("identity.species", "human")
    sheet.appearance.setText("identity.type", "person")
    sheet.appearance.garments = [.init(id: garmentID, order: 0, fields: [
      "type": .value(.text("field jacket")), "color": .value(.text("navy"))
    ])]
    sheet.appearance.surfaces = [.init(order: 0, fields: [
      "target": .value(.choice(id: garmentID.uuidString, displayValue: "selected garment")),
      "material": .value(.text("woven canvas"))
    ])]
    let result = CharacterSheetCompiler.compile(sheet)
    XCTAssertTrue(result.canGenerate)
    XCTAssertTrue(result.sections[7].text.contains("target: field jacket"))
    XCTAssertFalse(result.prompt.contains(garmentID.uuidString))
  }

  func testMissingSurfaceUUIDTargetBlocksGeneration() {
    var sheet = CharacterSheetDefinition.newDraft()
    sheet.appearance.setText("identity.species", "human")
    sheet.appearance.setText("identity.type", "person")
    sheet.appearance.surfaces = [.init(order: 0, fields: [
      "target": .value(.text("00000000-0000-0000-0000-000000000099")),
      "material": .value(.text("woven canvas"))
    ])]
    let result = CharacterSheetCompiler.compile(sheet)
    XCTAssertFalse(result.canGenerate)
    XCTAssertTrue(result.diagnostics.contains { $0.code == "surface.targetMissing" })
    XCTAssertFalse(result.prompt.contains("00000000-0000-0000-0000-000000000099"))
  }
}

private extension JSONEncoder {
  static var sorted: JSONEncoder { let value = JSONEncoder(); value.outputFormatting = [.sortedKeys]; return value }
}
