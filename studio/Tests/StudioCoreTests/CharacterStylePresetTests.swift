import XCTest
@testable import StudioCore

final class CharacterStylePresetTests: XCTestCase {
  func testRegistryContainsAllVersionOnePresets() {
    XCTAssertEqual(CharacterStylePresetRegistry.all.map(\.id), [
      "photograph", "cinematicPhotograph", "realistic3D", "stylized3D", "anime", "comic",
      "oilPainting", "conceptArt", "clay", "sculpture"
    ])
    XCTAssertTrue(CharacterStylePresetRegistry.all.allSatisfy { $0.version == 1 })
  }

  func testPhotographApplicabilityIsForensic() throws {
    var robot = CharacterAppearance()
    robot.setChoice("identity.species", id: "robot", displayValue: "robot")
    robot.setChoice("face.covering", id: "metal", displayValue: "metal")
    let robotStyle = try CharacterStylePresetRegistry.resolve(id: "photograph", version: 1, appearance: robot)
    XCTAssertFalse(robotStyle.clauses.contains { $0.text.contains("skin texture") })

    var human = CharacterAppearance()
    human.setChoice("identity.species", id: "human", displayValue: "human")
    human.setChoice("face.covering", id: "skin", displayValue: "skin")
    human.setText("hair.style", "uneven fringe")
    human.garments = [.init(order: 0, fields: ["type": .value(.text("shirt"))])]
    human.surfaces = [.init(order: 0, fields: ["target": .value(.text("shirt")), "material": .value(.text("woven textile"))])]
    let humanStyle = try CharacterStylePresetRegistry.resolve(id: "photograph", version: 1, appearance: human)
    XCTAssertTrue(humanStyle.clauses.contains { $0.text.contains("skin texture") })
    XCTAssertTrue(humanStyle.clauses.contains { $0.text.contains("hair strands") })
    XCTAssertTrue(humanStyle.clauses.contains { $0.text.contains("textile detail") })
  }

  func testPresetSwitchNeverMutatesAppearanceBytes() throws {
    var appearance = CharacterAppearance()
    appearance.setText("identity.species", "human")
    appearance.garments = [.init(order: 0, fields: ["type": .value(.text("anime print T-shirt"))])]
    let before = try JSONEncoder.sorted.encode(appearance)
    for preset in CharacterStylePresetRegistry.all {
      _ = try CharacterStylePresetRegistry.resolve(id: preset.id, version: preset.version, appearance: appearance)
      XCTAssertEqual(try JSONEncoder.sorted.encode(appearance), before)
    }
  }

  func testUnknownPresetVersionDoesNotRewriteHistoricalMeaning() {
    XCTAssertThrowsError(try CharacterStylePresetRegistry.resolve(id: "photograph", version: 2,
      appearance: .init()))
  }
}

private extension JSONEncoder {
  static var sorted: JSONEncoder { let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]; return encoder }
}
