import XCTest
@testable import StudioCore

final class ReferenceInputTests: XCTestCase {
  func testLocalReferenceChoicesDistinguishAppearanceMotionAndAudio() {
    for engine in [Engine.h3, .ltx23, .ltx25] {
      let clip = Clip(name: "Shot", engine: engine)
      let movie = MediaAsset(name: "Movie", kind: .video)
      XCTAssertEqual(Set(clip.referenceActions(for: movie).map(\.id)),
        ["movieAppearance", "movieMotion", "preprocessedControl"])
      let audio = MediaAsset(name: "Audio", kind: .audio)
      XCTAssertEqual(clip.referenceActions(for: audio).map(\.id),
        engine == .h3 ? ["audioDriver", "audioReference"] : ["audioDriver"])
      XCTAssertFalse(clip.canAssignMedia(audio, role: .first))
      XCTAssertFalse(clip.canAssignMedia(movie, role: .audioDriver))
    }
  }

  func testIngredientsIsAtomicAndInfersTheCorrectModelTask() throws {
    for engine in [Engine.ltx23, .ltx25] {
      var clip = Clip(name: "Shot", engine: engine)
      clip.generationSelection = GenerationSelection()
      let image = MediaAsset(name: "Cast sheet", kind: .image)
      let action = try XCTUnwrap(clip.referenceActions(for: image).first { $0.id == "ingredients" })
      try clip.attachReference(image, action: action)
      XCTAssertEqual(clip.attachments.count, 1)
      XCTAssertEqual(clip.attachments[0].controlType, "ingredients_reference_sheet")
      XCTAssertEqual(clip.inferredTask, engine == .ltx23 ? "ref2va" : "control")
      XCTAssertEqual(clip.attachments[0].description, "Cast sheet")
    }
  }

  func testLTX25IngredientsShowsItsSceneQualityLimitBeforeAttachment() throws {
    let image = MediaAsset(name: "Cast sheet", kind: .image)
    let action = try XCTUnwrap(Clip(name: "Shot", engine: .ltx25)
      .referenceActions(for: image).first { $0.id == "ingredients" })
    XCTAssertTrue(action.label.contains("Experimental"))
    XCTAssertTrue(action.detail.contains("results may reproduce the sheet layout"))
    XCTAssertEqual(action.role, .control)
    XCTAssertEqual(action.controlType, "ingredients_reference_sheet")
    let legacy = try XCTUnwrap(Clip(name: "Legacy", engine: .ltx23)
      .referenceActions(for: image).first { $0.id == "ingredients" })
    XCTAssertFalse(legacy.label.contains("Experimental"))
    XCTAssertTrue(legacy.detail.contains("768 × 448"))
  }

  func testDrawThingsReferencesRequireH3ReferenceModelAndStayImageOnly() throws {
    var clip = Clip(name: "DT", engine: .drawThings)
    clip.drawThings = DrawThingsSelection(profileID: "local", modelID: "h3", modelFamily: "minimaxH3")
    let image = MediaAsset(name: "Character", kind: .image)
    XCTAssertFalse(clip.canAssignDrawThingsInput(image, role: .reference))
    clip.drawThings?.modelModifier = "ref2va"
    XCTAssertTrue(clip.canAssignDrawThingsInput(image, role: .reference))
    XCTAssertFalse(clip.canAssignDrawThingsInput(image, role: .first))
    XCTAssertTrue(clip.referenceActions(for: MediaAsset(name: "Movie", kind: .video)).isEmpty)
    XCTAssertTrue(clip.referenceActions(for: MediaAsset(name: "Audio", kind: .audio)).isEmpty)
    try clip.attachReference(image, action: XCTUnwrap(clip.referenceActions(for: image).first))
    XCTAssertEqual(clip.inferredTask, "ref2va")
    XCTAssertTrue(clip.drawThingsConditioningIssues(assets: [image], fileExists: { _ in true }).isEmpty)
  }
}
