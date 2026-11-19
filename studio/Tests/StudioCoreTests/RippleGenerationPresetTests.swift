import Foundation
import XCTest
@testable import StudioCore

final class RippleGenerationPresetTests: XCTestCase {
  let model = GenerationModelRef(id: "ripple", backend: "ripple", family: "ltx25", variant: "distilled")
  func library() -> GenerationLibrarySnapshot {
    .init(models: [.init(reference: model, name: "Ripple · LTX 2.5", workspaces: [.ripple], tasks: ["ripple"], supportsLoRAs: false,
      availability: .installed, binding: .ripple(profileID: "/models/distilled.json", adapterPath: "/models/ripple.safetensors"))])
  }
  func testRecipeRetainsSourceClockRangeReferencesPromptAndAudio() throws {
    var clip = Clip(engine: .movie); clip.sourcePath = "/source.mov"; clip.sourceIn = 2; clip.duration = 5
    var draft = RippleDraft(clip: clip, frameRate: 29.97); draft.width = 1312; draft.height = 736
    draft.prompt = "Keep my edit"; draft.references[0].path = "/edited.png"; draft.references[0].strength = 0.7
    var snapshot = try RipplePresetRecipe.capture(from: draft, resolvedLibrary: library(), legacyModel: model)
    snapshot.fields["seed"] = .integer(7); snapshot.fields["loraStrength"] = .number(1.5)
    let candidate = try RipplePresetRecipe.candidate(applying: snapshot, to: draft, resolvedLibrary: library())
    XCTAssertEqual(candidate.libraryModel, model); XCTAssertEqual(candidate.seed, 7); XCTAssertEqual(candidate.loraStrength, 1.5)
    XCTAssertEqual(candidate.sourcePath, draft.sourcePath); XCTAssertEqual(candidate.sourceIn, draft.sourceIn); XCTAssertEqual(candidate.duration, draft.duration)
    XCTAssertEqual(candidate.frameRate, draft.frameRate); XCTAssertEqual(candidate.width, draft.width); XCTAssertEqual(candidate.height, draft.height)
    XCTAssertEqual(candidate.prompt, draft.prompt); XCTAssertEqual(candidate.references, draft.references); XCTAssertEqual(candidate.audioPolicy, draft.audioPolicy)
    let text = String(decoding: try JSONEncoder().encode(snapshot), as: UTF8.self)
    XCTAssertFalse(text.contains("/source.mov")); XCTAssertFalse(text.contains("/models/")); XCTAssertFalse(text.contains("Keep my edit"))
    XCTAssertEqual(try JSONDecoder().decode(RippleDraft.self, from: JSONEncoder().encode(candidate)), candidate)
  }
  func testWrongModelMissingComponentsAndUnknownConditioningFieldsReject() throws {
    var clip = Clip(engine: .movie); clip.sourcePath = "/source.mov"
    let draft = RippleDraft(clip: clip, frameRate: 24)
    var snapshot = try RipplePresetRecipe.capture(from: draft, resolvedLibrary: library(), legacyModel: model)
    snapshot.fields["width"] = .integer(1024)
    XCTAssertThrowsError(try RipplePresetRecipe.candidate(applying: snapshot, to: draft, resolvedLibrary: library()))
    snapshot.fields.removeValue(forKey: "width")
    var missing = library(); missing.modelEntries[0].availability = .driveDisconnected
    XCTAssertThrowsError(try RipplePresetRecipe.candidate(applying: snapshot, to: draft, resolvedLibrary: missing))
    var wrong = library(); wrong.modelEntries[0].binding = .nativeVideo(profileID: "/models/dev.json")
    XCTAssertThrowsError(try RipplePresetRecipe.candidate(applying: snapshot, to: draft, resolvedLibrary: wrong))
  }
}
