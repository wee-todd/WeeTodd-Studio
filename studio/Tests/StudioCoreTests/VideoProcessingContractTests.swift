import Foundation
import XCTest
@testable import StudioCore

final class VideoProcessingContractTests: XCTestCase {
  func testNativeThenFlashRecipeIsAcceptedAndPersistsItsOrder() throws {
    var recipe = VideoProcessingRecipe(pixelCQ: .init())
    var flash = VideoProcessingStep(settings: .init())
    flash.id = "flash"; flash.input = "restore"; recipe.steps.append(flash)
    XCTAssertNoThrow(try recipe.validate())
    let decoded = try JSONDecoder().decode(VideoProcessingRecipe.self, from: JSONEncoder().encode(recipe))
    XCTAssertEqual(decoded.steps.map(\.kind), ["native-pixel-cq", "flashvsr-turbo"])
  }
  private func document(audio: Bool = true, input: String = "source") throws -> Data {
    let clip = Clip(name: "Original H3", engine: .h3)
    var body = try JSONSerialization.jsonObject(with: JSONEncoder().encode(clip)) as! [String: Any]
    body["videoProcessing"] = ["format": "weetodd-video-processing-v1", "steps": [
      ["id": "restore", "kind": "flashvsr-turbo", "input": input,
       "settings": ["width": 1920, "height": 1088, "preserveAudio": audio]]
    ]]
    return try JSONSerialization.data(withJSONObject: body)
  }
  func testClipPersistsRestorationWithoutInvalidatingGeneration() throws {
    let decoded = try JSONDecoder().decode(Clip.self, from: document())
    let roundTrip = try JSONSerialization.jsonObject(with: JSONEncoder().encode(decoded)) as! [String: Any]
    let recipe = roundTrip["videoProcessing"] as? [String: Any]
    XCTAssertEqual(recipe?["format"] as? String, "weetodd-video-processing-v1")
    let step = (recipe?["steps"] as? [[String: Any]])?.first
    XCTAssertEqual((step?["settings"] as? [String: Any])?["decoder"] as? String, "tiny")
    var original = decoded
    var raw = roundTrip; raw.removeValue(forKey: "videoProcessing")
    original = try JSONDecoder().decode(Clip.self, from: JSONSerialization.data(withJSONObject: raw))
    XCTAssertEqual(decoded.generationFingerprint, original.generationFingerprint)
  }
  func testStepRejectsUnknownFieldsAndPortableExampleDecodes() throws {
    var raw = try JSONSerialization.jsonObject(with: document()) as! [String: Any]
    var recipe = raw["videoProcessing"] as! [String: Any]
    var steps = recipe["steps"] as! [[String: Any]]; steps[0]["decodeer"] = "vae"
    recipe["steps"] = steps; raw["videoProcessing"] = recipe
    XCTAssertThrowsError(try JSONDecoder().decode(Clip.self, from: JSONSerialization.data(withJSONObject: raw)))
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let example = try JSONDecoder().decode(VideoProcessingRecipe.self,
      from: Data(contentsOf: repo.appendingPathComponent("examples/flashvsr_turbo_recipe.json")))
    XCTAssertEqual(example.steps.first?.settings.decoder, .tiny)
    XCTAssertEqual(example.steps.first?.settings.width, 1920)
  }
  func testProcessingChangesDuringGenerationCanStillAcceptOriginal() throws {
    let clip = try JSONDecoder().decode(Clip.self, from: document())
    var changed = clip; changed.videoProcessing!.steps[0].settings.decoder = .vae
    XCTAssertTrue(changed.matchesGenerationSubmission(clip))
    changed.prompt = "different shot"
    XCTAssertFalse(changed.matchesGenerationSubmission(clip))
  }
  func testRepeatedRestorationRejectsUntilLosslessAVTransportIsSupported() throws {
    var recipe = VideoProcessingRecipe(); var next = recipe.steps[0]
    next.id = "second"; next.input = "restore"; recipe.steps.append(next)
    XCTAssertThrowsError(try recipe.validate())
    XCTAssertThrowsError(try JSONDecoder().decode(VideoProcessingRecipe.self, from: JSONEncoder().encode(recipe)))
  }
  func testRecipeCannotDropAudioOrReferenceItsOwnOutput() throws {
    XCTAssertThrowsError(try JSONDecoder().decode(Clip.self, from: document(audio: false)))
    XCTAssertThrowsError(try JSONDecoder().decode(Clip.self, from: document(input: "restore")))
  }
}
