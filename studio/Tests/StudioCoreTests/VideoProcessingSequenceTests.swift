import Foundation
import XCTest
@testable import StudioCore

final class VideoProcessingSequenceTests: XCTestCase {
  func testVersionTwoPreservesExplicitH3FlashOrderAndLegacyReader() throws {
    var h3 = VideoProcessingStep(h3Regenerate: .init()); h3.id = "regenerate"
    var settings = FlashVSRSettings(); settings.backend = .swiftMLX; settings.variant = .base
    var flash = VideoProcessingStep(settings: settings); flash.id = "restore"; flash.input = "regenerate"
    let recipe = try VideoProcessingRecipe(steps: [h3, flash])
    XCTAssertEqual(recipe.format, "weetodd-video-processing-v2")
    XCTAssertTrue(recipe.hasH3Regenerate); XCTAssertTrue(recipe.hasFlashVSR)
    let data = try JSONEncoder().encode(recipe)
    XCTAssertEqual(try JSONDecoder().decode(VideoProcessingRecipe.self, from: data), recipe)
    let object = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    let steps = object["steps"] as! [[String: Any]]
    XCTAssertNil(steps[0]["settings"])
    XCTAssertNotNil(steps[0]["h3Regenerate"])
    var legacy = VideoProcessingRecipe(settings: .init()); legacy.format = "weetodd-video-processing-v1"
    XCTAssertEqual(try JSONDecoder().decode(VideoProcessingRecipe.self,
      from: JSONEncoder().encode(legacy)).format, "weetodd-video-processing-v1")
  }
  func testSequenceRejectsLoopsDuplicateIDsAndIgnoredFamilySettings() throws {
    var first = VideoProcessingStep(h3Regenerate: .init()); first.id = "h3"
    var second = VideoProcessingStep(settings: .init()); second.id = "flash"; second.input = "source"
    XCTAssertThrowsError(try VideoProcessingRecipe(steps: [first, second]))
    second.input = "h3"; second.id = "h3"
    XCTAssertThrowsError(try VideoProcessingRecipe(steps: [first, second]))
    let valid = try VideoProcessingRecipe(steps: [first])
    var root = try JSONSerialization.jsonObject(with: JSONEncoder().encode(valid)) as! [String: Any]
    var steps = root["steps"] as! [[String: Any]]
    steps[0]["settings"] = ["backend": "swiftMLX"]
    root["steps"] = steps
    XCTAssertThrowsError(try JSONDecoder().decode(VideoProcessingRecipe.self,
      from: JSONSerialization.data(withJSONObject: root)))
  }
}
