import Foundation
import XCTest
@testable import InferenceContracts

final class NativeVideoJobEnvelopeTests: XCTestCase {
  private let jobID = UUID(uuidString: "11111111-2222-3333-4444-555555555555")!
  private let recipe = Data("abc".utf8)

  func testRoundTripBindsEngineOutputAndRecipeBytes() throws {
    let envelope = NativeVideoJobEnvelope(jobID: jobID, engine: .ltx25,
      recipePath: "/jobs/recipe.json", recipeData: recipe,
      outputDirectory: "/jobs/output", ffmpegPath: "/usr/bin/ffmpeg")
    let decoded = try NativeVideoJobEnvelope.decode(JSONEncoder().encode(envelope),
      expectedEngine: .ltx25, expectedOutputDirectory: "/jobs/output")
    XCTAssertEqual(decoded.jobID, jobID)
    XCTAssertEqual(decoded.ffmpegPath, "/usr/bin/ffmpeg")
    XCTAssertEqual(decoded.recipeSHA256,
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    XCTAssertNoThrow(try decoded.validateRecipe(recipe))
    XCTAssertThrowsError(try decoded.validateRecipe(Data("abd".utf8)))
  }

  func testRejectsWrongEngineOutputVersionAndUnknownFields() throws {
    let envelope = NativeVideoJobEnvelope(jobID: jobID, engine: .ltx25,
      recipePath: "/jobs/recipe.json", recipeData: recipe,
      outputDirectory: "/jobs/output", ffmpegPath: nil)
    let data = try JSONEncoder().encode(envelope)
    XCTAssertThrowsError(try NativeVideoJobEnvelope.decode(data,
      expectedEngine: .h3, expectedOutputDirectory: "/jobs/output"))
    XCTAssertThrowsError(try NativeVideoJobEnvelope.decode(data,
      expectedEngine: .ltx25, expectedOutputDirectory: "/jobs/other"))
    var object = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    object["version"] = 2
    XCTAssertThrowsError(try NativeVideoJobEnvelope.decode(JSONSerialization.data(withJSONObject: object),
      expectedEngine: .ltx25, expectedOutputDirectory: "/jobs/output"))
    object["version"] = 1
    object["ignoredReference"] = "/jobs/hidden.png"
    XCTAssertThrowsError(try NativeVideoJobEnvelope.decode(JSONSerialization.data(withJSONObject: object),
      expectedEngine: .ltx25, expectedOutputDirectory: "/jobs/output"))
  }

  func testRejectsRelativeOrUnboundedRequestFields() throws {
    let valid = NativeVideoJobEnvelope(jobID: jobID, engine: .h3,
      recipePath: "/jobs/recipe.json", recipeData: recipe,
      outputDirectory: "/jobs/output", ffmpegPath: nil)
    var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(valid)) as! [String: Any]
    object["recipePath"] = "recipe.json"
    XCTAssertThrowsError(try NativeVideoJobEnvelope.decode(JSONSerialization.data(withJSONObject: object),
      expectedEngine: .h3, expectedOutputDirectory: "/jobs/output"))
    object["recipePath"] = "/jobs/recipe.json"
    object["recipeSHA256"] = String(repeating: "a", count: 100000)
    XCTAssertThrowsError(try NativeVideoJobEnvelope.decode(JSONSerialization.data(withJSONObject: object),
      expectedEngine: .h3, expectedOutputDirectory: "/jobs/output"))
  }
}
