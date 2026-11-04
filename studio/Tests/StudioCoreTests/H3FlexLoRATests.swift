import Foundation
import XCTest
@testable import StudioCore

final class H3FlexLoRATests: XCTestCase {
  func testFlexSettingsRequireSixEvaluationsAndTrainedStrength() throws {
    for raw in ["pika_flex_fl2va", "pika_flex_ref2va"] {
      let profile = try XCTUnwrap(H3LoRAProfile(rawValue: raw))
      let settings = H3LoRASettings(profile: profile)
      XCTAssertNoThrow(try settings.validate(strength: 1, evaluations: 6, samplingMethod: "pika_flex"))
      XCTAssertThrowsError(try settings.validate(strength: 0.5, evaluations: 6, samplingMethod: "pika_flex"))
      XCTAssertThrowsError(try settings.validate(strength: 1, evaluations: 6, samplingMethod: "euler"))
      XCTAssertThrowsError(try settings.validate(strength: 1, evaluations: 4, samplingMethod: "pika_flex"))
      XCTAssertEqual(try JSONDecoder().decode(H3LoRASettings.self, from: JSONEncoder().encode(settings)), settings)
    }
    XCTAssertNotNil(NativeH3SamplingMethod(rawValue: "pika_flex"))
  }

  func testInstalledFlexIsRecognizedAndComposedWithoutFilenameGuessing() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_H3_FLEX_MODELS"] else {
      throw XCTSkip("Set WEETODD_H3_FLEX_MODELS for publisher-header qualification")
    }
    for task in ["fl2va", "ref2va"] {
      let url = URL(fileURLWithPath: root).appendingPathComponent("pika_h3_flex_\(task)_comfy_bf16.safetensors")
      let info = try NativeLoRAInspection.inspect(url)
      XCTAssertEqual(info["loraProfile"] as? String, "pika_flex_\(task)")
      XCTAssertEqual(info["loraLayout"] as? String, "contiguous_qkv")
      XCTAssertEqual(info["h3DeclaredEvaluations"] as? Int, 6)
      XCTAssertThrowsError(try NativeLoRAInspection.validateH3Sampling(path: url.path, selectedProfile: "standard", schedulePoints: 7))
      let result = try NativeH3LoRAComposition.compose(componentPairs: [[url.path, 1.0]], rootStack: nil,
        attachments: [], assets: [], schedulePoints: 31, explicitEvaluations: nil, samplingMethod: "euler",
        task: task, explicitSamplingMethod: nil)
      XCTAssertEqual(result.schedulePoints, 7)
      XCTAssertEqual(result.samplingMethod, "pika_flex")
      let descriptors = try XCTUnwrap(result.descriptors?["adapters"] as? [[String: Any]])
      XCTAssertEqual(descriptors.first?["profile"] as? String, "pika_flex_\(task)")
      XCTAssertThrowsError(try NativeH3LoRAComposition.compose(componentPairs: [[url.path, 1.0]], rootStack: nil,
        attachments: [], assets: [], schedulePoints: 31, explicitEvaluations: nil, samplingMethod: "euler",
        task: task, explicitSamplingMethod: "euler"))
      XCTAssertThrowsError(try NativeH3LoRAComposition.compose(componentPairs: [[url.path, 1.0]], rootStack: nil,
        attachments: [], assets: [], schedulePoints: 5, explicitEvaluations: 4, samplingMethod: "pika_flex",
        task: task, explicitSamplingMethod: "pika_flex"))
      XCTAssertThrowsError(try NativeH3LoRAComposition.compose(componentPairs: [[url.path, 1.0]], rootStack: nil,
        attachments: [], assets: [], schedulePoints: 7, explicitEvaluations: 6, samplingMethod: "pika_flex",
        task: task == "ref2va" ? "t2v" : "ref2va", explicitSamplingMethod: "pika_flex"))
    }
  }
}
