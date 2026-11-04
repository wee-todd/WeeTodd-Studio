import Foundation
import XCTest
@testable import H3MLX

final class H3FlexAdapterTests: XCTestCase {
  func testOfficialReleaseHeadersIdentifyTaskWithoutUsingFilename() throws {
    XCTAssertEqual(H3FlexAdapter.variant(headerSHA256: "d1a1a019dc4bf95f90d38b6a669b52ae776d41a92356712a6ce90b93539240f1",
      fileBytes: 1_956_172_176), .fl2va)
    XCTAssertEqual(H3FlexAdapter.variant(headerSHA256: "b4f84aa4ea051ccf3dbfb0ba728e00fc3ac5d45800d0e5d4cd78b4cf80408754",
      fileBytes: 1_956_172_176), .ref2va)
    XCTAssertNil(H3FlexAdapter.variant(headerSHA256: String(repeating: "a", count: 64), fileBytes: 1_956_172_176))
    XCTAssertNil(H3FlexAdapter.variant(headerSHA256: "d1a1a019dc4bf95f90d38b6a669b52ae776d41a92356712a6ce90b93539240f1", fileBytes: 100))
  }

  func testExplicitTaskProfilesRequireFullStrengthImmediateSixEvaluationSchedule() throws {
    for value in ["pika_flex_fl2va", "pika_flex_ref2va"] {
      let profile = try XCTUnwrap(H3LoRAProfile(rawValue: value))
      let adapter = try H3LoRAAdapter(url: URL(fileURLWithPath: "/unused.safetensors"), strength: 1, profile: profile)
      let method = try XCTUnwrap(H3SamplingMethod(rawValue: "pika_flex"))
      XCTAssertNoThrow(try adapter.validate(requestedSteps: 7, samplingMethod: method))
      XCTAssertThrowsError(try adapter.validate(requestedSteps: 5, samplingMethod: method))
      XCTAssertThrowsError(try adapter.validate(requestedSteps: 7, samplingMethod: .euler))
      let weak = try H3LoRAAdapter(url: adapter.url, strength: 0.5, profile: profile)
      XCTAssertThrowsError(try weak.validate(requestedSteps: 7, samplingMethod: method))
      let delayed = try H3LoRAAdapter(url: adapter.url, strength: 1, profile: profile, startAfterEvaluations: 1)
      XCTAssertThrowsError(try delayed.validate(requestedSteps: 7, samplingMethod: method))
    }
  }

  func testTaskAndSpecializedRouteAdmissionCannotSilentlyChangeFlexBehavior() throws {
    XCTAssertNoThrow(try H3FlexAdapter.validate(variant: .fl2va, task: "t2va", specialized: false))
    XCTAssertNoThrow(try H3FlexAdapter.validate(variant: .fl2va, task: "fl2va", specialized: false))
    XCTAssertNoThrow(try H3FlexAdapter.validate(variant: .ref2va, task: "ref2va", specialized: false))
    XCTAssertThrowsError(try H3FlexAdapter.validate(variant: .fl2va, task: "ref2va", specialized: false))
    XCTAssertThrowsError(try H3FlexAdapter.validate(variant: .ref2va, task: "t2va", specialized: false))
    XCTAssertThrowsError(try H3FlexAdapter.validate(variant: .fl2va, task: "t2va", specialized: true))
  }

  func testBasePartitionIsRequiredAndCannotCrossTaskFamilies() throws {
    XCTAssertNoThrow(try H3FlexAdapter.validateBase(partition: "FL2VA", variant: .fl2va))
    XCTAssertNoThrow(try H3FlexAdapter.validateBase(partition: "Ref2VA", variant: .ref2va))
    XCTAssertThrowsError(try H3FlexAdapter.validateBase(partition: "FL2VA", variant: .ref2va))
    XCTAssertThrowsError(try H3FlexAdapter.validateBase(partition: "Ref2VA", variant: .fl2va))
    XCTAssertThrowsError(try H3FlexAdapter.validateBase(partition: nil, variant: .fl2va))
  }

  func testInstalledOfficialAdaptersLoadWithDeclaredUnitScaleAndRejectWrongSampler() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_H3_FLEX_MODELS"] else {
      throw XCTSkip("Installed Pika H3 Flex adapters are optional")
    }
    for task in ["fl2va", "ref2va"] {
      let url = URL(fileURLWithPath: root).appendingPathComponent("pika_h3_flex_\(task)_comfy_bf16.safetensors")
      let profile = try H3FlexAdapter.resolvedProfile(url: url, selected: .auto)
      XCTAssertEqual(profile.rawValue, "pika_flex_\(task)")
      let file = try H3LoRAFile(url: url, strength: 1, requestedSteps: 7, samplingMethod: .pikaFlex)
      XCTAssertEqual(file.targetCount, 208)
      XCTAssertThrowsError(try H3LoRAFile(url: url, strength: 1, requestedSteps: 7, samplingMethod: .euler))
      XCTAssertThrowsError(try H3FlexAdapter.resolvedProfile(url: url, selected: .standard))
    }
  }
}
