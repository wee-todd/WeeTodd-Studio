import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3TokenRefinerTests: XCTestCase {
  func testCombinedContextReachesCheckpointAdmissionAndOverflowFailsBeforeReads() throws {
    try Device.withDefaultDevice(.cpu) {
      let missing = URL(fileURLWithPath:"/missing/weetodd-refiner-context.safetensors")
      for rows in [1056,2048] {
        XCTAssertThrowsError(try H3TokenRefiner.evaluateBlock(checkpointURL:missing,index:0,
          input:MLXArray.zeros([1,rows,5376],dtype:.bfloat16))) {
          XCTAssertEqual($0 as? H3CheckpointError,.invalid("H3 checkpoint is not a local regular file."))
        }
      }
      for rows in [0,2049] {
        XCTAssertThrowsError(try H3TokenRefiner.evaluateBlock(checkpointURL:missing,index:0,
          input:MLXArray.zeros([1,rows,5376],dtype:.bfloat16))) {
          XCTAssertTrue(String(describing:$0).contains("Invalid H3 text refiner"),String(describing:$0))
        }
      }
    }
  }

  func testInstalledRefinerAcceptsCompleteFullImageContext() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Set the installed H3 transformer for the full mixed-context refiner check.")
    }
    let rows = 1108
    let token = MLXArray((0..<rows).map(Float.init),[1,rows,1])
    let feature = MLXArray((0..<5376).map(Float.init),[1,1,5376])
    let input = sin(token * Float(0.007) + feature * Float(0.011)).asType(.bfloat16)
    let result = try H3TokenRefiner.evaluate(checkpointURL:URL(fileURLWithPath:checkpoint),input:input)
    XCTAssertEqual(result.shape,[1,rows,5376])
    XCTAssertEqual(result.dtype,.bfloat16)
    XCTAssertTrue(all(isFinite(result)).item(Bool.self))
  }

  func testInstalledTwoBlockRefinerMatchesReference() throws {
    guard let checkpoint = ProcessInfo.processInfo.environment["WEETODD_H3_TEST_CHECKPOINT"],
      let inputFixture = ProcessInfo.processInfo.environment["WEETODD_H3_PATCH_ORACLE"],
      let fixture = ProcessInfo.processInfo.environment["WEETODD_H3_REFINER_ORACLE"] else {
      throw XCTSkip("Set H3 checkpoint, input projection and refiner oracle paths.")
    }
    let inputData = try Data(contentsOf: URL(fileURLWithPath: inputFixture)
      .appendingPathComponent("condition_proj-output.u16"))
    let input = inputData.withUnsafeBytes {
      MLXArray($0, [1, 4, 5376], type: UInt16.self).view(dtype: .bfloat16)
    }
    var actual = input
    for index in 0..<2 {
      actual = try H3TokenRefiner.evaluateBlock(checkpointURL: URL(fileURLWithPath: checkpoint),
        index: index, input: actual) { name, value in
        guard index == 0 else { return }
        let stageData = try Data(contentsOf: URL(fileURLWithPath: fixture)
          .appendingPathComponent("\(name).u16"))
        let expectedStage = stageData.withUnsafeBytes {
          MLXArray($0, value.shape, type: UInt16.self).view(dtype: .bfloat16)
        }
        let difference = max(abs(value.asType(.float32) - expectedStage.asType(.float32)))
          .item(Float.self)
        XCTAssertEqual(difference, 0, name)
      }
      let expectedData = try Data(contentsOf: URL(fileURLWithPath: fixture)
        .appendingPathComponent("block\(index).u16"))
      let expected = expectedData.withUnsafeBytes {
        MLXArray($0, [1, 4, 5376], type: UInt16.self).view(dtype: .bfloat16)
      }
      XCTAssertEqual(max(abs(actual.asType(.float32) - expected.asType(.float32)))
        .item(Float.self), 0, "block \(index)")
    }
    let final = try H3TokenRefiner.evaluate(checkpointURL: URL(fileURLWithPath: checkpoint),
      input: input)
    let finalData = try Data(contentsOf: URL(fileURLWithPath: fixture)
      .appendingPathComponent("final.u16"))
    let finalExpected = finalData.withUnsafeBytes {
      MLXArray($0, [1, 4, 5376], type: UInt16.self).view(dtype: .bfloat16)
    }
    XCTAssertEqual(max(abs(final.asType(.float32) - finalExpected.asType(.float32)))
      .item(Float.self), 0)
  }
}
