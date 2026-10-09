import Foundation
import MLX
import MLXRandom
import XCTest
@testable import H3MLX

final class H3MPPProjectionTests: XCTestCase {
  func testTaskEligibilityPreservesQualifiedOrdinaryPathsAndRejectsUnqualifiedContexts() {
    // Canonical ordinary BF16 T2VA and Ref2VA remain eligible. FastH3's
    // ordinary T2VA admission is unchanged; packed Q8 never selects MPP.
    XCTAssertTrue(H3MPPProjection.isTaskEligible(.t2va, isRefinement: false))
    XCTAssertTrue(H3MPPProjection.isTaskEligible(.ref2va, isRefinement: false))
    XCTAssertTrue(H3MPPProjection.isTaskEligible(.fl2va, isRefinement: false))
    for task in [H3MPPProjection.ExecutionTask.continuation,
      .refinement, .motionFidelity] {
      XCTAssertFalse(H3MPPProjection.isTaskEligible(task, isRefinement: false))
    }
    for task in [H3MPPProjection.ExecutionTask.t2va, .ref2va, .fl2va] {
      XCTAssertFalse(H3MPPProjection.isTaskEligible(task, isRefinement: true))
      for contextFrames in [-1, 1, 22, Int.max] {
        XCTAssertFalse(H3MPPProjection.isTaskEligible(task, contextFrames: contextFrames,
          isRefinement: false))
      }
    }
  }

  func testFirstUseBF16ComparisonDistinguishesSignedZeroOnCPU() {
    Device.withDefaultDevice(.cpu) {
      let positive = MLXArray([UInt16(0x0000), UInt16(0x3f80)]).view(dtype: .bfloat16)
      let negative = MLXArray([UInt16(0x8000), UInt16(0x3f80)]).view(dtype: .bfloat16)
      XCTAssertTrue(all(positive .== negative).item(Bool.self))
      XCTAssertFalse(H3MPPProjection.firstUseMatches(reference: positive, candidate: negative))
      XCTAssertTrue(H3MPPProjection.firstUseMatches(reference: positive, candidate: positive))
      XCTAssertFalse(H3MPPProjection.firstUseMatches(reference: positive,
        candidate: positive.asType(.float32)))
      XCTAssertFalse(H3MPPProjection.firstUseMatches(reference: positive,
        candidate: positive.reshaped([1, 2])))
    }
  }

  func testCapabilityGateIsPureAndLimitedToMeasuredBF16GPU() {
    XCTAssertTrue(H3MPPProjection.isEligible(macOSMajor: 26,
      architecture: "applegpu_g15d", isGPU: true,
      sourceDType: .bfloat16, weightDType: .bfloat16))
    XCTAssertTrue(H3MPPProjection.isEligible(macOSMajor: 27,
      architecture: "APPLEGPU_G15D", isGPU: true,
      sourceDType: .bfloat16, weightDType: .bfloat16))
    for major in [14, 25] {
      XCTAssertFalse(H3MPPProjection.isEligible(macOSMajor: major,
        architecture: "applegpu_g15d", isGPU: true,
        sourceDType: .bfloat16, weightDType: .bfloat16))
    }
    for architecture in ["", "Unknown", "applegpu_g16", "applegpu_g17"] {
      XCTAssertFalse(H3MPPProjection.isEligible(macOSMajor: 26,
        architecture: architecture, isGPU: true,
        sourceDType: .bfloat16, weightDType: .bfloat16))
    }
    XCTAssertFalse(H3MPPProjection.isEligible(macOSMajor: 26,
      architecture: "applegpu_g15d", isGPU: false,
      sourceDType: .bfloat16, weightDType: .bfloat16))
    for types in [(DType.float16, DType.bfloat16), (.bfloat16, .float32)] {
      XCTAssertFalse(H3MPPProjection.isEligible(macOSMajor: 26,
        architecture: "applegpu_g15d", isGPU: true,
        sourceDType: types.0, weightDType: types.1))
    }
  }

  func testOnlyFeedForwardOutputUsesMeasuredLargerTile() {
    let fc2 = H3MPPProjection.tile(weightShape: [5376, 14336])
    XCTAssertEqual(fc2.rows, 64)
    XCTAssertEqual(fc2.columns, 128)
    XCTAssertEqual(fc2.simdgroups, 8)
    for shape in [[21504, 5376], [5376, 7168], [28672, 5376], [128, 256]] {
      let tile = H3MPPProjection.tile(weightShape: shape)
      XCTAssertEqual(tile.rows, 32)
      XCTAssertEqual(tile.columns, 64)
      XCTAssertEqual(tile.simdgroups, 2)
    }
  }

  func testInvalidGeometryIsRejectedBeforeKernelSubmission() {
    for shapes in [([32], [64, 32]), ([1, 32], [32]), ([1, 31], [64, 32])] {
      let source = MLXArray.zeros(shapes.0, dtype: .bfloat16)
      let weight = MLXArray.zeros(shapes.1, dtype: .bfloat16)
      XCTAssertThrowsError(try H3MPPProjection.apply(source: source,
        weight: weight, enabled: true))
    }
  }

  private func assertExact(_ source: MLXArray, _ weight: MLXArray,
    enabled: Bool = true, file: StaticString = #filePath, line: UInt = #line) throws {
    let expected = matmul(source, weight.T)
    let actual = try H3MPPProjection.apply(source: source, weight: weight, enabled: enabled)
    XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
    XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
    XCTAssertEqual(sum(actual .!= expected).item(Int.self), 0, file: file, line: line)
  }

  func testDisabledUnsupportedPrecisionAndUnalignedWidthsRetainStandardMLX() throws {
    let source = MLXRandom.normal([2, 7, 33], key: MLXRandom.key(720)).asType(.bfloat16)
    let weight = MLXRandom.normal([65, 33], key: MLXRandom.key(721)).asType(.bfloat16)
    try assertExact(source, weight)
    for dtype in [DType.float16, .float32] {
      try assertExact(source.asType(dtype), weight.asType(dtype))
    }
    try assertExact(source, weight, enabled: false)
  }

  func testUnqualifiedSmallMatricesRetainStandardBF16ForRaggedRows() throws {
    guard H3MPPProjection.isAvailable else {
      throw XCTSkip("MPP BF16 kernel qualification requires the measured macOS 26+ M3 Ultra GPU.")
    }
    let weight = MLXRandom.normal([128, 256], key: MLXRandom.key(722)).asType(.bfloat16)
    for rows in [1, 31, 32, 33, 65] {
      let source = MLXRandom.normal([1, rows, 256], key: MLXRandom.key(UInt64(723 + rows)))
        .asType(.bfloat16)
      try assertExact(source, weight)
      try assertExact(MLXArray.ones(source.shape, dtype: .bfloat16) * 0.25,
        MLXArray.ones(weight.shape, dtype: .bfloat16) * -0.5)
    }
  }

  func testUnqualifiedLeadingDimensionsAndStridedMatricesRetainStandardMLX() throws {
    guard H3MPPProjection.isAvailable else {
      throw XCTSkip("MPP BF16 strided-input qualification requires the measured M3 Ultra GPU.")
    }
    let source = MLXRandom.normal([2, 9, 512], key: MLXRandom.key(724))
      .asType(.bfloat16)[.ellipsis, .stride(by: 2)]
    let weight = MLXRandom.normal([256, 128], key: MLXRandom.key(725))
      .asType(.bfloat16).T
    try assertExact(source, weight)
  }

  func testOptInProductionFC2TileMatchesStandardBF16() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_H3_MPP_LARGE_PROJECTION_TEST"] == "1",
      H3MPPProjection.isAvailable else {
      throw XCTSkip("Opt-in production-size H3 FC2 MPP qualification.")
    }
    let source = MLXRandom.normal([1, 65, 14336], key: MLXRandom.key(726)).asType(.bfloat16)
    let weight = MLXRandom.normal([5376, 14336], key: MLXRandom.key(727)).asType(.bfloat16)
    let scope = UUID().uuidString
    defer { H3MPPProjection.forget(scope: scope) }
    let expected = matmul(source, weight.T)
    for _ in 0..<2 {
      let output = try H3MPPProjection.apply(source: source, weight: weight,
        enabled: true, verificationScope: scope)
      XCTAssertEqual(sum(output .!= expected).item(Int.self), 0)
    }
    let status = H3MPPProjection.verificationStatus(scope: scope)
    XCTAssertEqual(status.verified + status.fallback, 1)
    XCTAssertEqual(status.eligibleCalls, 2)
    XCTAssertEqual(status.firstUseReferenceCalls, 1)
    XCTAssertEqual(status.mppCalls, status.verified)
    XCTAssertEqual(status.knownFallbackCalls, status.fallback)
    H3MPPProjection.forget(scope: scope)
    XCTAssertEqual(H3MPPProjection.verificationStatus(scope: scope).verified, 0)
    XCTAssertEqual(H3MPPProjection.verificationStatus(scope: scope).fallback, 0)
    XCTAssertEqual(H3MPPProjection.verificationStatus(scope: scope).eligibleCalls, 0)
    XCTAssertEqual(H3MPPProjection.verificationStatus(scope: scope).mppCalls, 0)
    XCTAssertEqual(H3MPPProjection.verificationStatus(scope: scope).knownFallbackCalls, 0)
    XCTAssertEqual(H3MPPProjection.verificationStatus(scope: scope).firstUseReferenceCalls, 0)
  }
}
