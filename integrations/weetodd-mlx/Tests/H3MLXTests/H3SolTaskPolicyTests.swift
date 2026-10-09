import XCTest
@_spi(H3SolDiagnostic) @testable import H3MLX

final class H3SolTaskPolicyTests: XCTestCase {
  func testDenseWarmupAndExcludedBlocksRemainDense() throws {
    let policy = try H3SolTaskPolicy()
    for evaluation in 0..<2 { for block in 0..<50 {
      XCTAssertFalse(policy.usesSol(block: block, completedEvaluations: evaluation))
    } }
    XCTAssertEqual((0..<50).filter { policy.usesSol(block: $0, completedEvaluations: 2) }.count, 45)
    for block in [0, 1, 33, 34, 35, -1, 50] { XCTAssertFalse(policy.usesSol(block: block, completedEvaluations: 3)) }
    XCTAssertEqual((0..<4).reduce(0) { total, evaluation in
      total + (0..<50).filter { policy.usesSol(block: $0, completedEvaluations: evaluation) }.count
    }, 90)
  }
  func testPackedGeneratedRangeRejectsReferencesHolesAndNonSuffixes() throws {
    XCTAssertEqual(try H3SolTaskPolicy.generatedVideoRange(indices: Array(1783..<39967), rows: 39967), 1783..<39967)
    for indices in [[], [0, 2, 3], [1, 2], [3, 3], [-1, 0, 1, 2, 3]] {
      XCTAssertThrowsError(try H3SolTaskPolicy.generatedVideoRange(indices: indices, rows: 4))
    }
  }
  func testPartialBoundaryGroupsProtectEveryRowOutsideGeneratedRange() throws {
    let geometry = try H3SolGeometry(rows: 39967, heads: 56, approximationRange: 1783..<39967)
    XCTAssertTrue(geometry.requiresExact(queryBlock: 27, keyBlock: 100))
    XCTAssertTrue(geometry.requiresExact(queryBlock: 100, keyBlock: 27))
    XCTAssertFalse(geometry.requiresExact(queryBlock: 28, keyBlock: 100))
    XCTAssertEqual(geometry.keyCount(624), 31)
  }
  func testUnsupportedTasksAndExecutionPoliciesRejectBeforeWeights() throws {
    for task in ["t2va", "initialized", "motion_fidelity"] {
      XCTAssertThrowsError(try H3SolTaskPolicy.validateTask(task: task, contextFrames: 0,
        isRefinement: false, ordinaryCanvas: true, mlxBackend: true, hasFast: false,
        hasVDN: false, hasFun: false, hasMotion: false))
    }
    for field in 0..<8 {
      XCTAssertThrowsError(try H3SolTaskPolicy.validateTask(task: "ref2va", contextFrames: field == 0 ? 22 : 0,
        isRefinement: field == 1, ordinaryCanvas: field != 2, mlxBackend: field != 3,
        hasFast: field == 4, hasVDN: field == 5, hasFun: field == 6, hasMotion: field == 7))
    }
    try H3SolTaskPolicy.validateState(referenceLayout: true, blockCount: 50, weightDecoded: true,
      hasNativeWorker: false, maximumRows: 40000, hasFast: false, hasCurveRank: false, hasVDN: false, hasFun: false)
    XCTAssertThrowsError(try H3SolTaskPolicy.validateState(referenceLayout: true, blockCount: 50,
      weightDecoded: true, hasNativeWorker: false, maximumRows: 40000,
      hasFast: false, hasCurveRank: true, hasVDN: false, hasFun: false))
    for tau in [Float.nan, Float.infinity, Float(-0.1), Float(16.1)] { XCTAssertThrowsError(try H3SolTaskPolicy(tau: tau)) }
  }
  func testTaskScopeRestoresAfterSuccessAndFailureAndDefaultMetadataUnchanged() throws {
    enum Stop: Error { case expected }
    XCTAssertNil(H3SolTaskContext.policy)
    try H3SolDiagnostic.withPolicy(tau: 0.75) { XCTAssertEqual(H3SolTaskContext.policy?.tau, 0.75) }
    XCTAssertNil(H3SolTaskContext.policy)
    XCTAssertThrowsError(try H3SolDiagnostic.withPolicy { throw Stop.expected })
    XCTAssertNil(H3SolTaskContext.policy)
    XCTAssertNil(H3BackendReport().metadata["solAttention"])
  }
}
