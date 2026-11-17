import XCTest
import InferenceContracts
@testable import ReCaVSRMLX

final class ReCaMemoryTests: XCTestCase {
  func testAdmissionAccountsForRasterAndResidentWeightsBeforeAllocation() throws {
    let small = try ReCaGeometry(sourceWidth: 64, sourceHeight: 48, scale: 4)
    let huge = try ReCaGeometry(sourceWidth: 3840, sourceHeight: 2160, scale: 4)
    XCTAssertThrowsError(try ReCaMemoryPlan.admit(geometry: small, profile: .bf16, residency: .staged, budgetBytes: 1_000_000_000))
    XCTAssertNoThrow(try ReCaMemoryPlan.admit(geometry: small, profile: .bf16, residency: .staged, budgetBytes: 32 * 1024 * 1024 * 1024))
    XCTAssertThrowsError(try ReCaMemoryPlan.admit(geometry: huge, profile: .int8ConvRot, residency: .resident, budgetBytes: 32 * 1024 * 1024 * 1024))
  }
}
