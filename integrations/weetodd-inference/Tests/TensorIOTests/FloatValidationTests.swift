import XCTest
@testable import TensorIO

final class FloatValidationTests: XCTestCase {
  func testFiniteCheckPreservesEverySpecialValueAndTail() {
    for count in [0, 1, 15, 16, 17, 31, 32, 65] {
      var values = [Float](repeating: -Float.greatestFiniteMagnitude, count: count)
      XCTAssertTrue(FloatValidation.allFinite(values))
      for index in values.indices {
        for special in [Float.nan, .infinity, -.infinity, Float(bitPattern: 0x7f800001)] {
          values[index] = special
          XCTAssertFalse(FloatValidation.allFinite(values), "count=\(count) index=\(index)")
          values[index] = -.greatestFiniteMagnitude
        }
      }
    }
    XCTAssertTrue(FloatValidation.allFinite([0, -0, .leastNonzeroMagnitude, -.leastNormalMagnitude]))
  }
}
