import XCTest
import Darwin
@testable import TensorIO

final class Float16ConversionTests: XCTestCase {
  func testNearestRoundingMatchesSwiftAtTiesSubnormalsAndSigns() throws {
    var values: [Float] = [0,-0.0,65504,-65504,Float.leastNonzeroMagnitude,-Float.leastNonzeroMagnitude]
    for bits in stride(from: UInt16(0), to: UInt16(0x7bfe), by: 13) {
      let low = Float(Float16(bitPattern: bits)), high = Float(Float16(bitPattern: bits+1))
      let midpoint = (low+high)/2
      for x in [low,high,midpoint,midpoint.nextDown,midpoint.nextUp] { values += [x,-x] }
    }
    let actual = try Float16Conversion.nearest(values)
    XCTAssertEqual(actual.map(\.bitPattern), values.map { Float16($0).bitPattern })
    XCTAssertEqual(try Float16Conversion.nearest([]), [])
  }
  func testInvalidValuesAtVectorAndTailPositionsReject() throws {
    for length in [1,15,16,17,33] {
      for position in 0..<length {
        for value in [Float.nan,Float.infinity,-Float.infinity,Float(65504).nextUp,-Float(65504).nextUp] {
          var values = [Float](repeating: 0.125,count: length); values[position] = value
          XCTAssertThrowsError(try Float16Conversion.nearest(values))
        }
      }
    }
  }
  func testNonNearestRoundingModeRejectsWithoutChangingIt() throws {
    let original = fegetround()
    defer { fesetround(original) }
    XCTAssertEqual(fesetround(FE_DOWNWARD),0)
    XCTAssertThrowsError(try Float16Conversion.nearest([1.1]))
    XCTAssertEqual(fegetround(),FE_DOWNWARD)
  }
}
