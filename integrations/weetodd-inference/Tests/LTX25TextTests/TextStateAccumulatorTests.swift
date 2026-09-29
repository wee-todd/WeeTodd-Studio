import XCTest
@testable import LTX25Text

final class TextStateAccumulatorTests: XCTestCase {
  func testStreamingInterleaveIsBitIdenticalToCollectedStates() throws {
    let states: [[Float]] = (0..<5).map { layer in
      (0..<24).map { Float(($0 * 7 + layer) % 19 - 9) / 13 }
    }
    let accumulator = try TextStateAccumulator(tokens: 3, width: 8, layers: 5)
    for state in states { try accumulator.append(state) }
    let actual = try accumulator.take()
    XCTAssertEqual(actual, try TextMath.interleavedStates(states,tokens: 3,width: 8))
    XCTAssertThrowsError(try accumulator.take())
    XCTAssertThrowsError(try accumulator.append(states[0]))
  }
  func testBudgetShapeAndCompletionAreValidated() throws {
    XCTAssertThrowsError(try TextStateAccumulator(tokens: 1024,width: 3840,layers: 49,maximumBytes: 1024))
    XCTAssertThrowsError(try TextStateAccumulator(tokens: Int.max,width: 8,layers: 1))
    let accumulator = try TextStateAccumulator(tokens: 1,width: 2,layers: 2)
    XCTAssertThrowsError(try accumulator.take())
    XCTAssertThrowsError(try accumulator.append([.nan,1]))
    XCTAssertThrowsError(try accumulator.append([1]))
    try accumulator.append([1,2]); try accumulator.append([3,4])
    XCTAssertEqual(try accumulator.take(),try TextMath.interleavedStates([[1,2],[3,4]],tokens: 1,width: 2))
  }
}
