import XCTest
@testable import LTX25NNC

final class DenoiserMathTests: XCTestCase {
  func testRotaryWorkspaceBoundaryAndExplicitAllowancePreserveMath() throws {
    let bytes=31*42*24*32*64*8
    XCTAssertThrowsError(try DenoiserMath.rotaryElementCount(axes:3,tokens:31*42*24,heads:32,headWidth:128))
    XCTAssertEqual(try DenoiserMath.rotaryElementCount(axes:3,tokens:31*42*24,heads:32,headWidth:128,maximumBytes:bytes),bytes/8)
    XCTAssertThrowsError(try DenoiserMath.rotaryElementCount(axes:3,tokens:31*42*24,heads:32,headWidth:128,maximumBytes:bytes-1))
    for limit in [0,-1,Int.max] {
      XCTAssertThrowsError(try DenoiserMath.rotaryElementCount(axes:3,tokens:1,heads:32,headWidth:128,maximumBytes:limit))
    }
    let a=try DenoiserMath.rotary(positions:[0.3,80,92],axes:3,tokens:1,heads:2,headWidth:8,maximumPositions:[20,2048,2048])
    let b=try DenoiserMath.rotary(positions:[0.3,80,92],axes:3,tokens:1,heads:2,headWidth:8,maximumPositions:[20,2048,2048],maximumBytes:bytes)
    XCTAssertEqual(a.cos,b.cos);XCTAssertEqual(a.sin,b.sin)
  }
  func testBF16BoundaryUsesTiesToEvenAndTimestepIsCosThenSin() throws {
    XCTAssertEqual(DenoiserMath.bfloat16(Float(bitPattern: 0x3f808000)), 1)
    XCTAssertEqual(DenoiserMath.bfloat16(Float(bitPattern: 0x3f818000)).bitPattern, 0x3f820000)
    let zero = try DenoiserMath.timestep(0)
    XCTAssertEqual(Array(zero.prefix(128)), [Float](repeating: 1, count: 128))
    XCTAssertEqual(Array(zero.suffix(128)), [Float](repeating: 0, count: 128))
    XCTAssertThrowsError(try DenoiserMath.timestep(.nan))
    XCTAssertThrowsError(try DenoiserMath.timestep(-1))
  }

  func testSplitRotaryIncludesLeadingPaddingAndInterleavesAxes() throws {
    let r = try DenoiserMath.rotary(positions: [10, 1024, 1024], axes: 3, tokens: 1,
      heads: 2, headWidth: 8, maximumPositions: [20, 2048, 2048])
    XCTAssertEqual(r.cos, [Float](repeating: 1, count: 8))
    XCTAssertEqual(r.sin, [Float](repeating: 0, count: 8))
    let off = try DenoiserMath.rotary(positions: [0, 1024, 1024], axes: 3, tokens: 1,
      heads: 2, headWidth: 8, maximumPositions: [20, 2048, 2048])
    XCTAssertEqual(Array(off.cos.prefix(2)), [1, 1])
    XCTAssertEqual(off.sin[2], -1, accuracy: 0.000001)
    XCTAssertEqual(off.cos[3], 1)
    XCTAssertThrowsError(try DenoiserMath.rotary(positions: [0], axes: 3, tokens: 1,
      heads: 2, headWidth: 8, maximumPositions: [20, 2048, 2048]))
  }
}
