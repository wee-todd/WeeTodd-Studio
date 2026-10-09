import XCTest
import MLX
import MLXRandom
@testable import H3MLX

final class H3VideoRotaryTests: XCTestCase {
  private func reference(_ x:MLXArray,_ cosine:MLXArray,_ sine:MLXArray) -> MLXArray {
    let rotated = concatenated([-x[.ellipsis,24..<48],x[.ellipsis,0..<24]],axis:-1)
    return concatenated([x[.ellipsis,0..<48]*cosine+rotated*sine,
      x[.ellipsis,48..<64]],axis:-1)
  }

  func testBothPrecisionsPreserveRoundedProductsAndTailBits() throws {
    for dtype:DType in [.float32,.float16] {
      let x = MLXRandom.normal([2,17,3,64],key:MLXRandom.key(971)).asType(dtype)
      let c = MLXRandom.normal([2,17,1,48],key:MLXRandom.key(972)).asType(dtype)
      let s = MLXRandom.normal([2,17,1,48],key:MLXRandom.key(973)).asType(dtype)
      let result = try H3VideoRotary.apply(x,cosine:c,sine:s)
      XCTAssertEqual(result.dtype,dtype)
      XCTAssertEqual(result.asType(.float32).asArray(Float.self).map(\.bitPattern),
        reference(x,c,s).asType(.float32).asArray(Float.self).map(\.bitPattern))
    }
  }

  func testRuntimeStridesAndSignedZeroRemainExact() throws {
    let x = MLXRandom.normal([2,18,6,128],key:MLXRandom.key(974))[0..<2,.stride(by:2),.stride(by:2),.stride(by:2)]
    let c = MLXRandom.normal([2,18,1,96],key:MLXRandom.key(975))[0..<2,.stride(by:2),0..<1,.stride(by:2)]
    let s = MLXRandom.normal([2,9,1,48],key:MLXRandom.key(976))
    let reversed = x[.ellipsis,.stride(by:-1)]
    for input in [x,reversed,MLXArray.zeros(x.shape)] {
      let result = try H3VideoRotary.apply(input,cosine:c,sine:s)
      XCTAssertEqual(result.asArray(Float.self).map(\.bitPattern),
        reference(input,c,s).asArray(Float.self).map(\.bitPattern))
    }
  }

  func testCPUFallbackAndInvalidGeometry() throws {
    try Device.withDefaultDevice(.cpu) {
      let x = MLXArray.ones([1,3,1,64]),c = MLXArray.ones([1,3,1,48])
      let result = try H3VideoRotary.apply(x,cosine:c,sine:c)
      XCTAssertEqual(result.asArray(Float.self),reference(x,c,c).asArray(Float.self))
      XCTAssertThrowsError(try H3VideoRotary.apply(x,cosine:c[.ellipsis,0..<24],sine:c))
    }
  }

  func testHalfSubnormalSignedZeroOverflowAndReversedAxes() throws {
    let bits:[UInt16] = [0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,
      0x3c00,0xbc00,0x7bff,0xfbff]
    let x = MLXArray((0..<(2*3*2*64)).map { bits[$0%bits.count] },[2,3,2,64])
      .view(dtype:.float16)
    let c = MLXArray.full([2,3,1,48],values:MLXArray(Float(2)),dtype:.float16)
    let s = MLXArray.zeros([2,3,1,48],dtype:.float16)
    for input in [x,x[.stride(by:-1),.stride(by:-1),.stride(by:-1),.stride(by:-1)]] {
      let result = try H3VideoRotary.apply(input,cosine:c,sine:s)
      XCTAssertEqual(result.view(dtype:.uint16).asArray(UInt16.self),
        reference(input,c,s).view(dtype:.uint16).asArray(UInt16.self))
    }
    let reversedC = c[.stride(by:-1),.stride(by:-1),0..<1,.stride(by:-1)]
    let repeatedS = broadcast(s[0..<1,0..<1,0..<1,0..<1],to:s.shape)
    let result = try H3VideoRotary.apply(x,cosine:reversedC,sine:repeatedS)
    XCTAssertEqual(result.view(dtype:.uint16).asArray(UInt16.self),
      reference(x,reversedC,repeatedS).view(dtype:.uint16).asArray(UInt16.self))
  }
}
