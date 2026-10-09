import MLX
import MLXRandom
import XCTest
@testable import H3MLX

final class H3ModulationTests: XCTestCase {
  func testFusedModulationPreservesEveryBF16IntermediateAndRowMapping() {
    let width = 128, rows = 37
    let x = MLXRandom.normal([1,rows,width],key:MLXRandom.key(930)).asType(.bfloat16)
    let branch = MLXRandom.normal(x.shape,key:MLXRandom.key(931)).asType(.bfloat16)
    let table = MLXRandom.normal([12,6*width],key:MLXRandom.key(932)).asType(.bfloat16)
    let indices = MLXArray((0..<rows).map { Int32(($0*7)%12) })
    for pair in [(0,1),(3,4)] {
      let shift = take(table[.ellipsis,(pair.0*width)..<((pair.0+1)*width)],indices,axis:0)
      let scale = take(table[.ellipsis,(pair.1*width)..<((pair.1+1)*width)],indices,axis:0)
      let expected = x * (1+scale) + shift
      let actual = H3Modulation.scaleShift(x,table:table,indices:indices,
        shift:pair.0,scale:pair.1)
      XCTAssertEqual(actual.asType(.float32).asArray(Float.self),expected.asType(.float32).asArray(Float.self))
    }
    for slot in [2,5] {
      let expected = x + take(table[.ellipsis,(slot*width)..<((slot+1)*width)],indices,axis:0)*branch
      let actual = H3Modulation.residual(x,branch:branch,table:table,indices:indices,gate:slot)
      XCTAssertEqual(actual.asType(.float32).asArray(Float.self),expected.asType(.float32).asArray(Float.self))
    }
  }

  func testNoncontiguousInputsAndFP32FallbackPreserveArithmetic() {
    let width = 16, rows = 5
    let backing = MLXRandom.normal([1,rows,width*2],key:MLXRandom.key(933))
    for dtype in [DType.bfloat16,.float32] {
      let x = backing.asType(dtype)[.ellipsis,.stride(by:2)]
      let table = MLXArray((0..<(3*6*width)).map { Float($0%17)/32 },[3,6*width]).asType(dtype)
      let indices = MLXArray([Int32(2),0,1,2,0])
      let shift = take(table[.ellipsis,0..<width],indices,axis:0)
      let scale = take(table[.ellipsis,width..<(2*width)],indices,axis:0)
      let expected = x*(1+scale)+shift
      let actual = H3Modulation.scaleShift(x,table:table,indices:indices,shift:0,scale:1)
      XCTAssertEqual(actual.asArray(Float.self),expected.asArray(Float.self))
      let residual = H3Modulation.residual(x,branch:x,table:table,indices:indices,gate:2)
      let expectedResidual = x + take(table[.ellipsis,(2*width)..<(3*width)],indices,axis:0)*x
      XCTAssertEqual(residual.asArray(Float.self),expectedResidual.asArray(Float.self))
    }
  }
}
