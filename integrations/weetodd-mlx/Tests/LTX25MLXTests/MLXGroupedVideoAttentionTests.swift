import XCTest
import MLX
@testable import LTX25MLX

final class MLXGroupedVideoAttentionTests:XCTestCase {
  func testCompactReferenceTemplatesMatchDenseAdditiveMask() throws {
    let heads=2,rows=7,width=8
    let q=MLXArray((0..<(heads*rows*width)).map { Float($0%17-8)/19 },[1,heads,rows,width])
    let k=MLXArray((0..<(heads*rows*width)).map { Float($0%23-11)/21 },[1,heads,rows,width])
    let v=MLXArray((0..<(heads*rows*width)).map { Float($0%13-6)/11 },[1,heads,rows,width])
    let templates=MLXArray([Float](repeating:1,count:3)
      + [Float](repeating:0.35,count:4)
      + [Float](repeating:0.35,count:3)
      + [Float](repeating:1,count:4),[2,rows])
    let compact=try MLXGroupedVideoAttention.evaluate(q:q,k:k,v:v,
      groups:[3,4],templates:templates,scale:1/Float(width).squareRoot())
    let dense=concatenated([
      broadcast(templates[0].reshaped([1,rows]),to:[3,rows]),
      broadcast(templates[1].reshaped([1,rows]),to:[4,rows])],axis:0)
      .reshaped([1,1,rows,rows])
    let expected=MLXFast.scaledDotProductAttention(queries:q,keys:k,values:v,
      scale:1/Float(width).squareRoot(),mask:dense)
    XCTAssertLessThan((compact-expected).abs().max().item(Float.self),0.00003)
    XCTAssertThrowsError(try MLXGroupedVideoAttention.evaluate(q:q,k:k,v:v,
      groups:[3,3],templates:templates,scale:1))
  }
}
