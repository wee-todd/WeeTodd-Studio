import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXMSRLayoutTests:XCTestCase {
  func testFiveReferenceBoundaryUsesConsecutiveNegativeSlotsWithoutDenseMask() throws {
    let target=try AVGeometry(width:64,height:64,frames:17,fps:24)
    let guide=try AVGeometry(width:64,height:64,frames:25,fps:24)
    let layout=try MLXMSRLayout(target:target,groups:(0..<5).map { _ in
      .init(geometry:guide,strength:1,attentionStrength:1)
    })
    XCTAssertEqual(layout.groupRows.count,6)
    XCTAssertEqual(layout.videoTokens,target.videoTokens+5*guide.videoTokens)
    for index in 0..<5 {
      let row=target.videoTokens+index*guide.videoTokens
      XCTAssertEqual(layout.positions[row*3],guide.videoPositions[0]-Float(5-index)/24)
    }
    XCTAssertThrowsError(try MLXMSRLayout(target:target,groups:[]))
    XCTAssertThrowsError(try MLXMSRLayout(target:target,groups:(0..<6).map { _ in
      .init(geometry:guide,strength:1,attentionStrength:1)
    }))
  }

  func testTwoReferencesKeepOrderNegativeTimesAndCompactAttention() throws {
    let target=try AVGeometry(width:64,height:64,frames:17,fps:24)
    let first=try AVGeometry(width:64,height:64,frames:25,fps:24)
    let second=try AVGeometry(width:64,height:64,frames:33,fps:24)
    let layout=try MLXMSRLayout(target:target,groups:[
      .init(geometry:first,strength:1,attentionStrength:0.4),
      .init(geometry:second,strength:0.7,attentionStrength:0.8)])
    XCTAssertEqual(layout.groupRows,[target.videoTokens,first.videoTokens,second.videoTokens])
    XCTAssertEqual(layout.positions[target.videoTokens*3],first.videoPositions[0]-2/24)
    XCTAssertEqual(layout.positions[(target.videoTokens+first.videoTokens)*3],second.videoPositions[0]-1/24)
    let prepared=try layout.prepare(generated:.zeros([target.videoTokens,128]),
      references:[.ones([first.videoTokens,128]),.ones([second.videoTokens,128])*2])
    XCTAssertEqual(prepared.condition.mask.count,layout.videoTokens)
    XCTAssertEqual(prepared.condition.mask[target.videoTokens],0)
    XCTAssertEqual(prepared.condition.mask.last!,0.3,accuracy:0.00001)
    let templates=prepared.attentionTemplates.asArray(Float.self)
    XCTAssertEqual(templates[target.videoTokens],0.4,accuracy:0.00001)
    XCTAssertEqual(templates[target.videoTokens+first.videoTokens],0.8,accuracy:0.00001)
    XCTAssertEqual(templates[layout.videoTokens],0.4,accuracy:0.00001)
    XCTAssertThrowsError(try layout.prepare(generated:.zeros([target.videoTokens,128]),references:[]))
  }
}
