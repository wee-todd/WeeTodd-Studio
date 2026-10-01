import XCTest
import MLX
@testable import LTX25MLX

final class MLXMSRSlotEmbeddingTests:XCTestCase {
  func testInstalledMSRAdapterWhenProvided() throws {
    guard let path=ProcessInfo.processInfo.environment["WEETODD_MSR_TEST_ADAPTER"] else {
      throw XCTSkip("Set WEETODD_MSR_TEST_ADAPTER for installed checkpoint validation.")
    }
    let state=try MLXMSRSlotEmbedding.load(URL(fileURLWithPath:path))
    let first=try MLXMSRSlotEmbedding.embedding(slotID:1,state:state)
    let last=try MLXMSRSlotEmbedding.embedding(slotID:5,state:state)
    XCTAssertGreaterThan((first-last).abs().max().item(Float.self),0.00001)
  }
  func testFourierSlotMLPUsesOneBasedIdentityAndSiLU() throws {
    var first=[Float](repeating:0,count:256*33)
    first[0]=16
    var last=[Float](repeating:0,count:128*256)
    last[0]=1
    let state:[String:MLXArray]=[
      "frequencies":.zeros([16]),"net.0.weight":MLXArray(first,[256,33]),
      "net.0.bias":.zeros([256]),"net.2.weight":MLXArray(last,[128,256]),
      "net.2.bias":.zeros([128])]
    for id in [1,5] {
      let value=try MLXMSRSlotEmbedding.embedding(slotID:id,state:state).asArray(Float.self)
      XCTAssertEqual(value.count,128)
      let number=Float(id)
      XCTAssertEqual(value[0],number/(1+exp(-number)),accuracy:0.00001)
      XCTAssertTrue(value.dropFirst().allSatisfy({ abs($0)<0.00001 }))
    }
    XCTAssertThrowsError(try MLXMSRSlotEmbedding.embedding(slotID:0,state:state))
    XCTAssertThrowsError(try MLXMSRSlotEmbedding.embedding(slotID:6,state:state))
  }
}
