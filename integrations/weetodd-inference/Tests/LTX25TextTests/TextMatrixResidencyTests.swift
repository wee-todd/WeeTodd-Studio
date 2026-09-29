import XCTest
@testable import LTX25Text

final class TextMatrixResidencyTests: XCTestCase {
  func testCancellationWithPreparedStorageReleasesOnUnwindAndRetries() throws {
    let gpu = try TextMatrixGPU()
    weak var storage: AnyObject?
    try autoreleasepool {
      let left = try gpu.prepareLeft([1,2,3,4],rows: 2,inner: 2)
      storage = left.buffer
      var checks = 0
      XCTAssertThrowsError(try gpu.multiply(left,[2,3],columns: 1,checkCancelled: {
        checks += 1; if checks == 2 { throw CancellationError() }
      }))
      XCTAssertEqual(checks,2)
      XCTAssertEqual(try gpu.multiply(left,[2,3],columns: 1),[8,18])
    }
    XCTAssertNil(storage)
  }
  func testPreparedLeftIsUploadedOnceAndMatchesOrdinaryProducts() throws {
    let gpu = try TextMatrixGPU()
    let a: [Float] = [1,2,3,4,5,6], b: [Float] = [7,8,9,10,11,12]
    let left = try gpu.prepareLeft(a,rows: 2,inner: 3)
    let before = gpu.leftUploadCount
    let first = try gpu.multiply(left,b,columns: 2)
    let second = try gpu.multiply(left,b,columns: 2,scale: 0.5)
    XCTAssertEqual(gpu.leftUploadCount,before)
    XCTAssertEqual(first,try gpu.multiply(a,b,rows: 2,inner: 3,columns: 2))
    XCTAssertEqual(second,first.map { $0*0.5 })
    XCTAssertThrowsError(try gpu.prepareLeft(a,rows: Int.max,inner: 3))
    XCTAssertThrowsError(try gpu.multiply(left,[1],columns: 2))
  }
  func testPreparedMatrixRejectsAnotherOwner() throws {
    let first = try TextMatrixGPU(), second = try TextMatrixGPU()
    let left = try first.prepareLeft([1,2],rows: 1,inner: 2)
    XCTAssertThrowsError(try second.multiply(left,[3,4],columns: 1))
  }
}
