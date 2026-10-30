import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3RegenerateArtifactTests:XCTestCase {
  func testReturnedBytesOwnIndependentStorage() throws {
    let source=MLXArray((0..<64).map(Float.init),[8,8])
    let owned=try H3RegenerateArtifact.copyData(source)
    let borrowed=source.asData(access:.noCopyIfContiguous).data
    withExtendedLifetime(source) {
      owned.withUnsafeBytes { copy in
        borrowed.withUnsafeBytes { original in
          XCTAssertNotEqual(copy.baseAddress,original.baseAddress)
          XCTAssertEqual(Array(copy),Array(original))
        }
      }
    }
  }
  func testTemporaryNoncontiguousBytesSurviveOwnerReleaseAndAllocatorReuse() throws {
    let data=try autoreleasepool {
      let temporary=MLXArray((0..<64).map(Float.init),[8,8]).transposed()
      return try H3RegenerateArtifact.copyData(temporary[1..<7,2..<6])
    }
    Stream.gpu.synchronize();Memory.clearCache()
    for _ in 0..<4 { let replacement=MLXArray.zeros([1024,1024]);eval(replacement) }
    let actual=data.withUnsafeBytes { bytes in
      (0..<24).map { bytes.loadUnaligned(fromByteOffset:$0*4,as:Float.self) }
    }
    let expected=(1..<7).flatMap { column in (2..<6).map { row in Float(row*8+column) } }
    XCTAssertEqual(actual,expected)
  }
  func testOversizedArtifactRejectsBeforeMaterialization() {
    XCTAssertThrowsError(try H3RegenerateArtifact.copyData(MLXArray.zeros([256*1024*1024+1],dtype:.uint8)))
  }
}
