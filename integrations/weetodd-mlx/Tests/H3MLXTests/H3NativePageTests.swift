import Foundation
import MLX
import TensorIO
import XCTest
@testable import H3MLX

final class H3NativePageTests: XCTestCase {
  private func fixture() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
    try save(arrays: ["linear.weight": MLXArray([UInt32](repeating: 0x01010101, count: 32), [2,16]),
      "linear.scales": MLXArray([Float(0.5),2], [2,1]).asType(.bfloat16),
      "linear.biases": MLXArray.zeros([2,1], dtype: .bfloat16),
      "unused": MLXArray.ones([4], dtype: .bfloat16)], url: url)
    return url
  }
  func testNativeFactorsRetainPackedProjectionValues() throws {
    let url = try fixture(); defer { try? FileManager.default.removeItem(at: url) }
    let file = try SafeTensorFile(url: url)
    let source = H3NativePage(file: file, url: url)
    let projection = try H3QwenQ8Projection(packed: source.read("linear.weight"),
      scales: source.read("linear.scales"), biases: source.read("linear.biases"), columns: 64)
    let output = try projection.project(MLXArray.ones([1,64], dtype: .bfloat16))
    XCTAssertEqual(output.asArray(Float.self), [32,128])
    XCTAssertThrowsError(try source.read("linear.weight"))
    source.clear()
    XCTAssertThrowsError(try source.read("unused"))
  }
  func testNativePageRejectsChangedSourceBeforeReturningAnotherFactor() throws {
    let url = try fixture(); defer { try? FileManager.default.removeItem(at: url) }
    let file = try SafeTensorFile(url: url)
    let source = H3NativePage(file: file, url: url)
    _ = try source.read("linear.weight")
    let handle = try FileHandle(forWritingTo: url)
    try handle.truncate(atOffset: 0); try handle.close()
    XCTAssertThrowsError(try source.read("linear.scales"))
  }
  func testExplicitClearClosesHealthyOwnerAndDiscardsUnreadFactors() throws {
    let url = try fixture(); defer { try? FileManager.default.removeItem(at: url) }
    let source = H3NativePage(file: try SafeTensorFile(url: url), url: url)
    XCTAssertEqual(try source.read("linear.scales").asArray(Float.self), [0.5, 2])
    source.clear()
    source.clear()
    XCTAssertThrowsError(try source.read("linear.weight"))
    XCTAssertThrowsError(try source.read("unused"))
  }
  func testCancelledNativePageDoesNotReturnPendingWeights() async throws {
    let url = try fixture(); defer { try? FileManager.default.removeItem(at: url) }
    let task = Task {
      let source = H3NativePage(file: try SafeTensorFile(url: url), url: url)
      _ = try source.read("linear.weight")
      withUnsafeCurrentTask { $0?.cancel() }
      do { _ = try source.read("unused"); XCTFail("Cancelled page returned a factor") }
      catch is CancellationError { }
      XCTAssertThrowsError(try source.read("linear.scales"))
    }
    try await task.value
  }
}
