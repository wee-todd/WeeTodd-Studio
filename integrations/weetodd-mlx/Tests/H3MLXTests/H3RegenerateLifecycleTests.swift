import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3RegenerateLifecycleTests: XCTestCase {
  static func sparseRelease(_ kind: String, alpha: UInt16 = 0x4180) throws -> URL {
    let f=H3RegenerateProfileTests.fixture(kind)
    var header:[String:Any] = ["__metadata__":f.metadata]
    var offset:UInt64=0
    var alphas:[UInt64]=[]
    for key in f.tensors.keys.sorted() {
      let spec=f.tensors[key]!
      let size=UInt64(spec.shape.reduce(1,*)) * (spec.dtype == "F32" ? 4:2)
      header[key] = ["dtype":spec.dtype,"shape":spec.shape,"data_offsets":[offset,offset+size]]
      if key.hasSuffix(".alpha") { alphas.append(offset) }
      offset += size
    }
    let headerData=try JSONSerialization.data(withJSONObject:header)
    var length=UInt64(headerData.count).littleEndian
    var data=withUnsafeBytes(of:&length) { Data($0) }; data.append(headerData)
    let url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".safetensors")
    try data.write(to:url)
    let handle=try FileHandle(forWritingTo:url)
    defer { try? handle.close() }
    try handle.truncate(atOffset:UInt64(data.count)+offset)
    for position in alphas {
      try handle.seek(toOffset:UInt64(data.count)+position)
      var bits=alpha.littleEndian
      try handle.write(contentsOf:withUnsafeBytes(of:&bits) { Data($0) })
    }
    return url
  }
  func testDedicatedAdmissionLeavesOrdinaryLoaderStrictAndReleasesOnFailure() throws {
    let url=try Self.sparseRelease("official")
    defer { try? FileManager.default.removeItem(at:url) }
    XCTAssertThrowsError(try H3LoRAFile(url:url,strength:1))
    let adapter=try H3RegenerateAdapter(url:url,profile:.publisherThreeStep,companions:[.sharpCurve])
    XCTAssertNil(adapter.activeStage)
    enum Failure:Error { case deliberate }
    XCTAssertThrowsError(try adapter.withStage(.regenerate) {
      XCTAssertEqual(adapter.activeStage,.regenerate)
      throw Failure.deliberate
    })
    XCTAssertNil(adapter.activeStage)
    let source=MLXArray.ones([1,1]), base=MLXArray([Float(9)],[1,1])
    let unchanged=try adapter.withStage(.draft) {
      try adapter.apply(base:base,input:source,target:"diffusion_model.blocks.0.mlp.fc2",reorderQKV:false)
    }
    XCTAssertEqual(unchanged.item(Float.self),9)
    adapter.close()
    XCTAssertThrowsError(try adapter.withStage(.draft) {})
  }
  func testBlockWeightsClearAfterSuccessFailureAndClose() throws {
    let url=try Self.sparseRelease("official")
    defer { try? FileManager.default.removeItem(at:url) }
    let adapter=try H3RegenerateAdapter(url:url,profile:.publisherThreeStep,companions:[.sharpCurve])
    func seed() throws {
      _ = try adapter.mergedWeights.weight(block:0,target:"fixture",shape:[2,2],dtype:.bfloat16) {
        MLXArray.ones([2,2],dtype:.bfloat16)
      }
    }
    try adapter.withStage(.regenerate) { try seed();XCTAssertEqual(adapter.mergedWeights.count,1) }
    XCTAssertEqual(adapter.mergedWeights.count,0)
    enum Failure:Error { case injected }
    XCTAssertThrowsError(try adapter.withStage(.regenerate) { try seed();throw Failure.injected })
    XCTAssertEqual(adapter.mergedWeights.count,0)
    try seed();adapter.close();XCTAssertEqual(adapter.mergedWeights.count,0)
  }
  func testReducedFileRequiresExactBF16AlphaAndExpectedProfile() throws {
    let url=try Self.sparseRelease("r16")
    defer { try? FileManager.default.removeItem(at:url) }
    XCTAssertNoThrow(try H3RegenerateAdapter(url:url,profile:.extractedRefiner16,companions:[.turbo,.sharpLinear,.sharpCurve]))
    XCTAssertThrowsError(try H3RegenerateAdapter(url:url,profile:.extractedRefiner384,companions:[.turbo,.sharpLinear,.sharpCurve]))
    let wrong=try Self.sparseRelease("r16",alpha:0x4100)
    defer { try? FileManager.default.removeItem(at:wrong) }
    XCTAssertThrowsError(try H3RegenerateAdapter(url:wrong,profile:.extractedRefiner16,companions:[.turbo,.sharpLinear,.sharpCurve]))
  }
  func testCancellationBeforeStageNeverActivatesAdapter() async throws {
    let url=try Self.sparseRelease("official")
    defer { try? FileManager.default.removeItem(at:url) }
    let task=Task {
      let adapter=try H3RegenerateAdapter(url:url,profile:.publisherThreeStep,companions:[.sharpCurve])
      withUnsafeCurrentTask { $0?.cancel() }
      do { try adapter.withStage(.regenerate) {}; return false }
      catch is CancellationError { return adapter.activeStage == nil }
    }
    let cancelled = try await task.value
    XCTAssertTrue(cancelled)
  }
}
