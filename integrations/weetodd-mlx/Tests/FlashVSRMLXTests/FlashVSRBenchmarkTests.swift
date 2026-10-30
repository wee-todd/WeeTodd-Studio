import Foundation
import XCTest
import MLX
@testable import FlashVSRMLX

final class FlashVSRBenchmarkTests: XCTestCase {
  func testOptInAppleWorkingShapes() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_FLASH_BENCHMARK"] == "1" else { throw XCTSkip("Set WEETODD_FLASH_BENCHMARK=1 for bounded attention working-shape measurements.") }
    for canvas in [256,512] {
      let spatial=(canvas/128)*(canvas/128),qBlocks=2*spatial,kBlocks=5*spatial
      let qCount=12*qBlocks*128*128,kCount=12*kBlocks*128*128
      let q=sin(MLXArray(0..<qCount).asType(.float32)*0.037).reshaped([1,12,qBlocks,128,128]).asType(.bfloat16)
      let k=cos(MLXArray(0..<kCount).asType(.float32)*0.041).reshaped([1,12,kBlocks,128,128]).asType(.bfloat16)
      let v=k*0.7
      eval(q,k,v); Memory.peakMemory=Memory.activeMemory
      let start=ProcessInfo.processInfo.systemUptime
      let route=try FlashVSRAttention.route(q:q,k:k,rows:canvas/128,columns:canvas/128,localRange:11,topK:FlashVSRConfig(variant:.base).topK(spatialBlocks:spatial))
      let routed=ProcessInfo.processInfo.systemUptime
      let output=try FlashVSRAttention.attend(queries:q,keys:k,values:v,selection:route)
      eval(output)
      let finish=ProcessInfo.processInfo.systemUptime
      XCTAssertEqual(output.shape,q.shape)
      XCTAssertTrue(output.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
      print("FLASH_ATTENTION_BENCH canvas=\(canvas) heads=12 qBlocks=\(qBlocks) kBlocks=\(kBlocks) routeSeconds=\(routed-start) attentionSeconds=\(finish-routed) peakMLXBytes=\(Memory.peakMemory)")
    }
  }
}
