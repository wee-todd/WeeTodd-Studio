import XCTest
import MLX
import MLXRandom
import InferenceContracts
@testable import ReCaVSRMLX

final class ReCaGraphQueueTests: XCTestCase {
  func testCancellationAndWorkspaceLeaveNextCallUsable() throws {
    let plan = try ReCaPackedTilePlan(height:17,width:35,windowHeight:8,windowWidth:12,slidesAtEdge:true)
    let q = MLXArray.ones([17*35*2,4,128]).asType(.bfloat16)
    let k = MLXArray.ones([17*35*6,2,128]).asType(.bfloat16)
    let expected = try plan.apply(query:q,key:k,value:k).asData(access:.copy).data
    var checks = 0
    XCTAssertThrowsError(try plan.apply(query:q,key:k,value:k,compiled:true,asynchronous:true,
      checkCancellation:{ checks += 1; if checks == 3 { throw CancellationError() } })) { XCTAssertTrue($0 is CancellationError) }
    // The engine owns draining; the test does not synchronize before reuse.
    XCTAssertEqual(try plan.apply(query:q,key:k,value:k,compiled:true,asynchronous:true).asData(access:.copy).data,expected)
    XCTAssertThrowsError(try plan.apply(query:q,key:k,value:k,maximumWorkspaceBytes:1,compiled:true,asynchronous:true))
  }
  // Detect changed per-tile SDPA shapes, temporal planes, masks and inverse order.
  func testGraphAndQueueKeepCurrentPackedWords() throws {
    for d in [64,128] { for dtype in [DType.float32,.bfloat16] { for sliding in [true,false] {
      let h = 9, w = 17, qf = sliding ? 2 : 4, kf = sliding ? 6 : 8
      let plan = try ReCaPackedTilePlan(height:h,width:w,windowHeight:6,windowWidth:8,slidesAtEdge:sliding)
      MLXRandom.seed(761)
      let q = MLXRandom.normal([h*w*qf,4,d]).asType(dtype)
      let k = MLXRandom.normal([h*w*kf,2,d]).asType(dtype), v = MLXRandom.normal([h*w*kf,2,d]).asType(dtype)
      eval(q,k,v)
      let expected = try plan.apply(query:q,key:k,value:v).asData(access:.copy).data
      for (c,a) in [(true,false),(false,true),(true,true)] {
        XCTAssertEqual(try plan.apply(query:q,key:k,value:v,compiled:c,asynchronous:a).asData(access:.copy).data,expected,
          "D\(d) \(dtype) sliding\(sliding) compile\(c) async\(a)")
      }
    } } }
  }
  func testNativeGraphQueueBenchmarkWhenRequested() throws {
    guard let output = ProcessInfo.processInfo.environment["WEETODD_RECA_GRAPH_BENCHMARK"] else { throw XCTSkip("Opt-in native benchmark.") }
    let lease = try NativeInferenceLease.acquire(); defer { lease.release() }
    Memory.memoryLimit = 24 * 1024 * 1024 * 1024; Memory.cacheLimit = 3 * 1024 * 1024 * 1024
    MLXRandom.seed(42)
    var records: [[String: Any]] = []
    for (name,h,w,qh,kh,d,qf,kf) in [("wan2-prefix",48,84,24,24,128,6,6),("wan2-body",48,84,24,24,128,2,8),
      ("wan4-prefix",96,168,24,24,128,6,6),("wan4-body",96,168,24,24,128,2,8),
      ("decoder2",96,168,8,2,64,4,8),("decoder4",192,336,8,2,64,4,8)] {
      try autoreleasepool {
        let plan = try ReCaPackedTilePlan(height:h,width:w,windowHeight:d == 128 ? 22:24,windowWidth:d == 128 ? 40:24,slidesAtEdge:d == 128)
        let q = MLXRandom.normal([h*w*qf,qh,d]).asType(.bfloat16)
        let k = MLXRandom.normal([h*w*kf,kh,d]).asType(.bfloat16), v = MLXRandom.normal([h*w*kf,kh,d]).asType(.bfloat16)
        eval(q,k,v)
        let expected = try plan.apply(query:q,key:k,value:v).asData(access:.copy).data
        for mode in 0..<4 {
          XCTAssertEqual(try plan.apply(query:q,key:k,value:v,compiled:mode & 1 != 0,asynchronous:mode & 2 != 0).asData(access:.copy).data,expected)
        }
        for pair in 0..<4 { for mode in (pair%2 == 0 ? [0,1,2,3]:[3,2,1,0]) {
          Memory.peakMemory = 0
          let start = ProcessInfo.processInfo.systemUptime
          let result = try plan.apply(query:q,key:k,value:v,compiled:mode & 1 != 0,asynchronous:mode & 2 != 0)
          eval(result); Stream.gpu.synchronize()
          let seconds = ProcessInfo.processInfo.systemUptime-start, exact = result.asData(access:.copy).data == expected
          XCTAssertTrue(exact,name)
          records.append(["case":name,"pair":pair,"mode":mode,"seconds":seconds,"exactWords":exact,"peakMLXBytes":Memory.peakMemory])
        } }
      }
      Memory.clearCache()
    }
    try JSONSerialization.data(withJSONObject:records,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:output))
  }
}
