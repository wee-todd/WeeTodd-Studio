import XCTest
import MLX
import MLXRandom
import InferenceContracts
@testable import RelayVSR

final class RelayGraphQueueTests: XCTestCase {
  func testAdmissionReservesQueuedGathersBeforeLoadingWeights() throws {
    guard RelaySkippedAttention.runtimeQualified else { throw XCTSkip("Queue is qualified on g15d.") }
    let geometry = try RelayGeometry(width: 1344, height: 768)
    // The old single-frame admission was 15,209,269,248 bytes. A budget
    // with only 100 MB spare cannot also hold two eight-tile K/V batches.
    XCTAssertThrowsError(try RelayMemoryPlan.validate(geometry: geometry,
      options: .init(frameLimit: 1), budget: 15_309_269_248))
  }
  func testCancellationAndSmallWorkspaceLeaveNextCallUsable() throws {
    let plan = RelayAttentionPlan(height:17,width:35,
      windows:[RelayWindow(height:8,width:12,slidesAtEdge:true)],headDimension:128)
    let q = MLXArray.ones([17*35,4,128]).asType(.bfloat16)
    let k = MLXArray.ones([17*35,2,128]).asType(.bfloat16)
    let planes = [RelayKV(key:k,value:k)]
    let expected = try plan.evaluate(q,planes,useBlockSkipping:true).asData(access:.copy).data
    var checks = 0
    XCTAssertThrowsError(try plan.evaluate(q,planes,useBlockSkipping:true,compiled:true,asynchronous:true,
      checkCancellation: { checks += 1; if checks == 3 { throw CancellationError() } })) { XCTAssertTrue($0 is CancellationError) }
    // No test-side stream synchronization before the next real consumer.
    XCTAssertEqual(try plan(q,planes).asData(access:.copy).data,expected)
    XCTAssertEqual(try plan.evaluate(q,planes,useBlockSkipping:true,compiled:true,asynchronous:true,
      maximumWorkspaceBytes:1).asData(access:.copy).data,expected)
  }
  func testCompiledUnsafeOperandsAndWholeGridKeepCurrentWords() throws {
    for (h,w,size) in [(9,17,4),(2,3,40)] {
      let plan = RelayAttentionPlan(height:h,width:w,
        windows:[RelayWindow(height:size,width:size,slidesAtEdge:true)],headDimension:128)
      let q = MLXArray.ones([h*w,2,128]).asType(.bfloat16)
      let k = MLXArray.ones([h*w,1,128]).asType(.bfloat16)
      for value in [Float.zero,-Float.zero,Float.leastNormalMagnitude,Float.infinity] {
        let v = MLXArray.full(k.shape,values:MLXArray(value)).asType(.bfloat16)
        let planes = [RelayKV(key:k,value:v)]
        let expected = try plan.evaluate(q,planes,useBlockSkipping:true).asData(access:.copy).data
        XCTAssertEqual(try plan.evaluate(q,planes,useBlockSkipping:true,compiled:true,asynchronous:true).asData(access:.copy).data,expected)
      }
    }
  }
  // Detect changed gathers, GQA, ragged tiles or kernel arithmetic in either mode.
  func testGraphAndQueueKeepCurrentAttentionWords() throws {
    for d in [64, 128] { for dtype in [DType.float32, .bfloat16] {
      let h = 17, w = 35, n = h * w
      let plan = RelayAttentionPlan(height: h, width: w,
        windows: Array(repeating: RelayWindow(height: 8, width: 12, slidesAtEdge: d == 128), count: 3), headDimension: d)
      MLXRandom.seed(912)
      let q = MLXRandom.normal([n, 4, d]).asType(dtype)
      let k = MLXRandom.normal([n, 2, d]).asType(dtype)
      let v = MLXRandom.normal(k.shape).asType(dtype)
      eval(q,k,v)
      let planes = Array(repeating: RelayKV(key: k, value: v), count: 3)
      let expected = try plan.evaluate(q, planes, useBlockSkipping: true).asData(access: .copy).data
      for (c,a) in [(true,false),(false,true),(true,true)] {
        XCTAssertEqual(try plan.evaluate(q, planes, useBlockSkipping: true, compiled: c, asynchronous: a)
          .asData(access: .copy).data, expected, "D\(d) \(dtype) compile\(c) async\(a)")
      }
    } }
  }
  func testNativeGraphQueueBenchmarkWhenRequested() throws {
    guard let output = ProcessInfo.processInfo.environment["WEETODD_RELAY_GRAPH_BENCHMARK"] else { throw XCTSkip("Opt-in native benchmark.") }
    let lease = try NativeInferenceLease.acquire(); defer { lease.release() }
    Memory.memoryLimit = 16 * 1024 * 1024 * 1024; Memory.cacheLimit = 3 * 1024 * 1024 * 1024
    MLXRandom.seed(42)
    var records: [[String: Any]] = []
    for (name,h,w,qh,kh,d) in [("wan2",48,84,24,24,128),("wan4",96,168,24,24,128),
      ("decoder2",96,168,8,2,64),("decoder4",192,336,8,2,64)] {
      try autoreleasepool {
        let window = RelayWindow(height: 22, width: 40, slidesAtEdge: true)
        let ref = RelayWindow(height: 48, width: 48, slidesAtEdge: false)
        let video = RelayWindow(height: 24, width: 24, slidesAtEdge: false)
        let plan = RelayAttentionPlan(height: h, width: w,
          windows: d == 128 ? Array(repeating: window, count: 4) : [ref,ref,video,video], headDimension: d)
        let n = h*w, q = MLXRandom.normal([n,qh,d]).asType(.bfloat16)
        let k = MLXRandom.normal([n*4,kh,d]).asType(.bfloat16), v = MLXRandom.normal([n*4,kh,d]).asType(.bfloat16)
        eval(q,k,v)
        let planes = (0..<4).map { RelayKV(key:k[($0*n)..<(($0+1)*n)],value:v[($0*n)..<(($0+1)*n)]) }
        let expected = try plan.evaluate(q,planes,useBlockSkipping:true).asData(access:.copy).data
        for mode in 0..<4 {
          let c = mode & 1 != 0, a = mode & 2 != 0
          XCTAssertEqual(try plan.evaluate(q,planes,useBlockSkipping:true,compiled:c,asynchronous:a).asData(access:.copy).data,expected)
        }
        for pair in 0..<4 { for mode in (pair%2 == 0 ? [0,1,2,3] : [3,2,1,0]) {
          Memory.peakMemory = 0
          let start = ProcessInfo.processInfo.systemUptime
          let result = try plan.evaluate(q,planes,useBlockSkipping:true,compiled:mode & 1 != 0,asynchronous:mode & 2 != 0)
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
