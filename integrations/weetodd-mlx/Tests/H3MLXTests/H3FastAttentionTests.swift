import XCTest
import MLX
import MLXRandom
import CryptoKit
@testable import H3MLX

final class H3FastAttentionTests: XCTestCase {
  func testProductionBF16D128ConsumerAcceptsSortedRoutesAndRetainsSignedCompression() throws {
    let tiles = try H3FastTiles(prefixSegments: [3,65], videoGrid: [3,5,6])
    let shape = [1,3,tiles.rows,128]
    let q = MLXArray.zeros(shape,dtype:.bfloat16)
    let v = broadcast(MLXArray([Float(2),4,-2],[1,3,1,1]).asType(.bfloat16),to:shape)
    let gate = MLXArray.full(shape,values:MLXArray(Float(-0.75)),dtype:.bfloat16)
    let result = try H3FastAttention.evaluate(query:q,key:q,value:v,gate:gate,tiles:tiles,
      sparsity:0,minimumSparseRows:1)
    let expected = broadcast(MLXArray([Float(0.5),1,-0.5],[1,3,1,1]),to:shape)
    XCTAssertEqual(result.shape,shape)
    XCTAssertEqual(result.dtype,.bfloat16)
    XCTAssertEqual(result.asType(.float32).asArray(Float.self),expected.asArray(Float.self))
  }

  func testIndexedAttentionCancellationRejectsWorkBeforeMetalExecution() async throws {
    let cancelled = Task { () throws -> Void in
      withUnsafeCurrentTask { $0?.cancel() }
      let tiles = try H3FastTiles(prefixSegments:[1],videoGrid:[1,1,1])
      _ = try H3IndexedAttention.evaluate(
        query:MLXArray.zeros([1,1,64,128],dtype:.bfloat16),
        key:MLXArray.zeros([1,1,128,128],dtype:.bfloat16),
        value:MLXArray.zeros([1,1,128,128],dtype:.bfloat16),
        routes:MLXArray([Int32(0),1],[1,1,1,2]),tiles:tiles)
    }
    do { _ = try await cancelled.value; XCTFail("Cancelled attention executed.") }
    catch is CancellationError { }
  }

  func testIndexedAttentionNormalizesRouteViewsAndRejectsSteppedFeatures() throws {
    let tiles = try H3FastTiles(prefixSegments:[3],videoGrid:[3,5,6])
    let count = tiles.sizes.count, video = count - tiles.prefixTiles, heads = 3
    let q = MLXRandom.normal([1,heads,video*64,128],key:MLXRandom.key(920)).asType(.bfloat16)
    let k = MLXRandom.normal([1,heads,count*64,128],key:MLXRandom.key(921)).asType(.bfloat16)
    let v = MLXRandom.normal([1,heads,count*64,128],key:MLXRandom.key(922)).asType(.bfloat16)
    let rows = (0..<(heads*video)).flatMap { group in [Int32(group%count),Int32((group+1)%count)] }
    let dense = MLXArray(rows,[1,heads,video,2])
    let sliced = MLXArray(rows.flatMap { [$0,Int32(count-1)] },[1,heads,video,4])[.ellipsis,.stride(by:2)]
    let expected = try H3IndexedAttention.evaluate(query:q,key:k,value:v,routes:dense,tiles:tiles)
    let actual = try H3IndexedAttention.evaluate(query:q,key:k,value:v,routes:sliced,tiles:tiles)
    XCTAssertEqual(abs(actual.asType(.float32)-expected.asType(.float32)).max().item(Float.self),0)
    let broadcastRoutes = broadcast(MLXArray([Int32(0),Int32(count-1)],[1,1,1,2]),
      to:[1,heads,video,2])
    let repeated = MLXArray((0..<(heads*video)).flatMap { _ in [Int32(0),Int32(count-1)] },
      [1,heads,video,2])
    let broadcastResult = try H3IndexedAttention.evaluate(query:q,key:k,value:v,
      routes:broadcastRoutes,tiles:tiles)
    let repeatedResult = try H3IndexedAttention.evaluate(query:q,key:k,value:v,routes:repeated,tiles:tiles)
    XCTAssertEqual(abs(broadcastResult.asType(.float32)-repeatedResult.asType(.float32)).max().item(Float.self),0)
    let stepped = MLXRandom.normal([1,heads,video*64,256],key:MLXRandom.key(923))
      .asType(.bfloat16)[.ellipsis,.stride(by:2)]
    XCTAssertThrowsError(try H3IndexedAttention.evaluate(query:stepped,key:k,value:v,routes:dense,tiles:tiles))
    let steppedKeys = MLXRandom.normal([1,heads,count*64,256],key:MLXRandom.key(924))
      .asType(.bfloat16)[.ellipsis,.stride(by:2)]
    XCTAssertThrowsError(try H3IndexedAttention.evaluate(query:q,key:steppedKeys,value:v,routes:dense,tiles:tiles))
    XCTAssertThrowsError(try H3IndexedAttention.evaluate(query:q,key:k,value:steppedKeys,routes:dense,tiles:tiles))
  }

  func testIndexedBF16AttentionMasksEveryPartialTileAndPreservesPerHeadRepeatedRoutes() throws {
    let tiles = try H3FastTiles(prefixSegments: [3, 65], videoGrid: [3, 5, 6])
    let count = tiles.sizes.count, video = count - tiles.prefixTiles, heads = 3
    let q = MLXRandom.normal([1, heads, video * 64, 128], key: MLXRandom.key(910)).asType(.bfloat16)
    let k = MLXRandom.normal([1, heads, count * 64, 128], key: MLXRandom.key(911)).asType(.bfloat16)
    let v = MLXRandom.normal([1, heads, count * 64, 128], key: MLXRandom.key(912)).asType(.bfloat16)
    let selections = (0..<(heads * video)).flatMap { group in
      [Int32(group % count), Int32((group + 2) % count), Int32(group % count)]
    }
    let routes = MLXArray(selections, [1, heads, video, 3])
    let result = try H3IndexedAttention.evaluate(query: q, key: k, value: v, routes: routes, tiles: tiles)
    XCTAssertEqual(result.shape, q.shape)
    XCTAssertEqual(result.dtype, .bfloat16)
    for head in 0..<heads {
      for tile in 0..<video {
        let size = tiles.sizes[tiles.prefixTiles + tile]
        let selected = selections[((head * video + tile) * 3)..<((head * video + tile + 1) * 3)]
        // Independent native SDPA oracle: concatenate actual rows only.
        // Random nonzero padding makes an omitted mask fail this comparison.
        func actualRows(_ input: MLXArray) -> MLXArray {
          concatenated(selected.map { index in
            let start = Int(index) * 64
            return input[0..<1,head..<(head+1),start..<(start+tiles.sizes[Int(index)]),0..<128]
          },axis:2)
        }
        let query = q[0..<1,head..<(head+1),(tile*64)..<(tile*64+size),0..<128]
        let expected = MLXFast.scaledDotProductAttention(queries: query, keys: actualRows(k),
          values: actualRows(v), scale: 1 / Float(128).squareRoot(), mask: nil)
        let actual = result[0..<1,head..<(head+1),(tile*64)..<(tile*64+size),0..<128]
        XCTAssertLessThanOrEqual(abs(actual.asType(.float32)-expected.asType(.float32)).max().item(Float.self),0.008)
      }
    }
    XCTAssertThrowsError(try H3IndexedAttention.evaluate(query:q,key:k,value:v,
      routes:MLXArray.full(routes.shape,values:MLXArray(Int32(count))),tiles:tiles))
    XCTAssertThrowsError(try H3IndexedAttention.evaluate(query:q,key:k,value:v,
      routes:MLXArray.full(routes.shape,values:MLXArray(Int32(-1))),tiles:tiles))
    XCTAssertThrowsError(try H3IndexedAttention.evaluate(query:q.asType(.float32),key:k,
      value:v,routes:routes,tiles:tiles))
    XCTAssertThrowsError(try H3IndexedAttention.evaluate(query:q,key:k,value:v,
      routes:routes[0..<1,0..<1,0..<video,0..<3],tiles:tiles))
  }

  func testSparseHeadBatchRemainderMatchesIndependentHeads() throws {
    let tiles = try H3FastTiles(prefixSegments: [3, 5], videoGrid: [3, 5, 6])
    let shape = [1, 5, tiles.rows, 4]
    let data = (0..<(5 * tiles.rows * 4)).map { Float($0 % 37 - 18) / 37 }
    for dtype: DType in [.float32, .bfloat16] {
      let q = MLXArray(data, shape).asType(dtype)
      let k = (q * 0.7).asType(dtype), v = (q * 0.4).asType(dtype)
      let gate = (q * 8).asType(dtype)
      let result = try H3FastAttention.evaluate(query: q, key: k, value: v,
        gate: gate, tiles: tiles, minimumSparseRows: 1)
      var expected: [MLXArray] = []
      for head in 0..<5 {
        func slice(_ value: MLXArray) -> MLXArray {
          value[0..<1,head..<(head+1),0..<tiles.rows,0..<4]
        }
        expected.append(try H3FastAttention.evaluate(query: slice(q), key: slice(k),
          value: slice(v), gate: slice(gate), tiles: tiles, minimumSparseRows: 1))
      }
      XCTAssertEqual(result.dtype, dtype)
      let error = abs(result.asType(.float32) - concatenated(expected,axis:1).asType(.float32)).max().item(Float.self)
      XCTAssertLessThanOrEqual(error, dtype == .float32 ? 3e-6 : 0.012)
    }
  }
  func testTileGatherPreservesHeadGroupOrderRepeatedTilesAndEveryPaddedRow() throws {
    try Device.withDefaultDevice(.cpu) {
      let heads = 3, count = 5, width = 4
      let values = (0..<(heads * count * 64 * width)).map(Float.init)
      let routes: [Int32] = [0,3,4, 1,0,1, 3,2,0, 4,1,4, 2,4,0, 1,3,2]
      var expected: [Float] = []
      for head in 0..<heads {
        for group in 0..<2 {
          for slot in 0..<3 {
            let tile = Int(routes[(head * 2 + group) * 3 + slot])
            let start = (head * count + tile) * 64 * width
            expected += values[start..<(start + 64 * width)]
          }
        }
      }
      for dtype: DType in [.float32, .bfloat16] {
        let blocks = MLXArray(values, [heads, count, 64, width]).asType(dtype)
        let result = try H3FastAttention.gatherTiles(blocks: blocks,
          routes: MLXArray(routes, [1, heads, 2, 3]))
        XCTAssertEqual(result.shape, [heads * 2, 1, 3 * 64, width])
        XCTAssertEqual(result.dtype, dtype)
        XCTAssertEqual(result.asArray(Float.self), MLXArray(expected).asType(dtype).asArray(Float.self))
        XCTAssertThrowsError(try H3FastAttention.gatherTiles(blocks: blocks,
          routes: MLXArray([Int32(0)], [1, 1, 1, 1])))
      }
    }
  }
  /// Attention-only probe at the measured 672x384/124-frame workload. It
  /// loads no model weights and freezes every output bit before optimization.
  func testRepresentativeAttentionFrozenParityAndTiming() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let outputPath = environment["WEETODD_H3_VSA_PROBE_OUTPUT"] else {
      throw XCTSkip("Opt-in bounded VSA attention parity/performance probe.")
    }
    guard !FileManager.default.fileExists(atPath: outputPath) else {
      throw H3CheckpointError.invalid("VSA probe output already exists.")
    }
    let memoryBudget = environment["WEETODD_H3_VSA_PROBE_MAX_MLX_BYTES"].flatMap(Int.init)
    guard memoryBudget == nil || memoryBudget == 5 * 1024 * 1024 * 1024 / 2 else {
      throw H3CheckpointError.invalid("VSA probe requires its fixed 2.5 GiB allocation bound.")
    }
    let tiles = try H3FastTiles(prefixSegments: [171, 414], videoGrid: [37, 12, 21])
    let shape = [1, 56, tiles.rows, 128]
    let inputs = (0..<4).map {
      MLXRandom.normal(shape, key: MLXRandom.key(UInt64(801 + $0))).asType(.bfloat16)
    }
    eval(inputs)
    let previousLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = previousLimit }
    Memory.peakMemory = Memory.activeMemory
    var seconds: [Double] = []
    var output: MLXArray?
    for _ in 0..<4 {
      let start = CFAbsoluteTimeGetCurrent()
      output = try H3FastAttention.evaluate(query: inputs[0], key: inputs[1],
        value: inputs[2], gate: inputs[3], tiles: tiles)
      Stream.gpu.synchronize()
      seconds.append(CFAbsoluteTimeGetCurrent() - start)
    }
    if let memoryBudget { XCTAssertLessThanOrEqual(Memory.peakMemory, memoryBudget) }
    let result = try XCTUnwrap(output)
    XCTAssertEqual(result.shape, shape)
    XCTAssertEqual(result.dtype, .bfloat16)
    var digest = SHA256()
    for head in 0..<shape[1] {
      let values = result[0, head, 0..<tiles.rows, 0..<128].asArray(Float.self)
      values.withUnsafeBytes { digest.update(data: Data($0)) }
    }
    let hash = digest.finalize().map { String(format: "%02x", $0) }.joined()
    if let expected = environment["WEETODD_H3_VSA_PROBE_EXPECTED_SHA256"] {
      XCTAssertEqual(hash, expected, "Every output must retain its frozen Float32 value.")
    }
    let report: [String: Any] = ["shape": shape, "inputSeeds": [801, 802, 803, 804],
      "dtype": "bfloat16", "prefixSegments": [171, 414], "videoGrid": [37, 12, 21],
      "outputFloat32SHA256": hash, "secondsIncludingFirstCompilation": seconds,
      "warmSeconds": Array(seconds.dropFirst()), "peakMLXBytes": Memory.peakMemory,
      "scope": "attention only; no weights, sampler, VAE, hash time or whole-render speed claim"]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: outputPath))
  }
  func testTrainedCompressionGateIsUnboundedSignedLinearProjection() {
    Device.withDefaultDevice(.cpu) {
      let input=MLXArray([Float(1),2,3,4],[1,2,2])
      let weight=MLXArray([Float(1),0,0,-1,2,0,0,2],[4,2])
      let gate=H3TransformerBlock.trainedCompressionGate(input,weight:weight,heads:2,headWidth:2)
      XCTAssertEqual(gate.asArray(Float.self),[1,-2,3,-4,2,4,6,8])
      let zero=H3TransformerBlock.trainedCompressionGate(input,weight:MLXArray.zeros([4,2]),heads:2,headWidth:2)
      XCTAssertEqual(zero.asArray(Float.self),[Float](repeating:0,count:8))
    }
  }
  func testBF16SparseMaskRetainsOutputTypeAndMatchesFloatReference() throws {
    let url = try XCTUnwrap(Bundle.module.url(forResource:"fasth3-vsa-small",withExtension:"json",subdirectory:"Fixtures"))
    let data = try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as! [String:Any]
    let shape = data["shape"] as! [Int]
    func array(_ key:String) -> MLXArray { MLXArray((data[key] as! [Double]).map(Float.init),shape).asType(.bfloat16) }
    let tiles = try H3FastTiles(prefixSegments:data["prefixSegments"] as! [Int],videoGrid:data["videoGrid"] as! [Int])
    let result = try H3FastAttention.evaluate(query:array("query"),key:array("key"),value:array("value"),
      gate:array("gate"),tiles:tiles,minimumSparseRows:64)
    XCTAssertEqual(result.dtype,.bfloat16)
    XCTAssertLessThan(abs(result.asType(.float32)-array("expected").asType(.float32)).max().item(Float.self),0.012)
  }
  func testTrainedVSARequiresMatchingGeometryInsteadOfSilentlyUsingDenseAttention() throws {
    let tiles = try H3FastTiles(prefixSegments:[2,3],videoGrid:[1,1,1])
    XCTAssertThrowsError(try H3TransformerBlock.validateTrainedAttentionGeometry(variant:.vsaV1,tiles:nil,rows:6))
    XCTAssertThrowsError(try H3TransformerBlock.validateTrainedAttentionGeometry(variant:.vsaV1,tiles:tiles,rows:7))
    XCTAssertThrowsError(try H3TransformerBlock.validateTrainedAttentionGeometry(variant:.denseV1,tiles:tiles,rows:6))
    XCTAssertNoThrow(try H3TransformerBlock.validateTrainedAttentionGeometry(variant:.vsaV1,tiles:tiles,rows:6))
    XCTAssertNoThrow(try H3TransformerBlock.validateTrainedAttentionGeometry(variant:.denseV1,tiles:nil,rows:6))
  }

  func testInstalledPublicVSAComponentRejectsMissingGeometryBeforeWeightedWork() throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_FASTH3_ROOT"] else {
      throw XCTSkip("Explicit installed VSA component API admission without weighted execution.")
    }
    let checkpoint = URL(fileURLWithPath:root).appendingPathComponent("weetodd-fasth3-vsa-datafree-q8-paged")
    XCTAssertThrowsError(try H3TransformerBlock.evaluate(checkpointURL:checkpoint,index:0,
      input:MLXArray.zeros([1,6,5376],dtype:.bfloat16),modulation:MLXArray.zeros([1,96768],dtype:.bfloat16),
      modulationIndices:MLXArray.zeros([6],dtype:.int32),positions:MLXArray.zeros([6,3],dtype:.float32))) { error in
      XCTAssertTrue(String(describing:error).contains("explicit matching tile geometry"))
    }
  }
  func testNinetyPercentSparsityDoesNotSelectAnExtraTileAtExactBoundaries() {
    XCTAssertEqual(H3FastAttention.selectedTileCount(videoTiles:10,sparsity:0.9),1)
    XCTAssertEqual(H3FastAttention.selectedTileCount(videoTiles:100,sparsity:0.9),10)
    XCTAssertEqual(H3FastAttention.selectedTileCount(videoTiles:11,sparsity:0.9),2)
  }
  func testCubesKeepTextAndAudioSegmentsSeparateAndCoverPartialBoundaries() throws {
    let tiles = try H3FastTiles(prefixSegments: [3, 5], videoGrid: [3, 5, 6])
    XCTAssertEqual(tiles.prefixTiles, 2)
    XCTAssertEqual(tiles.rows, 98)
    XCTAssertEqual(tiles.sizes.reduce(0,+), tiles.rows)
    XCTAssertEqual(Set(tiles.rowSlots).count, tiles.rows)
    XCTAssertEqual(Array(tiles.rowSlots.prefix(3)), [0,1,2])
    XCTAssertEqual(Array(tiles.rowSlots[3..<8]), [64,65,66,67,68])
    XCTAssertThrowsError(try H3FastTiles(prefixSegments: [0], videoGrid: [3,5,6]))
  }

  func testSparseSelectionAndPaddingMatchIndependentScalarReference() throws {
    let url = try XCTUnwrap(Bundle.module.url(forResource:"fasth3-vsa-small",withExtension:"json",subdirectory:"Fixtures"))
    let data = try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as! [String:Any]
    let shape = data["shape"] as! [Int]
    func array(_ key:String) -> MLXArray { MLXArray((data[key] as! [Double]).map(Float.init),shape) }
    let tiles = try H3FastTiles(prefixSegments:data["prefixSegments"] as! [Int],videoGrid:data["videoGrid"] as! [Int])
    let result = try H3FastAttention.evaluate(query:array("query"),key:array("key"),value:array("value"),
      gate:array("gate"),tiles:tiles,minimumSparseRows:64)
    XCTAssertLessThan(abs(result-array("expected")).max().item(Float.self),3e-6)
  }

  func testDenseFallbackRetainsLearnedCompressionAndSparseMatchesAllTiles() throws {
    let tiles = try H3FastTiles(prefixSegments: [2,3], videoGrid: [2,2,3])
    let rows = tiles.rows, heads = 2, width = 4
    let values = (0..<(heads*rows*width)).map { Float($0 % 29) / 29 }
    let q = MLXArray(values,[1,heads,rows,width])
    let k = q * 0.7, v = q * 0.4, gate = MLXArray.full([1,heads,rows,width], values: MLXArray(Float(0.5)))
    let dense = try H3FastAttention.evaluate(query:q,key:k,value:v,gate:gate,tiles:tiles,
      sparsity:0.9,minimumSparseRows:4096)
    let sparseAll = try H3FastAttention.evaluate(query:q,key:k,value:v,gate:gate,tiles:tiles,
      sparsity:0,minimumSparseRows:1)
    let ordinary = MLXFast.scaledDotProductAttention(queries:q,keys:k,values:v,
      scale:1 / Float(width).squareRoot(),mask:nil)
    XCTAssertLessThan(abs(dense-sparseAll).max().item(Float.self), 2e-6)
    XCTAssertGreaterThan(abs(dense-ordinary).max().item(Float.self), 0.01)
    XCTAssertThrowsError(try H3FastAttention.evaluate(query:q,key:k,value:v,gate:gate,
      tiles:tiles,sparsity:1))
  }
}
