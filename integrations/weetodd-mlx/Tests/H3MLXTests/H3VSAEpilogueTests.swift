import XCTest
import MLX
import MLXRandom
@testable import H3MLX

final class H3VSAEpilogueTests: XCTestCase {
  private func reference(prefix: MLXArray, video: MLXArray, compression: MLXArray,
    gate: MLXArray, tiles: H3FastTiles) -> MLXArray {
    let expanded = take(compression,MLXArray(tiles.rowSlots.map { $0/64 }),axis:2)
      .asType(.bfloat16)
    let scatter = MLXArray(tiles.rowSlots.dropFirst(tiles.prefixRows)
      .map { $0-Int32(tiles.prefixTiles*64) })
    return concatenated([prefix,take(video,scatter,axis:2)],axis:2)+expanded*gate
  }
  private func assertParity(prefix: MLXArray, video: MLXArray, compression: MLXArray,
    gate: MLXArray, tiles: H3FastTiles, file: StaticString = #filePath,
    line: UInt = #line) throws {
    let actual = try H3VSAEpilogue.evaluate(prefixOutput:prefix,paddedVideo:video,
      compression:compression,gate:gate,tiles:tiles)
    let expected = reference(prefix:prefix,video:video,compression:compression,
      gate:gate,tiles:tiles)
    XCTAssertEqual(actual.shape,gate.shape,file:file,line:line)
    XCTAssertEqual(actual.dtype,.bfloat16,file:file,line:line)
    XCTAssertEqual(actual.view(dtype:.uint16).asArray(UInt16.self),
      expected.view(dtype:.uint16).asArray(UInt16.self),file:file,line:line)
  }

  func testPartialPrefixSegmentsAndVideoCubesMatchUnfusedBits() throws {
    let tiles = try H3FastTiles(prefixSegments:[3,65],videoGrid:[7,5,6])
    for heads in [1,3,56] {
      let prefix = MLXRandom.normal([1,heads,tiles.prefixRows,128],
        key:MLXRandom.key(981)).asType(.bfloat16)
      let video = MLXRandom.normal([1,heads,(tiles.sizes.count-tiles.prefixTiles)*64,128],
        key:MLXRandom.key(982)).asType(.bfloat16)
      let compression = MLXRandom.normal([1,heads,tiles.sizes.count,128],key:MLXRandom.key(983))
      let gate = MLXRandom.normal([1,heads,tiles.rows,128],key:MLXRandom.key(984)).asType(.bfloat16)
      try assertParity(prefix:prefix,video:video,compression:compression,gate:gate,tiles:tiles)
    }
  }

  func testTransposedStridedReversedAndBroadcastInputsMatchUnfusedBits() throws {
    let tiles = try H3FastTiles(prefixSegments:[3,65],videoGrid:[3,5,7])
    let heads = 3, videoRows = (tiles.sizes.count-tiles.prefixTiles)*64
    func strided(_ rows: Int, seed: UInt64, dtype: DType) -> MLXArray {
      MLXRandom.normal([1,rows*2,heads*2,256],key:MLXRandom.key(seed)).asType(dtype)[0..<1,.stride(by:2),.stride(by:2),.stride(by:2)].transposed(0,2,1,3)
    }
    let p = strided(tiles.prefixRows,seed:985,dtype:.bfloat16)
    let v = strided(videoRows,seed:986,dtype:.bfloat16)
    let c = strided(tiles.sizes.count,seed:987,dtype:.float32)
    let g = strided(tiles.rows,seed:988,dtype:.bfloat16)
    try assertParity(prefix:p,video:v,compression:c,gate:g,tiles:tiles)
    func reversed(_ x: MLXArray) -> MLXArray {
      x[0..<1,.stride(by:-1),.stride(by:-1),.stride(by:-1)]
    }
    try assertParity(prefix:reversed(p),video:reversed(v),compression:reversed(c),
      gate:reversed(g),tiles:tiles)
    func repeated(_ x: MLXArray) -> MLXArray {
      broadcast(x[0..<1,0..<1,0..<1,0..<1],to:x.shape)
    }
    try assertParity(prefix:repeated(p),video:repeated(v),compression:repeated(c),
      gate:repeated(g),tiles:tiles)
  }

  func testBF16CastMultiplyAndAddEdgesRemainSeparate() throws {
    let tiles = try H3FastTiles(prefixSegments:[3],videoGrid:[1,1,3])
    let bits: [UInt16] = [0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,
      0x3f7f,0xbf7f,0x3f80,0xbf80,0x3f81,0xbf81,0x7f7f,0xff7f]
    func bf16(_ rows: Int, phase: Int) -> MLXArray {
      MLXArray((0..<(rows*128)).map { bits[($0+phase)%bits.count] },[1,1,rows,128])
        .view(dtype:.bfloat16)
    }
    let values: [Float] = [0,-0.0,1,-1,0.5,-0.5,1.00390625,1.00390637,
      -1.00390625,-1.00390637,Float.leastNonzeroMagnitude,Float.leastNormalMagnitude,
      Float.greatestFiniteMagnitude,-Float.greatestFiniteMagnitude]
    let c = MLXArray((0..<(tiles.sizes.count*128)).map { values[$0%values.count] },
      [1,1,tiles.sizes.count,128])
    // Bounded gates avoid NaN payload comparisons while covering cast overflow,
    // BF16 subnormals, signed zero, cancellation, and half-way cast rounding.
    let gate = MLXArray((0..<(tiles.rows*128)).map { Float($0%4+1)/4 },
      [1,1,tiles.rows,128]).asType(.bfloat16)
    try assertParity(prefix:bf16(tiles.prefixRows,phase:0),
      video:bf16((tiles.sizes.count-tiles.prefixTiles)*64,phase:3),
      compression:c,gate:gate,tiles:tiles)
  }

  func testCPUFallbackAndInvalidHeadShapeDTypeRejectBeforeKernel() throws {
    try Device.withDefaultDevice(.cpu) {
      let tiles = try H3FastTiles(prefixSegments:[3],videoGrid:[1,1,1])
      let p = MLXArray.ones([1,1,tiles.prefixRows,128],dtype:.bfloat16)
      let v = MLXArray.ones([1,1,(tiles.sizes.count-tiles.prefixTiles)*64,128],dtype:.bfloat16)
      let c = MLXArray.ones([1,1,tiles.sizes.count,128])
      let g = MLXArray.ones([1,1,tiles.rows,128],dtype:.bfloat16)
      try assertParity(prefix:p,video:v,compression:c,gate:g,tiles:tiles)
      XCTAssertThrowsError(try H3VSAEpilogue.evaluate(prefixOutput:p,paddedVideo:v,
        compression:c.asType(.bfloat16),gate:g,tiles:tiles))
      XCTAssertThrowsError(try H3VSAEpilogue.evaluate(prefixOutput:p,paddedVideo:v,
        compression:c,gate:g[.ellipsis,0..<64],tiles:tiles))
      XCTAssertThrowsError(try H3VSAEpilogue.evaluate(prefixOutput:p,paddedVideo:v,
        compression:c,gate:broadcast(g,to:[1,57,tiles.rows,128]),tiles:tiles))
    }
  }

  func testCancellationRejectsWorkBeforeSubmission() async throws {
    let cancelled = Task { () throws -> Void in
      withUnsafeCurrentTask { $0?.cancel() }
      let tiles = try H3FastTiles(prefixSegments:[3],videoGrid:[1,1,1])
      _ = try H3VSAEpilogue.evaluate(
        prefixOutput:MLXArray.ones([1,1,3,128],dtype:.bfloat16),
        paddedVideo:MLXArray.ones([1,1,64,128],dtype:.bfloat16),
        compression:MLXArray.ones([1,1,2,128]),
        gate:MLXArray.ones([1,1,4,128],dtype:.bfloat16),tiles:tiles)
    }
    do { try await cancelled.value; XCTFail("Cancelled VSA epilogue executed.") }
    catch is CancellationError { }
  }
}
