import XCTest
import MLX
@testable import FlashVSRMLX

final class FlashVSRRoutingTests: XCTestCase {
  func testReleasedSchedulesAndCache() throws {
    let turbo = FlashVSRConfig(variant: .turbo)
    XCTAssertEqual(turbo.layers, 30); XCTAssertEqual(turbo.width, 1536)
    XCTAssertEqual(turbo.heads, 12); XCTAssertEqual(turbo.latentChannels, 16)
    XCTAssertEqual((0..<30).filter { !turbo.isReplacement($0) }, [0,6,12,18,24])
    XCTAssertEqual(turbo.cacheWindows(layer: 1), 1)
    XCTAssertEqual(turbo.cacheWindows(layer: 0), 3)
    XCTAssertTrue((0..<30).allSatisfy { !FlashVSRConfig(variant: .base).isReplacement($0) })
  }
  func testStrictThresholdTiesAndFirstCandidateFallback() throws {
    let route = try FlashVSRBlockRouting.select(probabilities: [[0.5,0.5],[0.5,0.5]], candidates: [[0,1],[0,1]], temporalGroups: 1, topK: 2)
    XCTAssertEqual(route, [[0],[0]])
    let distinct = try FlashVSRBlockRouting.select(probabilities: [[0.9,0.1],[0.6,0.4]], candidates: [[0,1],[0,1]], temporalGroups: 1, topK: 2)
    XCTAssertEqual(distinct, [[0],[0]])
  }
  func testLocalGridAndCompleteCacheSuffix() throws {
    let c = try FlashVSRBlockRouting.candidates(queryBlocks: 8, keyBlocks: 12, rows: 2, columns: 2, localRange: 1)
    XCTAssertEqual(c[3], [3,7,11]); XCTAssertEqual(c[4], [0,4,8])
    XCTAssertEqual(try FlashVSRBlockRouting.cacheIndices(keyBlocks: 12, spatialBlocks: 4, windows: 1), Array(8..<12))
    XCTAssertThrowsError(try FlashVSRBlockRouting.cacheIndices(keyBlocks: 11, spatialBlocks: 4, windows: 1))
  }
  func testMaskedOracleRejectsDenseSubstitution() throws {
    let q: [[Float]] = [[1,0], [0,1]], k: [[Float]] = [[1,0],[0,1]], v: [[Float]] = [[10,0],[0,20]]
    let selected = try FlashVSRAttention.oracle(queries:q, keys:k, values:v, selectedKeys:[[0],[1]])
    XCTAssertEqual(selected, v)
    let dense = try FlashVSRAttention.oracle(queries:q, keys:k, values:v, selectedKeys:[[0,1],[0,1]])
    XCTAssertNotEqual(selected, dense)
  }
  func testThreeGroupsAndResetIdentity() throws {
    let state = FlashVSRStreamState(sourceID: "clip-a")
    for i in 0..<3 { try state.validate(sourceID:"clip-a", tokenShape:[4,8,8]); XCTAssertEqual(state.groupIndex,i); state.finishGroup() }
    XCTAssertThrowsError(try state.validate(sourceID:"clip-b", tokenShape:[4,8,8]))
    state.reset(sourceID:"clip-b")
    try state.validate(sourceID:"clip-b", tokenShape:[4,8,8]); XCTAssertEqual(state.groupIndex,0)
  }
}
