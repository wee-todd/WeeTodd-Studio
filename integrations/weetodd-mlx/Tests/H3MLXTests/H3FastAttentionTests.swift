import XCTest
import MLX
@testable import H3MLX

final class H3FastAttentionTests: XCTestCase {
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
