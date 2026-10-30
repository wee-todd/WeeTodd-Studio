import XCTest
import MLX
@testable import FlashVSRMLX

final class FlashVSRAttentionTests: XCTestCase {
  func testBoundedAttentionAgainstIndependentOracle() throws {
    let q:[[Float]]=[[1,0],[0,1],[1,1],[-1,0]],k:[[Float]]=[[1,0],[0,1],[2,1],[-1,2]],v:[[Float]]=[[10,0],[0,20],[3,5],[7,9]]
    let selection=[[[[0],[1]]]]
    let actual=try FlashVSRAttention.attend(queries:MLXArray(q.flatMap{$0},[1,1,2,2,2]),keys:MLXArray(k.flatMap{$0},[1,1,2,2,2]),values:MLXArray(v.flatMap{$0},[1,1,2,2,2]),selection:selection,keySlabBlocks:1).asArray(Float.self)
    let expected=try FlashVSRAttention.oracle(queries:q,keys:k,values:v,selectedKeys:[[0,1],[0,1],[2,3],[2,3]]).flatMap{$0}
    for (a,b) in zip(actual,expected) { XCTAssertEqual(a,b,accuracy:2e-5) }
    let global=try FlashVSRAttention.attend(queries:MLXArray(q.flatMap{$0},[1,1,2,2,2]),keys:MLXArray(k.flatMap{$0},[1,1,2,2,2]),values:MLXArray(v.flatMap{$0},[1,1,2,2,2]),selection:[[[[0,1],[0,1]]]],keySlabBlocks:1).asArray(Float.self)
    XCTAssertNotEqual(actual,global)
  }
  func testWindowRoundtripAndInvalidPadding() throws {
    let x=MLXArray(Array(0..<1024).map(Float.init),[1,1,512,2])
    let packed=try FlashVSRAttention.partition(x,shape:[4,8,16])
    XCTAssertEqual(packed.shape,[1,1,4,128,2])
    XCTAssertEqual(FlashVSRAttention.unpartition(packed,shape:[4,8,16]).asArray(Float.self),x.asArray(Float.self))
    XCTAssertThrowsError(try FlashVSRAttention.partition(x,shape:[4,8,15]))
  }
  func testRoutingThreeGroupsPreservesWholeSpatialGrids() throws {
    // Constant vectors tie every score, so the release's fallback is the earliest
    // spatially admissible historical window, never an arbitrary last raster slice.
    let q=MLXArray.ones([1,1,8,2,2])
    var k=MLXArray.zeros([1,1,0,2,2])
    for group in 0..<3 {
      k=concatenated([k,q],axis:2)
      let route=try FlashVSRAttention.route(q:q,k:k,rows:2,columns:2,localRange:1,topK:0)
      XCTAssertEqual(route[0][0][3],[3])
      let suffix=try FlashVSRBlockRouting.cacheIndices(keyBlocks:k.shape[2],spatialBlocks:4,windows:1)
      XCTAssertEqual(suffix.count,4)
      k=take(k,MLXArray(suffix.map(Int32.init)),axis:2)
      XCTAssertEqual(k.shape[2],4,"group \(group)")
    }
  }
  func testRotaryAxisAllocationAndOffset() throws {
    let x=MLXArray([Float(1),0,1,0,1,0],[1,1,1,6])
    let y=try FlashVSRTransformer.rotary(x,shape:[1,1,1],temporalOffset:1).asArray(Float.self)
    XCTAssertEqual(y[0],cos(Float(1)),accuracy:1e-6); XCTAssertEqual(y[1],sin(Float(1)),accuracy:1e-6)
    XCTAssertEqual(Array(y[2...]),[1,0,1,0])
  }
  func testPooledRoutingMatchesGroupedOracleForMovingFeatures() throws {
    let qv:[Float]=[4,0,0,3,-2,1,1,-4],kv:[Float]=[-1,3,2,0,1,-3,4,1,0,1,-2,2]
    let q=MLXArray(qv,[1,1,4,1,2]),k=MLXArray(kv,[1,1,6,1,2])
    let result=try FlashVSRAttention.route(q:q,k:k,rows:1,columns:2,localRange:11,topK:5)
    var probabilities=[[Float]]()
    for row in 0..<4 {
      let scores=(0..<6).map { (qv[2*row]*kv[2*$0]+qv[2*row+1]*kv[2*$0+1])/sqrt(Float(2)) }
      let e=scores.map { exp($0-scores.max()!) },total=e.reduce(0,+)
      probabilities.append(e.map { $0/total })
    }
    let oracle=try FlashVSRBlockRouting.select(probabilities:probabilities,candidates:Array(repeating:Array(0..<6),count:4),temporalGroups:1,topK:5)
    XCTAssertEqual(result[0][0],oracle)
    XCTAssertTrue(result[0][0].contains(where:{$0.contains(5)}),"motion can retrieve a different spatial/history block")
  }
  func testEmptyAndShortGroupAdmission() throws {
    XCTAssertThrowsError(try FlashVSRBlockRouting.candidates(queryBlocks:0,keyBlocks:0,rows:1,columns:1,localRange:3))
    let state=FlashVSRStreamState(sourceID:"x")
    XCTAssertThrowsError(try state.validate(sourceID:"x",tokenShape:[3,8,8]))
    try state.bindModel("turbo")
    XCTAssertThrowsError(try state.bindModel("base"))
  }

  func testBF16AttentionWithinFrozenTolerance() throws {
    let q=MLXArray([Float(0.2),-0.4,0.7,0.1],[1,1,1,2,2])
    let k=MLXArray([Float(0.3),0.2,-0.1,0.8],[1,1,1,2,2])
    let v=MLXArray([Float(1.2),-0.9,0.4,0.6],[1,1,1,2,2])
    let selected=[[[[0]]]]
    let fp32=try FlashVSRAttention.attend(queries:q,keys:k,values:v,selection:selected).asArray(Float.self)
    let bf16=try FlashVSRAttention.attend(queries:q.asType(.bfloat16),keys:k.asType(.bfloat16),values:v.asType(.bfloat16),selection:selected).asType(.float32).asArray(Float.self)
    for (a,b) in zip(fp32,bf16) { XCTAssertEqual(a,b,accuracy:0.01) }
  }

  func testBaseCacheRetainsThreeTemporalWindowsAcrossGroups() throws {
    var cached=0
    var sizes=[Int]()
    for _ in 0..<3 {
      let suffix=try FlashVSRBlockRouting.cacheIndices(keyBlocks:cached+8,spatialBlocks:4,windows:3)
      cached=suffix.count; sizes.append(cached)
      XCTAssertEqual(suffix.count%4,0)
    }
    XCTAssertEqual(sizes,[8,12,12])
  }

}
