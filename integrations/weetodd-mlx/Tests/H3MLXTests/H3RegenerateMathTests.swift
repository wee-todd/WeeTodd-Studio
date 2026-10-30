import MLX
import XCTest
@testable import H3MLX

final class H3RegenerateMathTests: XCTestCase {
  func testSplitQKVUsesIndependentFactorsAndHeadMajorOrder() throws {
    let pairs = (0..<3).map { modality in
      H3RegenerateProjection.Pair(a:MLXArray([Float(modality+1)],[1,1]),
        b:MLXArray([Float(1),2,3,4],[4,1]),scale:1)
    }
    let input=MLXArray([Float(2)],[1,1,1])
    let delta=try H3RegenerateProjection.delta(input:input,pairs:pairs,layout:.splitQKV(heads:2,headWidth:2))
    XCTAssertEqual(delta.asArray(Float.self),[2,4,4,8,6,12,6,8,12,16,18,24])
    let merged=try H3RegenerateProjection.merge(base:.zeros([12,1]),pairs:pairs,
      layout:.splitQKV(heads:2,headWidth:2),rowChunk:3)
    XCTAssertEqual(matmul(input,merged.T).asArray(Float.self),delta.asArray(Float.self))
  }
  func testExtractedQKVAndOfficialSwiGLUSwapExactlyOnce() throws {
    let pair=H3RegenerateProjection.Pair(a:MLXArray([Float(1)],[1,1]),
      b:MLXArray((1...12).map(Float.init),[12,1]),scale:1)
    let input=MLXArray([Float(1)],[1,1])
    let qkv=try H3RegenerateProjection.delta(input:input,pairs:[pair],layout:.contiguousQKV(heads:2,headWidth:2))
    XCTAssertEqual(qkv.asArray(Float.self),[1,2,5,6,9,10,3,4,7,8,11,12])
    let swapped=try H3RegenerateProjection.delta(input:input,pairs:[pair],layout:.swapSwiGLU)
    XCTAssertEqual(swapped.asArray(Float.self),[7,8,9,10,11,12,1,2,3,4,5,6])
    let native=try H3RegenerateProjection.delta(input:input,pairs:[pair],layout:.native)
    XCTAssertEqual(native.asArray(Float.self),(1...12).map(Float.init))
  }
  func testUnitAndExplicitAlphaScalingDoNotDivideBakedFactorsAgain() throws {
    let a=MLXArray([Float(1),2],[2,1]), b=MLXArray([Float(3),4],[1,2])
    let cases: [(Float,Float)] = [(1,22),(0.5,11)]
    for (scale,want) in cases {
      let pair=H3RegenerateProjection.Pair(a:a,b:b,scale:scale)
      XCTAssertEqual(try H3RegenerateProjection.delta(input:MLXArray([Float(2)],[1,1]),pairs:[pair],layout:.native).item(Float.self),want)
    }
  }
  func testFP32RuntimeToleranceAndBF16PremergeRoundingAreSeparateContracts() throws {
    // Fixed acceptance: FP32 <= 1e-5 absolute; publisher BF16 must equal
    // FP32 weight premerge then cast exactly. Runtime BF16 is NOT equivalent.
    let pair=H3RegenerateProjection.Pair(a:MLXArray([Float(0.003)],[1,1]),b:.ones([1,1]),scale:1)
    let source=MLXArray([Float(3)],[1,1])
    let base=MLXArray.ones([1,1])
    let runtime=matmul(source,base.T) + (try H3RegenerateProjection.delta(input:source,pairs:[pair],layout:.native))
    let merged=try H3RegenerateProjection.merge(base:base,pairs:[pair],layout:.native,rowChunk:1)
    XCTAssertEqual(runtime.item(Float.self),matmul(source,merged.T).item(Float.self),accuracy:1e-5)
    let bfBase=base.asType(.bfloat16), bfInput=source.asType(.bfloat16)
    let bfMerged=try H3RegenerateProjection.merge(base:bfBase,pairs:[pair],layout:.native,rowChunk:1)
    let premerged=matmul(bfInput,bfMerged.T).item(Float.self)
    let runtimeBF=(matmul(bfInput,bfBase.T) + (try H3RegenerateProjection.delta(input:bfInput,pairs:[pair],layout:.native)).asType(.bfloat16)).item(Float.self)
    XCTAssertEqual(premerged,3)
    XCTAssertEqual(runtimeBF,3.015625)
    XCTAssertEqual(bfBase.item(Float.self),1,"Merge must not mutate the base identity")
  }
  func testMalformedPairsAndMergeWindowsRejectBeforeMatmul() throws {
    let pair=H3RegenerateProjection.Pair(a:.ones([2,3]),b:.ones([4,1]),scale:1)
    XCTAssertThrowsError(try H3RegenerateProjection.delta(input:.ones([1,3]),pairs:[pair],layout:.native))
    XCTAssertThrowsError(try H3RegenerateProjection.merge(base:.zeros([4,3]),pairs:[],layout:.native,rowChunk:0))
  }
  func testAffineQ8ReuseDecodesOnlyRequestedRowsBeforePublisherCast() throws {
    let packed=MLXArray([UInt32](repeating:0x03020100,count:32),[2,16])
    let scales=MLXArray([Float(0.5),1],[2,1]).asType(.bfloat16)
    let biases=MLXArray([Float(2),4],[2,1]).asType(.bfloat16)
    let rows=try H3RegenerateProjection.affineRows(parameters:[packed,scales,biases],range:1..<2)
    XCTAssertEqual(rows.shape,[1,64])
    XCTAssertEqual(Array(rows.asArray(Float.self).prefix(4)),[4,5,6,7])
    XCTAssertThrowsError(try H3RegenerateProjection.affineRows(parameters:[packed,scales,biases],range:1..<3))
    let pair=H3RegenerateProjection.Pair(a:.zeros([1,64]),b:.ones([2,1]),scale:1)
    var requested:[Range<Int>]=[]
    let result=try H3RegenerateProjection.projectMerged(input:.ones([1,64],dtype:.bfloat16),
      pairs:[pair],layout:.native,rows:2,columns:64,rowChunk:1) { range in
      requested.append(range)
      return try H3RegenerateProjection.affineRows(parameters:[packed,scales,biases],range:range)
    }
    XCTAssertEqual(requested,[0..<1,1..<2])
    XCTAssertEqual(result.asArray(Float.self),[176,352])
  }

}
