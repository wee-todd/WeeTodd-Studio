import XCTest
import MLX
@testable import H3MLX

final class H3OmniVSATests: XCTestCase {
  private let segments: [H3OmniSegment] = [
    .dense(role:.text, rows:2), .referenceVideo(grid:[1,1,17]),
    .dense(role:.image, rows:3), .referenceVideo(grid:[1,1,9]),
    .dense(role:.referenceAudio, rows:2), .dense(role:.targetAudio, rows:1)
  ]

  func testTrainedTileGeometryAndInversePreserveOrderedMedia() throws {
    let plan = try H3OmniVSAPlan(prefixSegments:segments, targetVideoGrid:[5,5,9])
    XCTAssertEqual(H3OmniVSAPlan.tileShape,[4,4,8])
    XCTAssertEqual(plan.denseTiles,4)
    XCTAssertEqual(plan.videoRegions,[4..<7,7..<9,9..<17])
    XCTAssertEqual(plan.sizes,[2,3,2,1,8,8,1,8,1,128,16,32,4,32,4,8,1])
    XCTAssertEqual(plan.denseRows,[0,1,19,20,21,31,32,33])
    XCTAssertEqual(plan.rows,259)
    XCTAssertEqual(Set(plan.rowSlots).count,259)
    for row in 0..<plan.rows {
      let slot = Int(plan.rowSlots[row])
      XCTAssertEqual(plan.indices[slot],Int32(row))
      XCTAssertLessThan(slot%128,plan.sizes[slot/128])
    }
    // Width is eight, temporal extent four: catches accidental 2×8×8 tiles.
    XCTAssertEqual(Array(plan.indices[(9*128)..<(9*128+10)]),[34,35,36,37,38,39,40,41,43,44])
    XCTAssertEqual(plan.indices[9*128+96],169)
  }

  func testRoutesSelectEachReferenceAndTargetIndependentlyWithStableTies() throws {
    let plan = try H3OmniVSAPlan(prefixSegments:segments,targetVideoGrid:[1,1,17])
    let n = plan.sizes.count, sparse = n-plan.denseTiles
    let values: [Float] = [9,8,7,6, 100,99,98, -20,-10, 0,1,2]
    let scores = broadcast(MLXArray(values,[1,1,1,n]),to:[1,2,n,n])
    let actual = try H3OmniAttention.routes(scores:scores,plan:plan).asArray(Int32.self)
    XCTAssertEqual(actual,Array(repeating:[Int32(0),1,2,3,4,8,11],count:2*sparse).flatMap{$0})
    let ties = MLXArray.zeros([1,2,n,n])
    XCTAssertEqual(try H3OmniAttention.routes(scores:ties,plan:plan).asArray(Int32.self),
      Array(repeating:[Int32(0),1,2,3,4,7,9],count:2*sparse).flatMap{$0})
    XCTAssertEqual(H3OmniVSAPlan.keepCount(10),1)
    XCTAssertEqual(H3OmniVSAPlan.keepCount(11),2)
    XCTAssertEqual(H3OmniVSAPlan.keepCount(100),10)
  }

  // Independent oracle: classify each actual row by coordinates (no production
  // gather, rowSlots, sizes, routes, or Metal helper is used).
  private func oracleTiles() -> [[Int]] {
    [[0,1],[19,20,21],[31,32],[33],Array(2..<10),Array(10..<18),[18],
     Array(22..<30),[30],Array(34..<42),Array(42..<50),[50]]
  }
  private func oracle(q:[Float], k:[Float], v:[Float], gate:[Float], heads:Int,
    width:Int, cast: (Float)->Float) -> [Float] {
    let tiles=oracleTiles(), rows=51, regions=[4..<7,7..<9,9..<12]
    let scale=1/Float(width).squareRoot()
    var result=[Float](repeating:0,count:q.count)
    for head in 0..<heads {
      func offset(_ row:Int,_ feature:Int)->Int { (head*rows+row)*width+feature }
      func pool(_ data:[Float])->[[Float]] {
        tiles.map { members in (0..<width).map { d in
          members.reduce(Float(0)) { $0+data[offset($1,d)] }/Float(members.count)
        } }
      }
      let pq=pool(q),pk=pool(k),pv=pool(v)
      func probs(_ logits:[Float])->[Float] {
        let m=logits.max()!, e=logits.map{exp($0-m)},sum=e.reduce(0,+)
        return e.map{$0/sum}
      }
      for (tile,members) in tiles.enumerated() {
        let scores=pk.map { key in zip(pq[tile],key).reduce(Float(0)){$0+$1.0*$1.1}*scale }
        let cp=probs(scores)
        let compression=(0..<width).map { d in
          (0..<tiles.count).reduce(Float(0)){$0+cp[$1]*pv[$1][d]}
        }
        var allowed=Array(0..<rows)
        if tile >= 4 {
          var selected=Array(0..<4)
          for region in regions {
            selected.append(region.sorted { a,b in scores[a] == scores[b] ? a<b : scores[a]>scores[b] }[0])
          }
          allowed=selected.flatMap{tiles[$0]}
        }
        for row in members {
          let logits=allowed.map { keyRow in
            (0..<width).reduce(Float(0)){$0+q[offset(row,$1)]*k[offset(keyRow,$1)]}*scale
          }
          let probability=probs(logits)
          for d in 0..<width {
            let attended=zip(allowed,probability).reduce(Float(0)){$0+v[offset($1.0,d)]*$1.1}
            result[offset(row,d)]=cast(cast(attended)+cast(compression[d]*gate[offset(row,d)]))
          }
        }
      }
    }
    return result
  }
  private func bf16(_ x:Float)->Float {
    let bits=x.bitPattern
    return Float(bitPattern:(bits &+ 0x7fff &+ ((bits>>16)&1)) & 0xffff0000)
  }

  func testNative128AttentionMatchesIndependentDenseMaskedCPUOracle() throws {
    let plan=try H3OmniVSAPlan(prefixSegments:segments,targetVideoGrid:[1,1,17])
    let shape=[1,2,51,128],n=shape.reduce(1,*)
    let q=(0..<n).map { bf16(Float(($0*13)%37-18)/30) }
    let k=(0..<n).map { bf16(Float(($0*7)%41-20)/35) }
    let v=(0..<n).map { bf16(Float(($0/128)%51)/19 + Float($0%11)/9) }
    let gate=(0..<n).map { bf16($0%3 == 0 ? -1.75 : ($0%3 == 1 ? 2.25 : 0)) }
    func array(_ x:[Float])->MLXArray { MLXArray(x,shape).asType(.bfloat16) }
    let actual=try H3OmniAttention.evaluate(query:array(q),key:array(k),value:array(v),gate:array(gate),plan:plan)
    let expected=oracle(q:q,k:k,v:v,gate:gate,heads:2,width:128,cast:bf16)
    XCTAssertEqual(actual.dtype,.bfloat16)
    XCTAssertEqual(actual.shape,shape)
    XCTAssertLessThanOrEqual(zip(actual.asArray(Float.self),expected).map{abs($0-$1)}.max()!,0.0625)
    // The geometry still executes trained sparse attention for a tiny fixture.
    XCTAssertEqual(plan.selectedTilesPerQuery,7)
  }

  func testCompressionCastsAfterSignedGateProductAndDenseQueriesGetCompression() throws {
    let plan=try H3OmniVSAPlan(prefixSegments:[.dense(role:.text,rows:1)],targetVideoGrid:[1,1,3])
    let shape=[1,1,4,128],q=MLXArray.zeros(shape,dtype:.bfloat16)
    let v=broadcast(MLXArray([Float(0.01),0.1,0.7,1.3],[1,1,4,1]).asType(.bfloat16),to:shape)
    let gate=MLXArray.full(shape,values:MLXArray(Float(-1.75)),dtype:.bfloat16)
    let actual=try H3OmniAttention.evaluate(query:q,key:q,value:v,gate:gate,plan:plan).asArray(Float.self)
    let values=[Float(0.01),0.1,0.7,1.3].map(bf16)
    let c=(values[0]+values.dropFirst().reduce(0,+)/3)/2
    let expected=bf16(bf16(values.reduce(0,+)/4)+bf16(c * -1.75))
    let wrong=bf16(bf16(values.reduce(0,+)/4)+bf16(bf16(c) * -1.75))
    XCTAssertNotEqual(expected,wrong)
    XCTAssertEqual(actual,Array(repeating:expected,count:512))
  }

  func testFull128TilesRaggedAxesAndNoDensePrefixUseAllFourQueryBlocks() throws {
    let plan=try H3OmniVSAPlan(prefixSegments:[],targetVideoGrid:[5,5,9])
    let shape=[1,2,225,128]
    let q=MLXArray.zeros(shape,dtype:.bfloat16)
    let values=(0..<225).map { bf16(Float($0%19-9)/7) }
    let v=broadcast(MLXArray(values,[1,1,225,1]).asType(.bfloat16),to:shape)
    let gate=MLXArray.full(shape,values:MLXArray(Float(2.25)),dtype:.bfloat16)
    // Independently bucket actual coordinates; no plan indices/sizes are used.
    var groups=[Int:[Int]]()
    for row in 0..<225 {
      let t=row/45,h=(row%45)/9,w=row%9
      let bucket=(t/4)*4+(h/4)*2+w/8
      groups[bucket,default:[]].append(row)
    }
    let means=(0..<8).map { b in groups[b]!.reduce(Float(0)){$0+values[$1]}/Float(groups[b]!.count) }
    let expected=bf16(bf16(means[0])+bf16(means.reduce(0,+)/8 * 2.25))
    let result=try H3OmniAttention.evaluate(query:q,key:q,value:v,gate:gate,plan:plan)
    XCTAssertEqual(result.asArray(Float.self),Array(repeating:expected,count:shape.reduce(1,*)))
    let pooled=try H3OmniMetal.summaries(query:q,key:q,value:v,plan:plan)[2].asArray(Float.self)
    for head in 0..<2 { for tile in 0..<8 { for feature in 0..<128 {
      XCTAssertEqual(pooled[(head*8+tile)*128+feature],means[tile],accuracy:1e-6)
    } } }
  }

  func testStridedHeadMajorInputsMatchContiguousAndRejectUnsupportedFeatures() throws {
    let plan=try H3OmniVSAPlan(prefixSegments:segments,targetVideoGrid:[1,1,17])
    let data: [Float]=(0..<39168).map { (i: Int) -> Float in bf16(Float((i*13)%37-18)/30) }
    let fused=MLXArray(data,[1,51,2,3,128]).asType(.bfloat16)
    let q=fused[.ellipsis,0,0..<128].transposed(0,2,1,3)
    let k=fused[.ellipsis,1,0..<128].transposed(0,2,1,3)
    let v=fused[.ellipsis,2,0..<128].transposed(0,2,1,3)
    let actual=try H3OmniAttention.evaluate(query:q,key:k,value:v,gate:q,plan:plan)
    let expected=try H3OmniAttention.evaluate(query:contiguous(q),key:contiguous(k),value:contiguous(v),gate:contiguous(q),plan:plan)
    XCTAssertEqual(actual.asArray(Float.self),expected.asArray(Float.self))
    XCTAssertThrowsError(try H3OmniAttention.evaluate(query:q[.ellipsis,.stride(by:-1)],key:k,value:v,gate:q,plan:plan))
    XCTAssertThrowsError(try H3OmniAttention.routes(scores:MLXArray.zeros([1,1,1,1]),plan:plan))
  }

  func testAdmissionBoundsRejectInvalidGeometryAndGatesBeforeGPUWork() throws {
    XCTAssertThrowsError(try H3OmniVSAPlan(prefixSegments:[.dense(role:.text,rows:0)],targetVideoGrid:[1,1,1]))
    XCTAssertThrowsError(try H3OmniVSAPlan(prefixSegments:[],targetVideoGrid:[0,1,1]))
    XCTAssertThrowsError(try H3OmniVSAPlan(prefixSegments:[],targetVideoGrid:[512,512,512]))
    XCTAssertThrowsError(try H3OmniVSAPlan(prefixSegments:[.referenceVideo(grid:[1,0,1])],targetVideoGrid:[1,1,1]))
    let maxRows=try H3OmniVSAPlan(prefixSegments:[],targetVideoGrid:[100,20,20])
    XCTAssertEqual(maxRows.rows,40_000)
    XCTAssertLessThan(try maxRows.workspaceBytes(heads:56),H3OmniVSAPlan.maximumWorkspaceBytes)
    XCTAssertThrowsError(try H3OmniVSAPlan(prefixSegments:[.dense(role:.text,rows:1)],targetVideoGrid:[100,20,20]))
    XCTAssertThrowsError(try H3OmniVSAPlan(prefixSegments:Array(repeating:.dense(role:.text,rows:1),count:20_000),targetVideoGrid:[1,1,1]))
    let plan=try H3OmniVSAPlan(prefixSegments:[],targetVideoGrid:[1,1,1])
    let q=MLXArray.zeros([1,1,1,128],dtype:.bfloat16)
    XCTAssertThrowsError(try H3OmniAttention.evaluate(query:q,key:q,value:q,gate:q.asType(.float32),plan:plan))
    let narrow=MLXArray.zeros([1,1,1,64],dtype:.bfloat16)
    XCTAssertThrowsError(try H3OmniAttention.evaluate(query:narrow,key:narrow,value:narrow,gate:narrow,plan:plan))
  }

  func testCancellationStopsBeforeKernelWork() async throws {
    let task=Task { () throws -> Void in
      withUnsafeCurrentTask{$0?.cancel()}
      let plan=try H3OmniVSAPlan(prefixSegments:[],targetVideoGrid:[1,1,1])
      let x=MLXArray.zeros([1,1,1,128],dtype:.bfloat16)
      _ = try H3OmniAttention.evaluate(query:x,key:x,value:x,gate:x,plan:plan)
    }
    do {try await task.value;XCTFail("Cancelled attention executed")}
    catch is CancellationError {}
  }
}
