import Foundation
import CryptoKit
import MLX
import MLXRandom
import XCTest
@testable import H3MLX

final class H3SolAttendedOverrideTests: XCTestCase {
  func testAttendedOverrideAdmitsOrdinaryFLAndRetainsVariantLimits() throws {
    XCTAssertNoThrow(try H3TransformerBlock.validateAttendedOverrideLayout(fastVariant:nil,curveRank:nil))
    XCTAssertNoThrow(try H3TransformerBlock.validateAttendedOverrideLayout(fastVariant:nil,curveRank:64))
    XCTAssertThrowsError(try H3TransformerBlock.validateAttendedOverrideLayout(fastVariant:nil,curveRank:128))
    XCTAssertThrowsError(try H3TransformerBlock.validateAttendedOverrideLayout(fastVariant:.vsaV1,curveRank:nil))
  }
  func testInstalledFLWeightedBlockExecutesExactAndSolHooksBeforeFullRender() throws {
    let env=ProcessInfo.processInfo.environment
    guard let checkpoint=env["WEETODD_H3_CACHE_CHECKPOINT"],let adapter=env["WEETODD_H3_CACHE_ADAPTER"],
      let library=env["WEETODD_H3_SOL_TEST_METALLIB"],let pinned=env["WEETODD_H3_SOL_TEST_METALLIB_SHA256"] else {
      throw XCTSkip("Opt-in installed FL weighted attention path, not just its header admission")
    }
    let metal=URL(fileURLWithPath:library)
    XCTAssertEqual(SHA256.hash(data:try Data(contentsOf:metal)).map { String(format:"%02x",$0) }.joined(),pinned)
    GPU.metallib=metal
    let url=URL(fileURLWithPath:checkpoint)
    let weights=try H3PreparedBlock(checkpointURL:url,index:2,projectionMode:.weightDecoded,retainProjections:true)
    defer { weights.close();Memory.clearCache() }
    XCTAssertEqual(weights.layout.curveRank,64)
    let stack=try H3LoRAStack(adapters:[H3LoRAAdapter(url:URL(fileURLWithPath:adapter),strength:1,profile:.turbo,qkvLayout:.contiguousQKV)])
    let rows=Int(env["WEETODD_H3_SOL_FL_ROWS"] ?? "257")!
    let geometry=try H3Geometry(width:1344,height:768,durationSeconds:5)
    let generatedStart=rows == 38937 ? rows - geometry.videoRows : 65
    let sol=try H3SolGeometry(rows:rows,heads:56,approximationRange:generatedStart..<rows)
    let input=(MLXRandom.normal([1,rows,5376],key:MLXRandom.key(299))*Float(0.01)).asType(.bfloat16)
    let modulation=(MLXRandom.normal([1,96768],key:MLXRandom.key(300))*Float(0.05)).asType(.bfloat16)
    let positions=MLXArray.zeros([rows,3],dtype:.float32)
    let indices=MLXArray.zeros([rows],dtype:.int32)
    let angles=H3RotaryAngles(rows:rows,cosine:MLXArray.ones([1,1,rows,96],dtype:.bfloat16),sine:MLXArray.zeros([1,1,rows,96],dtype:.bfloat16))
    var consumers=0
    func run(_ mode:Int) throws -> [UInt16] {
      let hook:((MLXArray,MLXArray,MLXArray) throws -> MLXArray)?=mode == 0 ? nil : { q,k,v in
        consumers += 1
        if mode == 1 { return MLXFast.scaledDotProductAttention(queries:q,keys:k,values:v,scale:1/Float(128).squareRoot(),mask:nil) }
        let routes=try H3SolRouting.prepare(query:q,key:k,value:v,geometry:sol)
        return try H3SolIndexedAttention.evaluate(query:q,key:k,value:v,prepared:routes)
      }
      let output=try H3TransformerBlock.evaluate(checkpointURL:url,index:2,input:input,modulation:modulation,
        modulationIndices:indices,positions:positions,lora:stack,rotaryAngles:angles,preparedWeights:weights,
        attendedOverride:hook,observe:{ _,_ in })
      eval(output)
      XCTAssertTrue(all(isFinite(output)).item(Bool.self))
      return output.view(dtype:.uint16).asArray(UInt16.self)
    }
    let dense=try run(0)
    XCTAssertEqual(try run(1),dense)
    let sparse=try run(2)
    XCTAssertEqual(try run(2),sparse)
    XCTAssertEqual(consumers,3)
    XCTAssertFalse(weights.isClosed)
    print("FL_WEIGHTED_SOL_SMOKE rows=\(rows) actualSolConsumers=2 exactHookPreservesDense=true repeatable=true")
  }
  func testAttendedOnlyHookPreservesDenseWordsAndProjectionRetirement() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["WEETODD_H3_SOL_CONSUMER_TEST"] == "1" else {
      throw XCTSkip("Opt in to the pinned Sol integration GPU test.")
    }
    guard let path = env["WEETODD_H3_SOL_TEST_METALLIB"], path.hasPrefix("/"),
      let expected = env["WEETODD_H3_SOL_TEST_METALLIB_SHA256"] else {
      throw H3CheckpointError.invalid("Sol integration requires an absolute pinned Metal library.")
    }
    let library = URL(fileURLWithPath:path)
    guard try library.resolvingSymlinksInPath().resourceValues(forKeys:[.isRegularFileKey]).isRegularFile == true,
      FileManager.default.isReadableFile(atPath:path) else {
      throw H3CheckpointError.invalid("Sol integration Metal library must be readable and regular.")
    }
    let digest = SHA256.hash(data:try Data(contentsOf:library)).map { String(format:"%02x",$0) }.joined()
    guard digest == expected, digest == "01be62d9367e145af21fca136a27fdb387cc26eb28ede8beb6c4587cff1e9bb9" else {
      throw H3CheckpointError.invalid("Changed Sol integration test Metal library.")
    }
    GPU.metallib = library
    let rows = 7, width = 4
    let x = MLXRandom.normal([1,rows,width], key:MLXRandom.key(963)).asType(.bfloat16)
    let mod = (MLXRandom.normal([1,18*width],key:MLXRandom.key(964))*Float(0.1)).asType(.bfloat16)
    let indices = MLXArray((0..<rows).map { Int32($0 % 3) })
    let angles = H3RotaryAngles(rows:rows,
      cosine:MLXArray.ones([1,1,rows,4],dtype:.bfloat16),
      sine:MLXArray.zeros([1,1,rows,4],dtype:.bfloat16))
    let weights = Dictionary(uniqueKeysWithValues:[("attn.qkv_proj",24,4),("attn.out_proj",4,8),
      ("mlp.fc1",12,4),("mlp.fc2",4,6)].enumerated().map { i,p in
        (p.0,(MLXRandom.normal([p.1,p.2],key:MLXRandom.key(UInt64(965+i)))*Float(0.1)).asType(.bfloat16))
      })
    func run(override:Bool, wrongShape:Bool = false) throws -> (MLXArray,[String]) {
      var order:[String] = []
      let hook:((MLXArray,MLXArray,MLXArray) throws -> MLXArray)? = override ? { q,k,v in
        order.append("override")
        XCTAssertEqual(v.strides[2],4) // compact V, not a view holding the three-way QKV
        if wrongShape { return MLXArray.zeros([1,1,1,1],dtype:.bfloat16) }
        return MLXFast.scaledDotProductAttention(queries:q,keys:k,values:v,
          scale:0.5,mask:nil)
      }:nil
      let output = try H3TransformerBlock.evaluateKernel(input:x,modulation:mod,
        modulationIndices:indices,angles:angles,hiddenWidth:4,heads:2,headWidth:4,
        feedWidth:6,rotaryWidth:4,read:{ _,shape in MLXArray.ones(shape,dtype:.bfloat16) },
        project:{ a,name,_,_,_ in order.append(name);return matmul(a,weights[name]!.T) },
        attendedOverride:hook,feedRowChunk:3,drainAttentionInputs:true,
        retireQKV:{ order.append("retire_qkv") },retireAttention:{ order.append("retire_attention") })
      return (output,order)
    }
    let reference = try run(override:false).0.view(dtype:.uint16).asArray(UInt16.self)
    let (candidate,order) = try run(override:true)
    XCTAssertEqual(candidate.view(dtype:.uint16).asArray(UInt16.self),reference)
    for pair in [("attn.qkv_proj","retire_qkv"),("retire_qkv","override"),
      ("override","attn.out_proj"),("attn.out_proj","retire_attention"),("retire_attention","mlp.fc1")] {
      XCTAssertLessThan(try XCTUnwrap(order.firstIndex(of:pair.0)),try XCTUnwrap(order.firstIndex(of:pair.1)))
    }
    XCTAssertThrowsError(try run(override:true,wrongShape:true))
  }
}
