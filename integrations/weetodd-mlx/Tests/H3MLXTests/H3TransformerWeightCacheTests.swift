import Foundation
import CryptoKit
import MLX
import XCTest
@testable import H3MLX

final class H3TransformerWeightCacheTests: XCTestCase {
  private let gb = 1024 * 1024 * 1024
  @MainActor
  func testInstalledCancellationClosesRetainedOwner() async throws {
    let env=ProcessInfo.processInfo.environment
    guard let checkpoint=env["WEETODD_H3_CACHE_CHECKPOINT"],
      let library=env["WEETODD_H3_SOL_TEST_METALLIB"],let digest=env["WEETODD_H3_SOL_TEST_METALLIB_SHA256"] else {
      throw XCTSkip("Opt-in cancellation of installed retained H3 block weights")
    }
    let metal=URL(fileURLWithPath:library)
    XCTAssertEqual(SHA256.hash(data:try Data(contentsOf:metal)).map { String(format:"%02x",$0) }.joined(),digest)
    GPU.metallib=metal
    let cache=try H3TransformerWeightCache(checkpointURL:URL(fileURLWithPath:checkpoint),blockCount:1,
      budgetGB:8,adapters:[],useMPP:false,verificationScope:"cache-cancel-test")
    defer { cache.close();Memory.clearCache() }
    let owner=try cache.weights(for:0)
    XCTAssertGreaterThan(cache.storageBytes,0)
    // This actor cannot start the child until the parent yields at value, so
    // cancellation is deterministic and precedes the next cached access.
    let task=Task { @MainActor in _=try cache.weights(for:0) }
    task.cancel()
    do { _=try await task.value;XCTFail("Cancelled cache access must throw") }
    catch is CancellationError { }
    XCTAssertTrue(cache.isClosed)
    XCTAssertTrue(owner.isClosed)
    XCTAssertTrue(cache.report.released)
    XCTAssertEqual(cache.storageBytes,0)
  }
  func testInstalledStateRetiresCacheBeforeCapturingFinalReport() throws {
    let env=ProcessInfo.processInfo.environment
    guard let checkpoint=env["WEETODD_H3_CACHE_CHECKPOINT"],
      let library=env["WEETODD_H3_SOL_TEST_METALLIB"],let digest=env["WEETODD_H3_SOL_TEST_METALLIB_SHA256"] else {
      throw XCTSkip("Opt-in installed state retirement and retained-cache metadata")
    }
    let metal=URL(fileURLWithPath:library)
    XCTAssertEqual(SHA256.hash(data:try Data(contentsOf:metal)).map { String(format:"%02x",$0) }.joined(),digest)
    GPU.metallib=metal
    let geometry=try H3Geometry(width:32,height:32,durationSeconds:2.5)
    let layout=try H3PackedLayout(geometry:geometry,textTags:[1,1,1,1],anchors:[])
    let state=try H3DiTState(checkpointURL:URL(fileURLWithPath:checkpoint),layout:layout,
      textEmbeddings:MLXArray.zeros([1,4,5120],dtype:.bfloat16),timestepTable:[0.1,0.5,1],
      blockCount:1,transformerWeightCacheGB:8)
    defer { state.unload() }
    let result=try state.predict(videoLatents:MLXArray.full([1,geometry.videoRows,96],values:MLXArray(Float(0.01))),
      audioLatents:MLXArray.full([1,geometry.audioRows,32],values:MLXArray(Float(0.01))),
      timestepIndices:Array(repeating:1,count:layout.tags.count))
    let video=result.video.asArray(Float.self),audio=result.audio.asArray(Float.self)
    XCTAssertEqual(state.backendReport.transformerWeightCache?.released,false)
    let report=state.unloadAndReport()
    XCTAssertEqual(report.transformerWeightCache?.released,true)
    XCTAssertEqual(report.transformerWeightCache?.loads,1)
    XCTAssertFalse(state.isResident)
    XCTAssertEqual(state.residentActivationBytes,0)
    XCTAssertEqual(result.video.asArray(Float.self),video)
    XCTAssertEqual(result.audio.asArray(Float.self),audio)
    XCTAssertEqual(state.unloadAndReport(),report)
  }
  func testZeroBudgetKeepsStreamingAndPrefixNeverExceedsBudget() throws {
    for budget in [0,8,16,32,48] {
      let plan = try H3TransformerCachePlan(budgetGB:budget, blockBytes:Array(repeating:770_725_376,count:50),
        physicalBytes:256*gb, recommendedBytes:192*gb, availableBytes:160*gb)
      XCTAssertLessThanOrEqual(plan.retainedBytes,budget*gb)
      XCTAssertEqual(plan.blockCount,budget == 0 ? 0 : min(50,budget*gb/770_725_376))
    }
  }
  func testInsufficientMemoryAndMalformedBudgetRejectWithoutWeights() throws {
    for budget in [-1,1,7,Int.max] {
      XCTAssertThrowsError(try H3TransformerCachePlan(budgetGB:budget,blockBytes:[gb],
        physicalBytes:64*gb,recommendedBytes:48*gb,availableBytes:40*gb))
    }
    XCTAssertThrowsError(try H3TransformerCachePlan(budgetGB:32,blockBytes:Array(repeating:gb,count:50),
      physicalBytes:64*gb,recommendedBytes:48*gb,availableBytes:40*gb))
    XCTAssertThrowsError(try H3TransformerCachePlan(budgetGB:16,blockBytes:Array(repeating:gb,count:50),
      physicalBytes:256*gb,recommendedBytes:192*gb,availableBytes:25*gb))
    for value:Any in [true,8.5,"8",-1,NSNull()] {
      XCTAssertThrowsError(try H3TransformerCachePlan.budget(config:["transformer_weight_cache_gb":value]))
    }
    XCTAssertEqual(try H3TransformerCachePlan.budget(config:[:]),0)
    XCTAssertEqual(try H3TransformerCachePlan.budget(config:["transformer_weight_cache_gb":8]),8)
  }
  func testUntouchedRecipeRejectsAdvancedWrappersAndDeferredAdapters() throws {
    let ordinary:[String:Any]=["config":["transformer_weight_cache_gb":48]]
    XCTAssertEqual(try H3TransformerWeightCachePolicy.admitRecipe(ordinary),48)
    for key in ["fasth3","vdn","continuation","refinement","joint_refinement","joint_latents","motion_fidelity"] {
      var root=ordinary;root[key]=[:]
      XCTAssertThrowsError(try H3TransformerWeightCachePolicy.admitRecipe(root),key)
      root["config"]=["transformer_weight_cache_gb":0]
      XCTAssertEqual(try H3TransformerWeightCachePolicy.admitRecipe(root),0)
    }
    var deferred=ordinary;deferred["loras"]=["adapters":[["start_after_evaluations":1]]]
    XCTAssertThrowsError(try H3TransformerWeightCachePolicy.admitRecipe(deferred))
  }
  func testWindowCanStartAfterCachedPrefixWithoutSkippingStreamedBlocks() throws {
    var plan = try H3PreparedBlockWindow.Plan(blockCount:50,windowSize:1,startIndex:11)
    XCTAssertThrowsError(try plan.begin(0))
    for index in 11..<50 { XCTAssertEqual(try plan.begin(index),index..<index+1);try plan.finish(index) }
    XCTAssertTrue(plan.isClosed)
    XCTAssertThrowsError(try H3PreparedBlockWindow.Plan(blockCount:50,windowSize:2,startIndex:11))
  }
  func testInstalledOwnerReusesWeightsAndProjectionExactlyThenReleases() throws {
    guard let path=ProcessInfo.processInfo.environment["WEETODD_H3_CACHE_CHECKPOINT"] else {
      throw XCTSkip("Opt-in installed weight cache parity and timing check")
    }
    let env=ProcessInfo.processInfo.environment
    guard let library=env["WEETODD_H3_SOL_TEST_METALLIB"],let digest=env["WEETODD_H3_SOL_TEST_METALLIB_SHA256"] else {
      throw H3CheckpointError.invalid("Installed cache test requires a pinned Metal library.")
    }
    let metal=URL(fileURLWithPath:library)
    XCTAssertEqual(SHA256.hash(data:try Data(contentsOf:metal)).map { String(format:"%02x",$0) }.joined(),digest)
    GPU.metallib=metal
    let checkpoint=URL(fileURLWithPath:path)
    let adapters=try env["WEETODD_H3_CACHE_ADAPTER"].map { [try H3LoRAAdapter(url:URL(fileURLWithPath:$0),strength:1,profile:.turbo,qkvLayout:.contiguousQKV)] } ?? []
    let application=try adapters.isEmpty ? nil : H3LoRAStack(adapters:adapters)
    let cache=try H3TransformerWeightCache(checkpointURL:checkpoint,blockCount:1,budgetGB:8,adapters:adapters,useMPP:false,verificationScope:"cache-test")
    let start=Date()
    let first=try cache.weights(for:0)
    let coldSeconds=Date().timeIntervalSince(start)
    let input=MLXArray(Array(repeating:Float(0.02),count:5376),[1,1,5376]).asType(.bfloat16)
    let scope=try first.prepareLoRA(application)
    let target="diffusion_model.blocks.0.attn.qkv_proj"
    let base=try first.project("attn.qkv_proj",activation:input,rows:21504,columns:5376,qkv:true)
    let output=try scope?.apply(base:base,input:input,target:target,reorderQKV:true) ?? base;eval(output)
    let expected=output.asArray(Float.self)
    let bytes=cache.storageBytes
    first.retireProjection("attn.qkv_proj")
    let warm=Date();let second=try cache.weights(for:0)
    let hitSeconds=Date().timeIntervalSince(warm)
    XCTAssertTrue(first === second)
    let secondBase=try second.project("attn.qkv_proj",activation:input,rows:21504,columns:5376,qkv:true)
    let again=try second.prepareLoRA(application)?.apply(base:secondBase,input:input,target:target,reorderQKV:true) ?? secondBase;eval(again)
    XCTAssertEqual(again.asArray(Float.self),expected)
    XCTAssertEqual(cache.storageBytes,bytes)
    XCTAssertEqual(cache.hits,1);XCTAssertEqual(cache.loads,1)
    let stream=try H3PreparedBlock(checkpointURL:checkpoint,index:0,projectionMode:.weightDecoded)
    let streamBase=try stream.project("attn.qkv_proj",activation:input,rows:21504,columns:5376,qkv:true)
    let streamResult=try stream.prepareLoRA(application)?.apply(base:streamBase,input:input,target:target,reorderQKV:true) ?? streamBase;eval(streamResult)
    XCTAssertEqual(streamResult.asArray(Float.self),expected)
    stream.close()
    print("H3_CACHE_COMPONENT cold=\(coldSeconds) hit=\(hitSeconds) retained=\(bytes)")
    XCTAssertThrowsError(try cache.weights(for:1))
    XCTAssertTrue(cache.report.released)
    cache.close();cache.close();XCTAssertEqual(cache.storageBytes,0);XCTAssertTrue(first.isClosed)
    XCTAssertThrowsError(try cache.weights(for:0))
  }
}
