import Foundation
import CryptoKit
import MLX
import MLXRandom
import XCTest
@testable import H3MLX

/// Use the production block kernel, installed projections and prepared Turbo
/// adapter to qualify temporary-buffer reuse before any additional full render.
final class H3PreparedActivationReuseTests: XCTestCase {
  func testInstalledFLMPPProductionBlockMatchesAndMeasures() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["WEETODD_H3_FL_MPP_TEST"] == "1", H3MPPProjection.isAvailable,
      let checkpoint = env["WEETODD_H3_CACHE_CHECKPOINT"],
      let adapter = env["WEETODD_H3_CACHE_ADAPTER"],
      let library = env["WEETODD_H3_SOL_TEST_METALLIB"],
      let digest = env["WEETODD_H3_SOL_TEST_METALLIB_SHA256"] else {
      throw XCTSkip("Opt-in installed full-geometry FL MPP execution")
    }
    let metal = URL(fileURLWithPath: library)
    XCTAssertEqual(SHA256.hash(data: try Data(contentsOf: metal)).map { String(format: "%02x", $0) }.joined(), digest)
    GPU.metallib = metal
    let url = URL(fileURLWithPath: checkpoint), scope = UUID().uuidString
    let ordinary = try H3PreparedBlock(checkpointURL: url, index: 2,
      projectionMode: .weightDecoded, retainProjections: true)
    let accelerated = try H3PreparedBlock(checkpointURL: url, index: 2,
      projectionMode: .weightDecoded, useMPP: true, verificationScope: scope, retainProjections: true)
    let stack = try H3LoRAStack(adapters: [H3LoRAAdapter(url: URL(fileURLWithPath: adapter),
      strength: 1, profile: .turbo, qkvLayout: .contiguousQKV)])
    defer { ordinary.close(); accelerated.close(); H3MPPProjection.forget(scope: scope); Memory.clearCache() }
    let rows = 38937
    let input = (MLXRandom.normal([1, rows, 5376], key: MLXRandom.key(299)) * Float(0.01)).asType(.bfloat16)
    let modulation = (MLXRandom.normal([1, 96768], key: MLXRandom.key(300)) * Float(0.05)).asType(.bfloat16)
    let indices = MLXArray.zeros([rows], dtype: .int32)
    let positions = MLXArray.zeros([rows, 3], dtype: .float32)
    let angles = H3RotaryAngles(rows: rows,
      cosine: MLXArray.ones([1, 1, rows, 96], dtype: .bfloat16),
      sine: MLXArray.zeros([1, 1, rows, 96], dtype: .bfloat16))
    func run(_ owner: H3PreparedBlock) throws -> (MLXArray, Double) {
      Stream.gpu.synchronize()
      let start = Date()
      let value = try H3TransformerBlock.evaluate(checkpointURL: url, index: 2,
        input: input, modulation: modulation, modulationIndices: indices, positions: positions,
        lora: stack, rotaryAngles: angles, preparedWeights: owner, observe: { _, _ in })
      eval(value)
      return (value, Date().timeIntervalSince(start))
    }
    let gold = try run(ordinary).0
    _ = try run(accelerated) // First-use numerical qualification is not a warm timing.
    var normalTimes: [Double] = [], acceleratedTimes: [Double] = []
    for useAccelerated in [false, true, true, false] {
      Memory.peakMemory = Memory.activeMemory
      let (value, elapsed) = try run(useAccelerated ? accelerated : ordinary)
      XCTAssertTrue(H3MPPProjection.firstUseMatches(reference: gold, candidate: value))
      XCTAssertTrue(all(isFinite(value)).item(Bool.self))
      if useAccelerated { acceleratedTimes.append(elapsed) } else { normalTimes.append(elapsed) }
    }
    let stats = H3MPPProjection.verificationStatus(scope: scope)
    print("H3_FL_MPP \(normalTimes) accelerated=\(acceleratedTimes) verified=\(stats.verified) rejected=\(stats.fallback) calls=\(stats.mppCalls) peak=\(Memory.peakMemory)")
    XCTAssertGreaterThan(stats.mppCalls, 0, "Timing must exercise accelerated execution, not only fallback")
  }
  func testInstalledFLActivationReuseCostAndExactOutput() throws {
    let env=ProcessInfo.processInfo.environment
    guard env["WEETODD_H3_ACTIVATION_REUSE_TEST"] == "1",
      let checkpoint=env["WEETODD_H3_CACHE_CHECKPOINT"],let adapter=env["WEETODD_H3_CACHE_ADAPTER"],
      let library=env["WEETODD_H3_SOL_TEST_METALLIB"],let digest=env["WEETODD_H3_SOL_TEST_METALLIB_SHA256"] else {
      throw XCTSkip("Opt-in real-sized installed FL block activation reuse benchmark")
    }
    let metal=URL(fileURLWithPath:library)
    XCTAssertEqual(SHA256.hash(data:try Data(contentsOf:metal)).map { String(format:"%02x",$0) }.joined(),digest)
    GPU.metallib=metal
    let url=URL(fileURLWithPath:checkpoint)
    let weights=try H3PreparedBlock(checkpointURL:url,index:2,projectionMode:.weightDecoded,retainProjections:true)
    let stack=try H3LoRAStack(adapters:[H3LoRAAdapter(url:URL(fileURLWithPath:adapter),strength:1,profile:.turbo,qkvLayout:.contiguousQKV)])
    let application=try XCTUnwrap(weights.prepareLoRA(stack))
    let previous=Memory.cacheLimit
    defer { weights.close();Memory.clearCache();Memory.cacheLimit=previous }
    let rows=38937
    let input=(MLXRandom.normal([1,rows,5376],key:MLXRandom.key(299))*Float(0.01)).asType(.bfloat16)
    let modulation=(MLXRandom.normal([1,96768],key:MLXRandom.key(300))*Float(0.05)).asType(.bfloat16)
    let indices=MLXArray.zeros([rows],dtype:.int32)
    let positions=MLXArray.zeros([rows,3],dtype:.float32)
    let angles=H3RotaryAngles(rows:rows,cosine:MLXArray.ones([1,1,rows,96],dtype:.bfloat16),sine:MLXArray.zeros([1,1,rows,96],dtype:.bfloat16))
    let gold=try H3TransformerBlock.evaluate(checkpointURL:url,index:2,input:input,modulation:modulation,
      modulationIndices:indices,positions:positions,lora:stack,rotaryAngles:angles,preparedWeights:weights,observe:{ _,_ in })
    eval(gold)
    func run(chunk:Int,drain:Bool) throws -> (MLXArray,Double) {
      let start=Date()
      let output=try H3TransformerBlock.evaluateKernel(input:input,modulation:modulation,
        modulationIndices:indices,angles:angles,
        read:{ try weights.read(weights.layout.prefix+"blocks.2."+$0,shape:$1) },
        project:{ x,name,r,c,qkv in
          let base=try weights.project(name,activation:x,rows:r,columns:c,qkv:qkv)
          let result=try application.applyQueued(base:base,input:x,target:"diffusion_model.blocks.2."+name,reorderQKV:qkv)
          asyncEval(result);return result
        },feedRowChunk:chunk,drainAttentionInputs:drain)
      eval(output)
      return (output,Date().timeIntervalSince(start))
    }
    var reports:[[String:Any]]=[]
    let cases:[(String,Int,Int,Bool)]=[("baseline",128,8192,true),("reuse_2g",2048,8192,true),
      ("reuse_2g_16k",2048,16384,true),("reuse_2g_16k_no_qkv_fences",2048,16384,false)]
    for (name,mib,chunk,drain) in cases {
      Stream.gpu.synchronize();Memory.clearCache();Memory.cacheLimit=mib*1024*1024
      _=try run(chunk:chunk,drain:drain)
      var timings:[Double]=[];var differences:[Float]=[]
      Memory.peakMemory=Memory.activeMemory
      for _ in 0..<2 {
        let (value,seconds)=try run(chunk:chunk,drain:drain)
        timings.append(seconds)
        differences.append(max(abs(value.asType(.float32)-gold.asType(.float32))).item(Float.self))
        XCTAssertTrue(all(isFinite(value)).item(Bool.self))
      }
      reports.append(["name":name,"cacheMiB":mib,"feedRows":chunk,"drainQKV":drain,
        "seconds":timings,"maximumAbsoluteDifference":differences,"peakMLXBytes":Memory.peakMemory])
    }
    print("H3_ACTIVATION_REUSE "+String(data:try JSONSerialization.data(withJSONObject:reports,options:[.sortedKeys]),encoding:.utf8)!)
    XCTAssertEqual(reports[0]["maximumAbsoluteDifference"] as? [Float],[0,0],"Diagnostic must reproduce the actual production block exactly")
  }
}
