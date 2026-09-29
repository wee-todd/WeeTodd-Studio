import Foundation
import Metal
import XCTest
import AdapterRuntime
import LTX25Engine
import TensorIO
@testable import LTX25NNC

final class ActiveLoRATests: XCTestCase {
  func testInstalledMixedVersionStackMatchesIndependentWeightOracle() throws {
    let e = ProcessInfo.processInfo.environment
    guard let root = e["WEETODD_LORA_ROOT"],let older = e["WEETODD_LORA_23"],let newer = e["WEETODD_LORA_25"],
      let reference = e["WEETODD_LORA_ORACLE"] else { throw XCTSkip("Installed LoRA math qualification is opt-in") }
    let stack = try LTXAdapterCompatibility.weightStack([
      LoRAAdapter(path: older,strength: 0.3),LoRAAdapter(path: newer,strength: -0.2)])
    XCTAssertEqual(stack.activePairCount,3320)
    XCTAssertLessThanOrEqual(stack.admittedWorkspaceBytes,64*1024*1024)
    let weights = try DenoiserWeights(root: URL(fileURLWithPath: root),
      configuration: AVBlockConfiguration(videoTokens: 8,audioTokens: 3,textTokens: 4),adapters: stack)
    if e["WEETODD_LORA_FULL_BLOCK"] == "1" {
      let start = Date()
      let configuration = try AVBlockConfiguration(videoTokens: 8,audioTokens: 3,textTokens: 4)
      for (name,shape) in try AVBlockRunner.expectedWeightShapes(configuration: configuration).sorted(by: { $0.key < $1.key }) {
        let matrix = try weights.readBlock(0,name: name,shape: shape)
        XCTAssertTrue(FloatValidation.allFinite(matrix))
      }
      print("ACTIVE_LORA_FULL_BLOCK seconds=\(Date().timeIntervalSince(start))")
    }
    let oracle = try SafeTensorFile(url: URL(fileURLWithPath: reference))
    for (name,shape) in [("proj_out.weight",[128,4096]),("audio_proj_out.weight",[128,2048]),
      ("transformer_blocks.0.attn1.to_q.weight",[4096,4096]),
      ("transformer_blocks.0.audio_attn1.to_q.weight",[2048,2048])] {
      try autoreleasepool {
        let start = Date()
        let actual: [Float]
        if name.hasPrefix("transformer_blocks.0.") {
          actual = try weights.readBlock(0,name: String(name.dropFirst("transformer_blocks.0.".count)),shape: shape)
        } else { actual = try weights.readFixed(name,shape: shape) }
        let elapsed = Date().timeIntervalSince(start)
        let expected = try oracle.readFloat32(named: name)
        XCTAssertEqual(actual.count,expected.count)
        var maxError: Double = 0, error: Double = 0, norm: Double = 0
        for i in actual.indices {
          let diff = Double(actual[i])-Double(expected[i]); maxError = max(maxError,abs(diff))
          error += diff*diff; norm += Double(expected[i])*Double(expected[i])
        }
        let relative = sqrt(error/max(norm,1e-30))
        // Frozen before installed execution; Float32 merge arithmetic can use
        // different fused accumulation than the independent Float64 oracle.
        XCTAssertLessThanOrEqual(maxError,0.00002)
        XCTAssertLessThanOrEqual(relative,0.00001)
        print("ACTIVE_LORA_ORACLE \(name) maxabs=\(maxError) relative=\(relative) load_merge_seconds=\(elapsed) workspace=\(stack.admittedWorkspaceBytes)")
      }
    }
  }
}

extension ActiveLoRATests {
  func testInstalledActiveLoRACancellationReleasesModelAndAllowsRestart() throws {
    let e = ProcessInfo.processInfo.environment
    guard e["WEETODD_LORA_CANCEL"] == "1",let root = e["WEETODD_TWO_STAGE_ROOT"],
      let upscaler = e["WEETODD_UPSCALER"],let vae = e["WEETODD_VIDEO_VAE"],
      let context = e["WEETODD_TWO_STAGE_CONTEXT"],let selection = e["WEETODD_TWO_STAGE_LORAS"] else {
      throw XCTSkip("Installed active-adapter cancellation is opt-in")
    }
    let adapters = try JSONDecoder().decode([LoRAAdapter].self,from: Data(contentsOf: URL(fileURLWithPath: selection)))
    XCTAssertTrue(adapters.contains { $0.enabled && $0.strength != 0 })
    func load(_ name: String) throws -> [Float] {
      let data = try Data(contentsOf: URL(fileURLWithPath: context+"-"+name+".f32"))
      return data.withUnsafeBytes { bytes in
        (0..<(bytes.count/4)).map { bytes.loadUnaligned(fromByteOffset: $0*4,as: Float.self) }
      }
    }
    let recipe = try DistilledTwoStageRecipe(width: 512,height: 256,frames: 33,fps: 24,seed: 42)
    let runner = try DistilledSamplingRunner(recipe: recipe,transformerRoot: URL(fileURLWithPath: root),
      upscalerCheckpoint: URL(fileURLWithPath: upscaler),statisticsCheckpoint: URL(fileURLWithPath: vae),loras: adapters)
    let video = try load("video"),audio = try load("audio")
    for attempt in 1...2 {
      var reachedBlock = false
      XCTAssertThrowsError(try runner.evaluate(videoContext: video,audioContext: audio) { event,completed,_ in
        if event == "stage1:transformer" && completed == 1 {
          reachedBlock = true
          XCTAssertGreaterThan(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0,1024*1024*1024)
          throw CancellationError()
        }
      }) { XCTAssertTrue($0 is CancellationError) }
      XCTAssertTrue(reachedBlock,"Restart must reach actual weighted work")
      let released = MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? Int.max
      XCTAssertLessThan(released,64*1024*1024)
      print("ACTIVE_LORA_CANCEL attempt=\(attempt) released_metal_bytes=\(released)")
    }
  }
}
