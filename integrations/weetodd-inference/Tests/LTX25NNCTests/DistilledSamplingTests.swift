import Foundation
import AdapterRuntime
import XCTest
import Metal
import LTX25Engine
@testable import LTX25NNC

final class DistilledSamplingTests: XCTestCase {
  func testInstalledTwoStageSamplingAndRelease() throws {
    let e = ProcessInfo.processInfo.environment
    guard let root = e["WEETODD_TWO_STAGE_ROOT"],let upscaler = e["WEETODD_UPSCALER"],
      let vae = e["WEETODD_VIDEO_VAE"],let context = e["WEETODD_TWO_STAGE_CONTEXT"],
      let output = e["WEETODD_TWO_STAGE_OUTPUT"] else { throw XCTSkip("Installed two-stage sampling is opt-in") }
    func load(_ name: String) throws -> [Float] {
      let data = try Data(contentsOf: URL(fileURLWithPath: context+"-"+name+".f32"))
      XCTAssertEqual(data.count % 4,0)
      return data.withUnsafeBytes { bytes in (0..<(bytes.count/4)).map { bytes.loadUnaligned(fromByteOffset: $0*4,as: Float.self) } }
    }
    let recipe = try DistilledTwoStageRecipe(width: 512,height: 256,frames: 33,fps: 24,seed: 42)
    let loras = try e["WEETODD_TWO_STAGE_LORAS"].map {
      try JSONDecoder().decode([LoRAAdapter].self,from: Data(contentsOf: URL(fileURLWithPath: $0)))
    } ?? []
    let runner = try DistilledSamplingRunner(recipe: recipe,transformerRoot: URL(fileURLWithPath: root),
      upscalerCheckpoint: URL(fileURLWithPath: upscaler),statisticsCheckpoint: URL(fileURLWithPath: vae),loras: loras)
    let video = try load("video"), audio = try load("audio")
    XCTAssertThrowsError(try runner.evaluate(videoContext: [],audioContext: audio))
    if e["WEETODD_TWO_STAGE_CANCEL"] == "1" {
      var reachedResidentStageTwo = false
      XCTAssertThrowsError(try runner.evaluate(videoContext: video,audioContext: audio) { event,completed,_ in
        if event == "stage2:transformer" && completed == 1 {
          reachedResidentStageTwo = true
          XCTAssertGreaterThan(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0,1024*1024*1024)
          throw CancellationError()
        }
      })
      XCTAssertTrue(reachedResidentStageTwo)
      XCTAssertLessThan(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? Int.max,64*1024*1024)
      print("TWO_STAGE_CANCEL_RELEASED")
    }
    var checkedReentry = false
    var boundaryBytes: [String:Int] = [:]
    var steps: [Int] = [], releases: [String] = []
    let result = try runner.evaluate(videoContext: video,audioContext: audio) { event,completed,total in
      if event == "sampling" { steps.append(completed); XCTAssertEqual(total,11) }
      if event.hasSuffix("weights_released") {
        releases.append(event)
        boundaryBytes[event] = MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? Int.max
        XCTAssertLessThan(boundaryBytes[event]!,64*1024*1024)
      }
      if !checkedReentry {
        checkedReentry = true
        XCTAssertThrowsError(try runner.evaluate(videoContext: video,audioContext: audio))
      }
      print("TWO_STAGE_PROGRESS \(event) \(completed)/\(total)")
      fflush(stdout)
    }
    XCTAssertEqual(steps,Array(1...11))
    XCTAssertEqual(releases,["stage1_weights_released","upscaler_weights_released","stage2_weights_released"])
    XCTAssertEqual(runner.graphBuilds,[1:1,2:1])
    XCTAssertEqual(result.video.count,recipe.high.videoTokens*128)
    XCTAssertEqual(result.audio.count,recipe.high.audioFrames*128)
    XCTAssertTrue(result.video.allSatisfy(\.isFinite)); XCTAssertTrue(result.audio.allSatisfy(\.isFinite))
    for (name,values) in [("video",result.video),("audio",result.audio)] {
      try values.withUnsafeBytes { try Data($0).write(to: URL(fileURLWithPath: output+"-"+name+".f32")) }
    }
    if let baseline = e["WEETODD_TWO_STAGE_BASELINE"],!loras.isEmpty {
      for (name,actual) in [("video",result.video),("audio",result.audio)] {
        let data = try Data(contentsOf: URL(fileURLWithPath: baseline+"-"+name+".f32"))
        let previous: [Float] = data.withUnsafeBytes { bytes in
          (0..<(bytes.count/4)).map { bytes.loadUnaligned(fromByteOffset: $0*4,as: Float.self) }
        }
        XCTAssertEqual(actual.count,previous.count)
        XCTAssertGreaterThan(zip(actual,previous).reduce(0.0) { $0+abs(Double($1.0)-Double($1.1)) },0.001,
          "Active LoRAs must change both sampled modalities relative to the same base recipe")
      }
    }
    let released = MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? Int.max
    XCTAssertLessThan(released,64*1024*1024)
    let report: [String:Any] = ["active_lora_count":loras.filter { $0.enabled && $0.strength != 0 }.count,"stage_seconds":runner.stageSeconds,"released_metal_bytes":released,
      "width":512,"height":256,"frames":33,"steps":steps,"boundary_metal_bytes":boundaryBytes,"graph_builds":runner.graphBuilds.mapKeysForReport]
    try JSONSerialization.data(withJSONObject: report,options: [.prettyPrinted,.sortedKeys])
      .write(to: URL(fileURLWithPath: output+"-report.json"))
  }
}

private extension Dictionary where Key == Int, Value == Int {
  var mapKeysForReport: [String:Int] { Dictionary<String,Int>(uniqueKeysWithValues: map { (String($0.key),$0.value) }) }
}
