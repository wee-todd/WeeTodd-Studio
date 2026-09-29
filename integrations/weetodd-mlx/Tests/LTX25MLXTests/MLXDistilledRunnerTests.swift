import XCTest
import Foundation
import Darwin
import MLX
import LTX25Engine
import LTX25MLX
import TensorIO

final class MLXDistilledRunnerTests:XCTestCase {
  func testOversizedGeometryFailsBeforeCheckpointAccess() throws {
    let recipe=try DistilledTwoStageRecipe(width:2048,height:1024,frames:129,fps:24,seed:1)
    let missing=URL(fileURLWithPath:"/missing/checkpoint")
    XCTAssertThrowsError(try MLXDistilledSamplingRunner(recipe:recipe,transformerRoot:missing,
      upscalerCheckpoint:missing,statisticsCheckpoint:missing)) { error in
      XCTAssertTrue(String(describing:error).contains("activation"))
    }
  }
  func testInstalledTwoStages() throws {
    let env=ProcessInfo.processInfo.environment
    guard let path=env["WEETODD_MLX_TWO_STAGE_REQUEST"], let text=env["WEETODD_MLX_TWO_STAGE_TEXT"] else {
      throw XCTSkip("Installed two-stage qualification is opt-in.")
    }
    let request=try MLXDistilledRequest.load(URL(fileURLWithPath:path)), recipe=try request.recipe()
    let runner=try MLXDistilledSamplingRunner(recipe:recipe,transformerRoot:URL(fileURLWithPath:request.transformerRoot),
      upscalerCheckpoint:URL(fileURLWithPath:request.spatialUpscalerCheckpoint),statisticsCheckpoint:URL(fileURLWithPath:request.videoCheckpoint),
      stageOneLoras:request.stageOneLoras,stageTwoLoras:request.stageTwoLoras)
    let file=try SafeTensorFile(url:URL(fileURLWithPath:text))
    let video=try MLXWeight.read(file,"video").reshaped([1024,4096])
    let audio=try MLXWeight.read(file,"audio").reshaped([1024,2048])
    var stages:[String]=[], steps:[Int]=[], reentry=false
    Memory.peakMemory=0; let started=Date()
    let result=try runner.evaluate(videoContext:video,audioContext:audio) { stage,completed,total in
      if !reentry {
        reentry=true
        XCTAssertThrowsError(try runner.evaluate(videoContext:video,audioContext:audio))
      }
      if stage == "sampling" { steps.append(completed); XCTAssertEqual(total,11) }
      if stage.hasSuffix("weights_released") { stages.append(stage) }
      if stage == "sampling" || stage.hasSuffix("weights_released") {
        print("TWO_STAGE_PROGRESS \(stage) \(completed)/\(total)")
      }
    }
    XCTAssertEqual(steps,Array(1...11))
    XCTAssertEqual(stages,["stage1_weights_released","upscaler_weights_released","stage2_weights_released"])
    XCTAssertEqual(result["video"]!.shape,[recipe.high.videoTokens,128])
    XCTAssertEqual(result["audio"]!.shape,[recipe.high.audioFrames,128])
    for value in result.values { XCTAssertTrue(MLX.isFinite(value).all().item(Bool.self)) }
    var info=task_vm_info_data_t(), count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
    let status=withUnsafeMutablePointer(to:&info) { p in p.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
      task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
    } }
    XCTAssertEqual(status,KERN_SUCCESS)
    print("TWO_STAGE_METRICS seconds=\(Date().timeIntervalSince(started)) peak_mlx=\(Memory.peakMemory) peak_footprint=\(info.ledger_phys_footprint_peak) stages=\(runner.stageSeconds)")
    if let prefix=env["WEETODD_MLX_TWO_STAGE_OUTPUT"] {
      for (name,x) in result {
        try x.asArray(Float.self).withUnsafeBytes { try Data($0).write(to:URL(fileURLWithPath:prefix+"-"+name+".f32")) }
      }
    }
  }
}
