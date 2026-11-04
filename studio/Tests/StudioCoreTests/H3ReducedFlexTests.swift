import Foundation
import XCTest
@testable import StudioCore

final class H3ReducedFlexTests:XCTestCase {
  func testPinnedReducedReleaseHeadersIdentifyTasks() {
    XCTAssertEqual(NativeH3FlexAdapter.profile(headerSHA256:"b29a94ff979d024491dc9cb007f322d9c9b11e1963f659d94757cf1418a21a74",fileBytes:559266144),.pikaFlexFL2VA)
    XCTAssertEqual(NativeH3FlexAdapter.profile(headerSHA256:"bdc5bdcd9d3ff2ade9e58d2b171abe676dc2e791465ff3ab0ce5b255cd243689",fileBytes:553202008),.pikaFlexRef2VA)
    XCTAssertNil(NativeH3FlexAdapter.profile(headerSHA256:"b29a94ff979d024491dc9cb007f322d9c9b11e1963f659d94757cf1418a21a74",fileBytes:553202008))
  }
  func testReducedFormatRejectsExtraExecutionDeclarations() {
    let metadata=["converted_layout":"comfyui_minimax_h3","format":"pt","baked_scale":"1.0",
      "base_model":"MiniMaxAI/MiniMax-H3","base_revision":"42ed227ee7df40d41602854ae760620d6eb651fe",
      "modified_by":"Mellis, Inc.","ss_network_dim":"Dynamic","ss_network_alpha":"Dynamic",
      "conversion":"qkv block-diag fused (contiguous q|k|v); mlp.fc1 swiglu halves swapped; alpha folded into lora_B; scale 1.0 baked as a flat multiplier"]
    XCTAssertTrue(NativeH3FlexAdapter.isReducedNativeFormat(metadata))
    for key in ["adapter_profile","profile","distillation_profile","adapter_role","inference_steps","num_inference_steps",
      "steps","transformer_evaluations","schedule_points","qkv_layout","qkv_fusion"] {
      XCTAssertFalse(NativeH3FlexAdapter.isReducedNativeFormat(metadata.merging([key:"unexpected"]) {_,new in new}),key)
    }
  }
  func testInstalledReducedFilesComposeTheSameFlexContract() throws {
    guard let root=ProcessInfo.processInfo.environment["WEETODD_H3_KIJAI_FLEX_MODELS"] else {throw XCTSkip("Optional installed Kijai weights")}
    for (task,rank) in [("fl2va",56),("ref2va",55)] {
      let path=URL(fileURLWithPath:root).appendingPathComponent("pika_h3_flex_\(task)_v1_avg_rank_\(rank)_bf16.safetensors").path
      let info=try NativeLoRAInspection.inspect(URL(fileURLWithPath:path))
      XCTAssertEqual(info["loraProfile"] as? String,"pika_flex_\(task)")
      let result=try NativeH3LoRAComposition.compose(componentPairs:[[path,1.0]],rootStack:nil,attachments:[],assets:[],
        schedulePoints:31,explicitEvaluations:nil,samplingMethod:"euler",task:task)
      XCTAssertEqual(result.schedulePoints,7);XCTAssertEqual(result.samplingMethod,"pika_flex")
      XCTAssertThrowsError(try NativeLoRAInspection.validateH3Sampling(path:path,selectedProfile:"standard",schedulePoints:7))
    }
  }
}
