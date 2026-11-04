import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3ReducedFlexTests: XCTestCase {
  private func fixture(qkv: Bool = false, marker: UInt16 = 0x4000,
    changes: [String:String] = [:]) throws -> URL {
    let target="diffusion_model.blocks.0." + (qkv ? "attn.qkv_proj" : "mlp.fc2")
    let columns=qkv ? 5376 : 14336, rows=qkv ? 21504 : 5376
    var a=[UInt16](repeating:0,count:2*columns),b=[UInt16](repeating:0,count:rows*2)
    a[0]=0x3f80;b[(qkv ? 128 : 0)*2]=0x3f80
    var payload=a.withUnsafeBytes {Data($0)};let split=payload.count
    payload.append(b.withUnsafeBytes {Data($0)});let end=payload.count
    var alpha=marker.littleEndian;payload.append(withUnsafeBytes(of:&alpha) {Data($0)})
    let metadata=["converted_layout":"comfyui_minimax_h3","format":"pt","baked_scale":"1.0",
      "base_model":"MiniMaxAI/MiniMax-H3","base_revision":"42ed227ee7df40d41602854ae760620d6eb651fe",
      "modified_by":"Mellis, Inc.","ss_network_dim":"Dynamic","ss_network_alpha":"Dynamic",
      "conversion":"qkv block-diag fused (contiguous q|k|v); mlp.fc1 swiglu halves swapped; alpha folded into lora_B; scale 1.0 baked as a flat multiplier"]
      .merging(changes) {_,new in new}
    let header:[String:Any]=["__metadata__":metadata,
      target+".lora_A.weight":["dtype":"BF16","shape":[2,columns],"data_offsets":[0,split]],
      target+".lora_B.weight":["dtype":"BF16","shape":[rows,2],"data_offsets":[split,end]],
      target+".alpha":["dtype":"BF16","shape":[],"data_offsets":[end,end+2]]]
    let bytes=try JSONSerialization.data(withJSONObject:header);var length=UInt64(bytes.count).littleEndian
    var data=withUnsafeBytes(of:&length) {Data($0)};data.append(bytes);data.append(payload)
    let url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".safetensors")
    try data.write(to:url);addTeardownBlock {try? FileManager.default.removeItem(at:url)};return url
  }

  func testReducedRankMarkersPreserveUnitScaleAndContiguousQKV() throws {
    for qkv in [false,true] {
      let url=try fixture(qkv:qkv)
      let file=try H3LoRAFile(url:url,strength:1,requestedSteps:7,samplingMethod:.pikaFlex,profile:.pikaFlexFL2VA)
      let width=qkv ? 5376 : 14336;var input=[Float](repeating:0,count:width);input[0]=2
      let result=try file.apply(base:MLXArray.ones([1,1,qkv ? 21504 : 5376]),
        input:MLXArray(input,[1,1,width]),target:"diffusion_model.blocks.0."+(qkv ? "attn.qkv_proj" : "mlp.fc2"),reorderQKV:qkv)
      XCTAssertEqual(result[0,0,qkv ? 384 : 0].item(Float.self),3)
      if qkv {XCTAssertEqual(result[0,0,128].item(Float.self),1)}
    }
  }

  func testReducedScaleRejectsConflictingMarkersAndLayout() throws {
    for marker:UInt16 in [0x3f80,0x7fc0] {
      XCTAssertThrowsError(try H3LoRAFile(url:fixture(marker:marker),strength:1,requestedSteps:7,samplingMethod:.pikaFlex,profile:.pikaFlexFL2VA))
    }
    XCTAssertThrowsError(try H3LoRAFile(url:fixture(changes:["baked_scale":"0.125"]),strength:1,requestedSteps:7,samplingMethod:.pikaFlex,profile:.pikaFlexFL2VA))
    XCTAssertThrowsError(try H3LoRAFile(url:fixture(),strength:1,requestedSteps:7,samplingMethod:.pikaFlex,qkvLayout:.nativeInterleaved,profile:.pikaFlexFL2VA))
  }

  func testReducedFormatRejectsContradictoryDeclarations() throws {
    for changes in [["adapter_profile":"turbo"],["profile":"standard"],["distillation_profile":"turbo"],
      ["inference_steps":"4"],["num_inference_steps":"4"],["steps":"4"],["transformer_evaluations":"4"],
      ["schedule_points":"5"],["qkv_layout":"native_interleaved"],["qkv_fusion":"per-head"],["adapter_role":"turbo"]] {
      XCTAssertThrowsError(try H3LoRAFile(url:fixture(changes:changes),strength:1,requestedSteps:7,samplingMethod:.pikaFlex,profile:.pikaFlexFL2VA),"\(changes)")
    }
  }

  func testInstalledKijaiAdaptersAreTaskIdentifiedAndLoadAll208Targets() throws {
    guard let root=ProcessInfo.processInfo.environment["WEETODD_H3_KIJAI_FLEX_MODELS"] else {throw XCTSkip("Optional installed Kijai weights")}
    for (task,rank) in [("fl2va",56),("ref2va",55)] {
      let url=URL(fileURLWithPath:root).appendingPathComponent("pika_h3_flex_\(task)_v1_avg_rank_\(rank)_bf16.safetensors")
      XCTAssertEqual(try H3FlexAdapter.resolvedProfile(url:url,selected:.auto).rawValue,"pika_flex_\(task)")
      let file=try H3LoRAFile(url:url,strength:1,requestedSteps:7,samplingMethod:.pikaFlex)
      XCTAssertEqual(file.targetCount,208)
      XCTAssertThrowsError(try H3LoRAFile(url:url,strength:1,requestedSteps:7,samplingMethod:.euler))
      XCTAssertThrowsError(try H3FlexAdapter.resolvedProfile(url:url,selected:.standard))
    }
  }
}
