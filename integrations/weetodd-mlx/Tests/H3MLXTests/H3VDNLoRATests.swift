import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3VDNLoRATests:XCTestCase {
  func testSplitQKVUsesHeadMajorRowsAndSwiGLUUsesGateThenValue() throws {
    let q=MLXArray([Float(1),2],[1,1,2])
    let k=MLXArray([Float(3),4],[1,1,2])
    let v=MLXArray([Float(5),6],[1,1,2])
    let fused=try H3VDNLoRAFile.fuseQKV([q,k,v],heads:2,headWidth:1)
    XCTAssertEqual(fused.asArray(Float.self),[1,3,5,2,4,6])
    let feed=try H3VDNLoRAFile.swapSwiGLU(MLXArray([Float(1),2,3,4],[1,4]))
    XCTAssertEqual(feed.asArray(Float.self),[3,4,1,2])
    XCTAssertThrowsError(try H3VDNLoRAFile.fuseQKV([q,k],heads:2,headWidth:1))
    XCTAssertThrowsError(try H3VDNLoRAFile.swapSwiGLU(MLXArray.ones([1,3])))
  }

  func testInstalledNamedAdaptersAdmitEveryProjectionIncludingModulation() throws {
    guard let value=ProcessInfo.processInfo.environment["WEETODD_H3_VDN_STAGE"] else {
      throw XCTSkip("Opt-in installed VDN PEFT adapter admission")
    }
    let root=URL(fileURLWithPath:value).appendingPathComponent("adapters")
    let original=try H3VDNLoRAFile(directory:root.appendingPathComponent("default"),kind:.standard)
    let turbo=try H3VDNLoRAFile(directory:root.appendingPathComponent("turbo"),kind:.turbo)
    XCTAssertEqual(original.targetCount,208)
    XCTAssertEqual(turbo.targetCount,363)
    XCTAssertEqual(turbo.modulationTargetCount,51)
    let time=MLXArray.ones([1,2688],dtype:.bfloat16)*Float(0.02)
    let modulation=try turbo.apply(base:MLXArray.zeros([1,96768],dtype:.bfloat16),
      input:time,target:"diffusion_model.blocks.49.adaln_proj.linear")
    XCTAssertTrue(MLX.isFinite(modulation).all().item(Bool.self))
    XCTAssertGreaterThan(max(abs(modulation.asType(.float32))).item(Float.self),0)
    XCTAssertThrowsError(try turbo.apply(base:MLXArray.zeros([1,96768],dtype:.bfloat16),
      input:MLXArray.zeros([1,64],dtype:.float32),target:"diffusion_model.blocks.49.adaln_proj.linear"))
  }

  func testInstalledProjectionMatchesExportedNamedPEFTReference() throws {
    let env=ProcessInfo.processInfo.environment
    guard let stage=env["WEETODD_H3_VDN_STAGE"],let fixtures=env["WEETODD_H3_VDN_ORACLE"] else {
      throw XCTSkip("Opt-in installed PEFT numerical reference")
    }
    for kind in [H3VDNLoRAFile.Kind.standard,.turbo] {
      let name=kind == .standard ? "default" : "turbo"
      let adapter=try H3VDNLoRAFile(directory:URL(fileURLWithPath:stage)
        .appendingPathComponent("adapters/"+name),kind:kind)
      let arrays=try loadArrays(url:URL(fileURLWithPath:fixtures).appendingPathComponent("lora-"+name+".safetensors"))
      let targets=["attn.qkv_proj","attn.out_proj"]+(kind == .turbo ? ["mlp.fc1","adaln_proj.linear"] : [])
      for target in targets {
        let expected=try XCTUnwrap(arrays[target+".expected"])
        let result=try adapter.apply(base:MLXArray.zeros(expected.shape,dtype:.bfloat16),
          input:try XCTUnwrap(arrays[target+".input"]),
          target:"diffusion_model.blocks.49."+target,reorderQKV:target == "attn.qkv_proj")
        XCTAssertEqual(result.shape,expected.shape)
        XCTAssertEqual(max(abs(result.asType(.float32)-expected.asType(.float32))).item(Float.self),0)
      }
    }
  }
}
