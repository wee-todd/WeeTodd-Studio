import Foundation
import XCTest
import MLX
@testable import FlashVSRMLX

final class FlashVSRCheckpointTests: XCTestCase {
  func testReleasedPairedIdentitiesAndEveryTensorShape() throws {
    let context="4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef"
    for variant in [FlashVSRVariant.turbo,.base] {
      let pins=FlashVSRCheckpoint.releasedDigests(variant)
      try FlashVSRCheckpoint.validateIdentity(.init(variant:variant,transformerSHA256:pins.transformer,projectionSHA256:pins.projection,contextSHA256:context),expected:variant)
      let opposite=FlashVSRCheckpoint.releasedDigests(variant == .turbo ? .base : .turbo)
      XCTAssertThrowsError(try FlashVSRCheckpoint.validateIdentity(.init(variant:variant,transformerSHA256:pins.transformer,projectionSHA256:opposite.projection,contextSHA256:context),expected:variant))
      XCTAssertThrowsError(try FlashVSRCheckpoint.validateIdentity(.init(variant:variant,transformerSHA256:pins.transformer,projectionSHA256:pins.projection,contextSHA256:String(repeating:"a",count:64)),expected:variant))
    }
    let dit=FlashVSRCheckpoint.expectedShapes(component:.transformer)
    XCTAssertEqual(dit.count,825)
    XCTAssertEqual(dit["blocks.29.ffn.0.weight"],[8960,1536])
    XCTAssertEqual(FlashVSRCheckpoint.expectedShapes(component:.projection).count,8)
    XCTAssertNil(FlashVSRCheckpoint.expectedShapes(component:.projection)["linear_layers.1.weight"])
  }
  func testFP32ComponentMath() throws {
    let x=MLXArray([Float(-2),0,2],[1,3])
    let values=FlashVSRTransformer.gelu(x).asArray(Float.self)
    XCTAssertEqual(values[0],-0.0454023,accuracy:1e-6)
    XCTAssertEqual(values[1],0,accuracy:1e-6)
    XCTAssertEqual(values[2],1.9545977,accuracy:1e-6)
    let norm=FlashVSRTransformer.norm(x).asArray(Float.self)
    XCTAssertEqual(norm[0],-sqrt(Float(1.5)),accuracy:1e-6)
    XCTAssertEqual(norm[2],sqrt(Float(1.5)),accuracy:1e-6)
    let projected=try FlashVSRTransformer.linear(x,"p",["p.weight":MLXArray([Float(1),2,3],[1,3]),"p.bias":MLXArray([Float(4)])])
    XCTAssertEqual(projected.asArray(Float.self),[8])
  }
  func testOptInInstalledCheckpointAdmission() throws {
    guard let directory=ProcessInfo.processInfo.environment["WEETODD_FLASH_MODEL_ROOT"] else { throw XCTSkip("Set WEETODD_FLASH_MODEL_ROOT to inspect installed released assets.") }
    let root=URL(fileURLWithPath:directory)
    let context=try FlashVSRCheckpoint.loadFixedContext(url:root.appendingPathComponent("posi_prompt.pth"))
    XCTAssertEqual(context.shape,[1,512,4096])
    for variant in [FlashVSRVariant.turbo,.base] {
      let folder=root.appendingPathComponent(variant.rawValue)
      let dit=try FlashVSRCheckpoint(url:folder.appendingPathComponent("dit.safetensors"),variant:variant,component:.transformer)
      XCTAssertEqual(try dit.tensor("patch_embedding.weight").shape,[1536,16,1,2,2])
      let projection=try FlashVSRCheckpoint(url:folder.appendingPathComponent("lq_proj_in.safetensors"),variant:variant,component:.projection)
      XCTAssertEqual(try projection.tensor("linear_layers.0.bias").shape,[1536])
    }
  }

}
