import Foundation
import XCTest
@testable import StudioCore

final class NativeLTXDiffusionVAETests:XCTestCase {
  private func fixture(wrongStep:Any=1,overlap:Bool=false) throws -> URL {
    let url=FileManager.default.temporaryDirectory.appendingPathComponent("studio-diffvae-\(UUID().uuidString).safetensors")
    addTeardownBlock { try? FileManager.default.removeItem(at:url) }
    let d:[String:Any]=["_class_name":"NADiffusionDecoder","in_channels":128,"out_channels":3,"patch_size":4,"head_dim":64,
      "stage_channels":[2048,1024,512,512,256],"stage_depths":[4,6,4,2,8],"stage_kernels":[[3,7,7],[3,7,7],[3,5,5],[3,5,5],[11,11,11]],
      "stage5_kernel":[11,11,11],"upsamples":[[[1,2,2],2],[[2,1,1],2],[[2,2,2],1],[[2,2,2],2]],"default_num_inference_steps":wrongStep,
      "timestep_scale_multiplier":1000,"resampler_kind":"linear","spatial_padding_mode":"zeros"]
    let config=try JSONSerialization.data(withJSONObject:["vae":["_class_name":"CausalDiffusionVAE","model_output_type":"x0","decoder":d]])
    var header:[String:Any]=["__metadata__":["model_version":"2.5.0","config":String(data:config,encoding:.utf8)!]],offset=0
    for name in NativeLTXDiffusionVAE.shapes.keys.sorted() {
      let shape=NativeLTXDiffusionVAE.shapes[name]!,bytes=shape.reduce(1,*)*2
      header[name]=["dtype":"BF16","shape":shape,"data_offsets":[overlap ? 0:offset,offset+bytes]];offset+=bytes
    }
    let bytes=try JSONSerialization.data(withJSONObject:header,options:.sortedKeys);var length=UInt64(bytes.count).littleEndian
    var data=withUnsafeBytes(of:&length) { Data($0) };data.append(bytes);try data.write(to:url,options:.withoutOverwriting)
    let file=try FileHandle(forWritingTo:url);try file.truncate(atOffset:UInt64(data.count+offset));try file.close()
    return url
  }
  func testBoundedHeaderRejectsBooleanStepOverlappingSpansAndTruncation() throws {
    XCTAssertEqual(try NativeLTXDiffusionVAE.validate(at:fixture()).count,64)
    XCTAssertThrowsError(try NativeLTXDiffusionVAE.validate(at:fixture(wrongStep:true)))
    XCTAssertThrowsError(try NativeLTXDiffusionVAE.validate(at:fixture(overlap:true)))
    let source=try fixture(),file=try FileHandle(forWritingTo:source)
    try file.truncate(atOffset:1024);try file.close()
    XCTAssertThrowsError(try NativeLTXDiffusionVAE.validate(at:source))
  }
  func testManualNilSettingsPreserveConfigurationWithoutCheckpointIO() throws {
    let config:[String:Any]=["seed":43,"diffvae_optimization":"combined","diffvae_query_chunk_size":512]
    let actual=try NativeLTXDiffusionVAE.apply(nil,checkpoint:URL(fileURLWithPath:"/missing-model"),config:config)
    XCTAssertEqual(try JSONSerialization.data(withJSONObject:actual,options:.sortedKeys),try JSONSerialization.data(withJSONObject:config,options:.sortedKeys))
  }
  func testExplicitControlsPropagateAllValuesAndRejectIgnoredTileBeforeModelIO() throws {
    let options=LTX25DiffusionVAESettings(experimentalEnabled:true,optimization:.stage4WidthTiles,queryChunkSize:31,contextWidthChunks:3,stage4TileWidth:7)
    let result=try NativeLTXDiffusionVAE.apply(options,checkpoint:fixture(),config:["seed":43])
    XCTAssertEqual(result["diffvae_optimization"] as? String,"stage4_width_tiles")
    XCTAssertEqual(result["diffvae_query_chunk_size"] as? Int,31)
    XCTAssertEqual(result["diffvae_context_width_chunks"] as? Int,3)
    XCTAssertEqual(result["diffvae_stage4_tile_width"] as? Int,7)
    var invalid=options;invalid.optimization = .combined
    XCTAssertThrowsError(try NativeLTXDiffusionVAE.apply(invalid,checkpoint:URL(fileURLWithPath:"/missing-model"),config:[:])) { error in
      XCTAssertTrue(String(describing:error).contains("query/context/width"))
    }
    var disabled=options;disabled.experimentalEnabled=false
    XCTAssertThrowsError(try NativeLTXDiffusionVAE.apply(disabled,checkpoint:fixture(),config:[:]))
  }
  func testSelectionPersistenceResetAndLegacyAbsence() throws {
    var selection=GenerationSelection(task:"fflf")
    let encoder=JSONEncoder();encoder.outputFormatting = .sortedKeys
    let legacy=try encoder.encode(selection)
    XCTAssertNil((try JSONSerialization.jsonObject(with:legacy) as? [String:Any])?["ltx25DiffusionVAE"])
    selection.ltx25DiffusionVAE=LTX25DiffusionVAESettings(experimentalEnabled:true,optimization:.deferredStage4,queryChunkSize:113,contextWidthChunks:2)
    let restored=try JSONDecoder().decode(GenerationSelection.self,from:JSONEncoder().encode(selection))
    XCTAssertEqual(restored.ltx25DiffusionVAE,selection.ltx25DiffusionVAE);XCTAssertTrue(restored.isModified)
    selection.resetOverrides();XCTAssertNil(selection.ltx25DiffusionVAE)
    XCTAssertEqual(try encoder.encode(selection),legacy)
  }
  func testInstalledCompatibleDiffusionHeaderWhenProvided() throws {
    guard let path=ProcessInfo.processInfo.environment["WEETODD_LTX_DIFFVAE_CHECKPOINT"] else { throw XCTSkip("Optional header-only installed DiffVAE") }
    XCTAssertEqual(try NativeLTXDiffusionVAE.validate(at:URL(fileURLWithPath:path)).count,64)
  }
}

extension NativeLTXDiffusionVAETests {
  func testDedicatedProducersDoNotSilentlyIgnoreCustomProfileControls() throws {
    XCTAssertTrue(NativeLTXDiffusionVAE.profileControls(["diffvae_optimization":"combined","diffvae_query_chunk_size":512,"diffvae_context_width_chunks":4,"diffvae_stage4_tile_width":0]).isEmpty)
    let custom=NativeLTXDiffusionVAE.profileControls(["diffvae_query_chunk_size":113])
    XCTAssertThrowsError(try NativeLTXDiffusionVAE.resolvedWire(nil,profile:custom,checkpoint:URL(fileURLWithPath:"/missing"))) { error in
      XCTAssertTrue(String(describing:error).contains("cannot ignore custom profile"))
    }
    XCTAssertNil(try NativeLTXDiffusionVAE.resolvedWire(nil,profile:[:],checkpoint:URL(fileURLWithPath:"/missing")))
  }
}
