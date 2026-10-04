import Foundation
import CryptoKit
import XCTest
import MLX
import TensorIO
import LTX25Engine
import InferenceTestSupport
@testable import LTX25MLX

final class MLXTransformerPageConverterTests:XCTestCase {
  private let stem="model.diffusion_model."
  private func withTiny(_ body:(URL,MLXTransformerPageConverter.Plan) throws -> Void) throws {
    let matrix=(0..<128).map { Float($0<64 ? ($0%2==0 ? 0 : 255) : ($0%2==0 ? -255 : 0)) }
    let raw=matrix.withUnsafeBytes { Data($0) }
    let fixed=Data(repeating:0x42,count:1024)
    var tensors:[(String,[Int],String)]=[(stem+"patchify_proj.weight",[4,64],"F32")]
    var payloads=[stem+"patchify_proj.weight":fixed]
    for index in 0..<2 {
      let prefix=stem+"transformer_blocks.\(index)."
      tensors += [(prefix+"attn1.to_q.weight",[2,64],"F32"),(prefix+"attn1.to_q.bias",[2],"F32")]
      payloads[prefix+"attn1.to_q.weight"]=raw
      payloads[prefix+"attn1.to_q.bias"]=Data(repeating:0,count:8)
    }
    try withTensorFile(metadata:["model_version":"2.5.0"],tensors:tensors,payloads:payloads) { url in
      let file=try SafeTensorFile(url:url)
      let plan=try MLXTransformerPageConverter.plan(source:url,file:file,metadata:["model_version":"2.5.0"],blockCount:2,
        blockShapes:["attn1.to_q.weight":[2,64],"attn1.to_q.bias":[2]],fixedShapes:["patchify_proj.weight":[4,64]],
        maximumTensorBytes:512*1024*1024,maximumWorkingBytes:2*1024*1024*1024)
      try body(url,plan)
    }
  }
  private func withDestination(_ body:(URL) throws -> Void) throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    try body(root.appendingPathComponent("paged"))
  }
  func testKnownAffineQ8CPUConversionAndFixedBytesRemainUnchanged() throws {
    try Device.withDefaultDevice(.cpu) {
      try withTiny { source,plan in
        try withDestination { destination in
          var stages:[String]=[]
          let result=try MLXTransformerPageConverter.convert(plan,destination:destination) { stages.append($0.stage) }
          XCTAssertEqual(result.path,(try MLXTransformerPageConverter.canonicalLocalURL(destination)).path)
          let raw=try SafeTensorFile(url:source),fixed=try SafeTensorFile(url:destination.appendingPathComponent("pages/fixed.safetensors"))
          XCTAssertEqual(fixed.tensors[stem+"patchify_proj.weight"]?.dtype,"F32")
          XCTAssertEqual(try raw.withTensorBytes(named:stem+"patchify_proj.weight") { Data($0) },
            try fixed.withTensorBytes(named:stem+"patchify_proj.weight") { Data($0) })
          for index in 0..<2 {
            let page=try SafeTensorFile(url:destination.appendingPathComponent(String(format:"pages/layer-%03d.safetensors",index)))
            let name=stem+"transformer_blocks.\(index).attn1.to_q"
            let packed=try MLXWeight.read(page,name+".weight").asArray(UInt32.self)
            // Independent exact endpoints: scale=-1, bias=255 maps alternating
            // 0/255 to byte codes255/0, little-endian four codes per UInt32.
            XCTAssertEqual(packed,[UInt32](repeating:0x00FF00FF,count:16)+[UInt32](repeating:0xFF00FF00,count:16))
            XCTAssertEqual(try MLXWeight.read(page,name+".scales").asArray(Float.self),[-1,1])
            XCTAssertEqual(try MLXWeight.read(page,name+".biases").asArray(Float.self),[255,-255])
            XCTAssertEqual(page.tensors[name+".weight"]?.shape,[2,16])
          }
          let manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:destination.appendingPathComponent("paged_manifest.json"))) as? [String:Any])
          XCTAssertEqual(manifest["format"] as? String,"weetodd-ltx25-transformer-paged-q8-v1")
          XCTAssertEqual(manifest["num_layers"] as? Int,2)
          XCTAssertEqual(manifest["output_tensor_bytes"] as? UInt64,plan.outputTensorBytes)
          let provenance=try XCTUnwrap(manifest["conversion_provenance"] as? [String:Any])
          let hash=SHA256.hash(data:try Data(contentsOf:source)).map { String(format:"%02x",$0) }.joined()
          XCTAssertEqual(provenance["source_sha256"] as? String,hash)
          XCTAssertEqual(stages.filter { $0=="page_complete" }.count,3)
          XCTAssertThrowsError(try MLXTransformerPageConverter.convert(plan,destination:destination))
          let parent=destination.deletingLastPathComponent(),alias=parent.appendingPathComponent("alias")
          try FileManager.default.createSymbolicLink(at:alias,withDestinationURL:parent)
          let nested=alias.appendingPathComponent("new/nested/paged")
          let actual=try MLXTransformerPageConverter.convert(plan,destination:nested)
          XCTAssertEqual(actual.path,(try MLXTransformerPageConverter.canonicalLocalURL(parent)).path+"/new/nested/paged")
          XCTAssertEqual(try MLXTransformerPageConverter.canonicalLocalURL(actual).path,actual.path)
          XCTAssertTrue(FileManager.default.fileExists(atPath:actual.appendingPathComponent("paged_manifest.json").path))
        }
      }
    }
  }
  func testSourceReplacementAndObserverCancellationCannotPublish() throws {
    try Device.withDefaultDevice(.cpu) {
      try withTiny { source,plan in
        try withDestination { destination in
          XCTAssertThrowsError(try MLXTransformerPageConverter.convert(plan,destination:destination,progress:{ event in
            if event.stage=="page_complete" { throw CancellationError() }
          }))
          XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
          let remaining=try FileManager.default.contentsOfDirectory(atPath:destination.deletingLastPathComponent().path)
          XCTAssertTrue(remaining.isEmpty)
          let bytes=try Data(contentsOf:source)
          try FileManager.default.removeItem(at:source);try bytes.write(to:source)
          XCTAssertThrowsError(try MLXTransformerPageConverter.convert(plan,destination:destination))
          XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
        }
      }
    }
  }
  @MainActor func testTaskCancellationAfterFirstPageRemovesPartialTree() async throws {
    let canceled=try await Task {
      var caught=false
      try Device.withDefaultDevice(.cpu) {
        try withTiny { _,plan in
          try withDestination { destination in
            do {
              _ = try MLXTransformerPageConverter.convert(plan,destination:destination) { event in
                if event.stage=="page_complete" { withUnsafeCurrentTask { $0?.cancel() } }
              }
            } catch is CancellationError { caught=true }
            XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
            XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath:destination.deletingLastPathComponent().path).isEmpty)
          }
        }
      }
      return caught
    }.value
    XCTAssertTrue(canceled)
  }
  func testInstalledRawHeaderAdmissionWhenRequested() throws {
    guard let source=ProcessInfo.processInfo.environment["WEETODD_LTX_RAW_TRANSFORMER_HEADER"] else {
      throw XCTSkip("Installed raw transformer header admission is opt-in; no conversion.")
    }
    let plan=try MLXTransformerPageConverter.preflight(source:URL(fileURLWithPath:source))
    XCTAssertEqual(plan.blockCount,48)
    XCTAssertGreaterThan(plan.sourceTensorBytes,32*1024*1024*1024)
    try plan.checkUnchanged()
  }
  func testPerTensorWorkingAdmissionDoesNotLimitStreamedFixedPage() throws {
    try withTiny { source,plan in
      let file=try SafeTensorFile(url:source)
      XCTAssertThrowsError(try MLXTransformerPageConverter.plan(source:source,file:file,metadata:[:],blockCount:2,
        blockShapes:["attn1.to_q.weight":[2,64],"attn1.to_q.bias":[2]],fixedShapes:[:],maximumTensorBytes:511,maximumWorkingBytes:8*1024*1024))
      // Oversized manifest metadata fails Plan admission, before hashing or
      // quantizing a source payload or creating any destination directory.
      XCTAssertThrowsError(try MLXTransformerPageConverter.plan(source:source,file:file,
        metadata:["padding":String(repeating:"z",count:1024*1024)],blockCount:2,
        blockShapes:["attn1.to_q.weight":[2,64],"attn1.to_q.bias":[2]],fixedShapes:[:],
        maximumTensorBytes:512,maximumWorkingBytes:8*1024*1024))
      XCTAssertEqual(plan.largestQuantizationInputBytes,512)
      XCTAssertEqual(file.tensors[stem+"patchify_proj.weight"]?.byteCount,1024)
      XCTAssertNoThrow(try MLXTransformerPageConverter.plan(source:source,file:file,metadata:[:],blockCount:2,
        blockShapes:["attn1.to_q.weight":[2,64],"attn1.to_q.bias":[2]],fixedShapes:[:],maximumTensorBytes:512,maximumWorkingBytes:8*1024*1024))
      XCTAssertThrowsError(try MLXTransformerPageConverter.preflight(source:source))
    }
  }
  private func architecture() -> [String:Any] {
    ["num_layers":48,"num_attention_heads":32,"audio_num_attention_heads":32,"attention_head_dim":128,
      "audio_attention_head_dim":64,"cross_attention_dim":4096,"audio_cross_attention_dim":2048,
      "in_channels":128,"out_channels":128,"audio_out_channels":128,"timestep_scale_multiplier":1000,
      "av_ca_timestep_scale_multiplier":1000,"positional_embedding_theta":10000,"frequencies_precision":"float64",
      "positional_embedding_max_pos":[20,2048,2048],"audio_positional_embedding_max_pos":[20],"ff_bias":false,
      "apply_gated_attention":true,"cross_attention_adaln":true,"use_audio_video_cross_attention":true,
      "rope_type":"split","qk_norm":"rms_norm","norm_eps":1e-6,"activation_fn":"gelu-approximate",
      "attention_bias":true,"double_self_attention":false,"only_cross_attention":false,"share_ff":false,
      "av_cross_ada_norm":true,"norm_elementwise_affine":false]
  }
  func testCompleteProductionShapePreflightReadsOnlySparseHeader() throws {
    let c=try AVBlockConfiguration(videoTokens:1,audioTokens:1,textTokens:1)
    let shapes=MLXTransformerPageConverter.expectedBlockShapes(c)
    XCTAssertEqual(shapes,try MLXAVBlock(configuration:c).weightShapes,
      "Conversion admission must match the actual shared native block layout")
    var tensors:[(String,[Int],String)]=[]
    for index in 0..<48 { tensors += shapes.keys.sorted().map { (stem+"transformer_blocks.\(index)."+$0,shapes[$0]!,"BF16") } }
    var fixed=DenoiserLayout.weightShapes(c);fixed["keyframes_abs_pos_embedding"]=[1,4096]
    tensors += fixed.keys.sorted().map { (stem+$0,fixed[$0]!,"BF16") }
    let config=String(data:try JSONSerialization.data(withJSONObject:["transformer":architecture()]),encoding:.utf8)!
    try withTensorFile(metadata:["model_version":"2.5.0","config":config],tensors:tensors) { url in
      let plan=try MLXTransformerPageConverter.preflight(source:url)
      XCTAssertEqual(plan.blockCount,48)
      XCTAssertGreaterThan(plan.sourceTensorBytes,32*1024*1024*1024)
      XCTAssertEqual(plan.largestQuantizationInputBytes,134217728)
      try plan.checkUnchanged()
      // Transport admission uses the same actual sparse full-layout source;
      // captured identity mismatch rejects before any source payload/hash.
      let destination=url.deletingLastPathComponent().appendingPathComponent("converted")
      let captured=try XCTUnwrap(JSONSerialization.jsonObject(with:JSONEncoder().encode(plan.sourceIdentity)) as? [String:Any])
      var request:[String:Any]=["version":1,"engine":"ltx25","task":"transformer-page-conversion",
        "source_path":url.path,"output_directory":destination.path,"source_identity":captured]
      let contract=try MLXTransformerConversionRequest(data:JSONSerialization.data(withJSONObject:request),
        outputDirectory:destination,requiresSourceIdentity:true)
      XCTAssertEqual(try contract.preflight().sourceIdentity,plan.sourceIdentity)
      for key in ["inode","headerSHA256"] {
        var different=captured
        if key=="inode" { different[key]=NSNumber(value:plan.sourceIdentity.inode+1) }
        else { different[key]=String(repeating:"0",count:64) }
        request["source_identity"]=different
        let changed=try MLXTransformerConversionRequest(data:JSONSerialization.data(withJSONObject:request),
          outputDirectory:destination,requiresSourceIdentity:true)
        XCTAssertThrowsError(try changed.preflight())
      }
      XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
    }
    // One missing block tensor fails before any payload can be touched.
    tensors.removeFirst()
    try withTensorFile(metadata:["model_version":"2.5.0","config":config],tensors:tensors) { url in
      XCTAssertThrowsError(try MLXTransformerPageConverter.preflight(source:url))
    }
  }
}
