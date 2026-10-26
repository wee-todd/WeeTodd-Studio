import Foundation
import XCTest
@testable import H3MLX

final class H3ImageVAEPurposeTests: XCTestCase {
  /// Complete decoder directory with a sparse zero payload. Admission reads only
  /// its header; this fixture contains no installed weights and performs no GPU work.
  private func imageDecoder() throws -> URL {
    let metadata: [String:Any] = ["format":"minimax-h3-mlx-video-vae", "format_version":1,
      "tensor_layout":"ODHWI", "vae_clip_length":17, "vae_token_drop":3,
      "latents_mean":[Float](repeating:0,count:24), "latents_std":[Float](repeating:1,count:24),
      "quantization":["format":"mlx-affine", "bits":8, "group_size":64,
        "scope":"decoder-transformer-core", "quantized_layers":144],
      "image_only":true, "image_output_slice":3]
    let encoded = try JSONSerialization.data(withJSONObject: metadata)
    var header: [String:Any] = ["__metadata__":["minimax_h3_video_vae":String(decoding:encoded,as:UTF8.self)]]
    var offset: UInt64 = 0
    func tensor(_ name: String, _ dtype: String, _ shape: [Int]) {
      let count = shape.reduce(1,*) * (dtype == "U32" ? 4 : 2)
      header[name] = ["dtype":dtype, "shape":shape, "data_offsets":[offset,offset+UInt64(count)]]
      offset += UInt64(count)
    }
    tensor("post_quant_conv.weight","F16",[24,1,1,1,24])
    tensor("post_quant_conv.bias","F16",[24])
    tensor("decoder.x_embedder.weight","F16",[2048,24])
    tensor("decoder.x_embedder.bias","F16",[2048])
    tensor("decoder.register_tokens","F16",[1,4,2048])
    tensor("decoder.norm_out.weight","F16",[2048])
    tensor("decoder.norm_out.bias","F16",[2048])
    tensor("decoder.proj_out.weight","F16",[3072,2048])
    tensor("decoder.proj_out.bias","F16",[3072])
    for index in 0..<36 {
      let prefix = "decoder.transformer_blocks.\(index)."
      for suffix in ["norm1.weight","norm2.weight","scale1","scale2"] {
        tensor(prefix+suffix,"F16",[2048])
      }
      for (name,rows,columns) in [("attn.to_qkv",6144,2048),("attn.to_out",2048,2048),
        ("ff.w1",16384,2048),("ff.w2",2048,8192)] {
        tensor(prefix+name+".weight","U32",[rows,columns/4])
        tensor(prefix+name+".scales","F16",[rows,columns/64])
        tensor(prefix+name+".biases","F16",[rows,columns/64])
        tensor(prefix+name+".bias","F16",[rows])
      }
    }
    let json = try JSONSerialization.data(withJSONObject:header,options:[.sortedKeys])
    var length = UInt64(json.count).littleEndian
    var bytes = withUnsafeBytes(of:&length) { Data($0) }; bytes.append(json)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".safetensors")
    try bytes.write(to:url)
    let file = try FileHandle(forWritingTo:url); defer { try? file.close() }
    try file.truncate(atOffset:UInt64(bytes.count)+offset)
    return url
  }

  private func assertImagePurposeError(_ body: () throws -> Void,
    file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertThrowsError(try body(),file:file,line:line) { error in
      guard case H3CheckpointError.invalid(let message) = error else {
        return XCTFail("Expected image-only purpose rejection, received \(error)",file:file,line:line)
      }
      XCTAssertTrue(message.contains("single images only"),message,file:file,line:line)
    }
  }

  func testImageOnlyDecoderFailsVideoEncoderPreflightBeforeEncoderInspection() throws {
    let url = try imageDecoder(); defer { try? FileManager.default.removeItem(at:url) }
    XCTAssertTrue(try H3VideoVAELayout(url:url).imageOnly)
    assertImagePurposeError { try H3VideoVAEEncoder.preflight(checkpointURL:url) }
  }

  func testImageOnlyDecoderFailsDirectReferenceEncodingBeforeWeights() throws {
    let url = try imageDecoder(); defer { try? FileManager.default.removeItem(at:url) }
    assertImagePurposeError {
      _ = try H3VideoVAEEncoder.encodeStill(checkpointURL:url,
        rgb8:[UInt8](repeating:0,count:32*32*3),width:32,height:32)
    }
  }

  func testImageOnlyDecoderFailsContinuationBeforeOtherComponentsAreRead() throws {
    let url = try imageDecoder(); defer { try? FileManager.default.removeItem(at:url) }
    let missing = URL(fileURLWithPath:"/missing/h3-component")
    let request = try H3T2VARequest(prompt:"A robot.",width:32,height:32,
      durationSeconds:2.5,seed:42,requestedSteps:5,transformer:missing,
      qwenPages:missing,tokenizer:missing,videoVAE:url,audioVAE:missing)
    assertImagePurposeError { _ = try H3ContinuationRunner.preflight(request,contextFrames:22) }
  }
}
