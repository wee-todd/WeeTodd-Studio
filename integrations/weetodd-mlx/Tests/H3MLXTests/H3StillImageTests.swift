import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3StillImageTests: XCTestCase {
  func testDirect1080StillCanvasIsBoundedIndependentlyOfVideo() throws {
    for mode in H3StillImageMode.allCases {
      let landscape = try H3Geometry(width: 1920, height: 1088, stillImageMode: mode)
      let portrait = try H3Geometry(width: 1088, height: 1920, stillImageMode: mode)
      XCTAssertEqual(landscape.videoRows, mode.latentFrames * 2040)
      XCTAssertEqual(portrait.videoRows, landscape.videoRows)
      XCTAssertThrowsError(try H3Geometry(width: 1920, height: 1120, stillImageMode: mode))
      XCTAssertThrowsError(try H3Geometry(width: 1952, height: 1024, stillImageMode: mode))
      XCTAssertThrowsError(try H3Geometry(width: 1920, height: 1080, stillImageMode: mode))
    }
    XCTAssertThrowsError(try H3CanvasAdmission.ordinary.validate(width: 1920, height: 1088))
    let model = URL(fileURLWithPath: "/tmp/component")
    XCTAssertThrowsError(try H3T2VARequest(prompt: "A copper robot.", width: 1920, height: 1088,
      durationSeconds: 3, seed: 1, requestedSteps: 5, transformer: model, qwenPages: model,
      tokenizer: model, videoVAE: model, audioVAE: model))
  }
  func testInstalledImageDecoderProducesOneFiniteFrameAndRejectsVideoUse() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_IMAGE_VAE_Q8"] else {
      throw XCTSkip("Set the installed image-only Q8 VAE for real decoder qualification.")
    }
    let url = URL(fileURLWithPath:path)
    let layout = try H3VideoVAELayout(url:url)
    XCTAssertTrue(layout.imageOnly)
    XCTAssertThrowsError(try layout.validateVideoUse())
    let image = try H3VideoVAEDecoder.decodeStill(checkpointURL:url,
      latent: MLXArray.zeros([1,1,2,2,24]), mode:.singleLatent)
    XCTAssertEqual(image.shape,[1,1,32,32,3])
    XCTAssertTrue(image.asArray(Float.self).allSatisfy(\.isFinite))
    var chunks=0
    XCTAssertThrowsError(try H3VideoVAEDecoder.decodeChunks(checkpointURL:url,
      latent:MLXArray.zeros([1,7,2,2,24])) { _ in chunks += 1 })
    XCTAssertEqual(chunks,0)
  }
  func testImageGridsKeepPackedAudioWithoutChangingVideoMinimum() throws {
    let quality = try H3Geometry(width: 768, height: 768, stillImageMode: .qualityFiveFrames)
    XCTAssertEqual(quality.frames, 5)
    XCTAssertEqual(quality.videoLatentFrames, 2)
    XCTAssertEqual(quality.audioLatentFrames, 8)
    XCTAssertEqual(quality.videoRows, 1152)
    let single = try H3Geometry(width: 768, height: 768, stillImageMode: .singleLatent)
    XCTAssertEqual(single.frames, 1)
    XCTAssertEqual(single.videoLatentFrames, 1)
    XCTAssertEqual(single.audioLatentFrames, 2)
    XCTAssertThrowsError(try H3Geometry(width: 768, height: 768, durationSeconds: 5.0 / 24))
    XCTAssertThrowsError(try H3Geometry(width: 770, height: 768, stillImageMode: .singleLatent))
    XCTAssertThrowsError(try H3Geometry(width: 2048, height: 2048, stillImageMode: .qualityFiveFrames))
  }
  func testDecodeSelectionDropsOnlyTheThreePaddingFrames() throws {
    let raw = MLXArray(Array(0..<8).map(Float.init), [1,8,1,1,1])
    let middle = try H3StillImageMode.qualityFiveFrames.select(decoded: raw, selection: .middle)
    XCTAssertEqual(middle.shape, [1,1,1,1,1])
    XCTAssertEqual(middle.item(Float.self), 5)
    XCTAssertEqual(try H3StillImageMode.qualityFiveFrames.select(decoded: raw, selection: .first).item(Float.self), 3)
    XCTAssertEqual(try H3StillImageMode.qualityFiveFrames.select(decoded: raw, selection: .last).item(Float.self), 7)
    let one = MLXArray(Array(0..<4).map(Float.init), [1,4,1,1,1])
    XCTAssertEqual(try H3StillImageMode.singleLatent.select(decoded: one, selection: .middle).item(Float.self), 3)
    XCTAssertThrowsError(try H3StillImageMode.singleLatent.select(decoded: raw, selection: .middle))
  }
  func testImageDecoderPurposeCannotBeUsedForVideo() throws {
    let object: [String:Any] = ["format":"minimax-h3-mlx-video-vae", "format_version":1,
      "tensor_layout":"ODHWI", "vae_clip_length":17, "vae_token_drop":3,
      "latents_mean":[Float](repeating:0,count:24), "latents_std":[Float](repeating:1,count:24),
      "quantization":["format":"mlx-affine", "bits":8, "group_size":64,
        "scope":"decoder-transformer-core", "quantized_layers":144],
      "image_only":true, "image_output_slice":3]
    let purpose = try H3VideoVAELayout.validateMetadata(object)
    XCTAssertTrue(purpose.imageOnly)
    XCTAssertThrowsError(try H3StillImageMode.qualityFiveFrames.validateDecoder(imageOnly: purpose.imageOnly))
    XCTAssertNoThrow(try H3StillImageMode.singleLatent.validateDecoder(imageOnly: purpose.imageOnly))
    XCTAssertThrowsError(try H3StillImageMode.singleLatent.validateDecoder(imageOnly: false))
    var bad = object; bad["image_output_slice"] = 0
    XCTAssertThrowsError(try H3VideoVAELayout.validateMetadata(bad))
  }
}
