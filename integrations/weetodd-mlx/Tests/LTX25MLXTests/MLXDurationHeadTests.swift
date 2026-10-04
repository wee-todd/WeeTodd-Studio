import Foundation
import MLX
import XCTest
@testable import LTX25MLX

final class MLXDurationHeadTests: XCTestCase {
  private func fixture(wrongShape: Bool = false) throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let path = root.appendingPathComponent("duration.safetensors")
    var shapes = MLXDurationHead.shapes
    if wrongShape { shapes["video_input_proj.weight"] = [256,4095] }
    let sparse: [String: [Int: Float]] = [
      "video_input_proj.weight": [0:1,4097:1], "video_input_proj.bias": [0:0.25,1:-0.125],
      "video_modality_emb": [0:0.125,1:0.25],
      "audio_input_proj.weight": [0:1,2049:1], "audio_input_proj.bias": [0:-0.25,1:0.125],
      "audio_modality_emb": [0:-0.125,1:-0.25],
      "attention_pooler.query_tokens": [0:0.5,1:-0.25],
      "attention_pooler.cross_attn.in_proj_weight": [0:1,257:1,65536:1,65793:1,131072:0.5,131329:1.5],
      "attention_pooler.cross_attn.in_proj_bias": [0:0.125,1:0.125,256:-0.25,257:0.125,512:0.25,513:-0.25],
      "attention_pooler.cross_attn.out_proj.weight": [0:1,1:0.25,256:-0.5,257:1],
      "attention_pooler.cross_attn.out_proj.bias": [0:-0.125,1:0.125],
      "mlp_hidden.weight": [0:0.5,1:0.25,256:0.25,257:-0.5], "mlp_hidden.bias": [0:0.25,1:-0.125],
      "mlp_out.weight": [0:0.25,1:-0.375], "mlp_out.bias": [0:0.5]]
    let configuration = try JSONSerialization.data(withJSONObject: ["transformer": ["cross_attention_dim":4096,"audio_cross_attention_dim":2048],"duration_head":[:]] as [String:Any])
    var header: [String: Any] = ["__metadata__": ["model_version":"2.5.0","config":String(data:configuration,encoding:.utf8)!]]
    var payload = Data()
    for name in shapes.keys.sorted() {
      let shape = shapes[name]!, count = shape.reduce(1, *)
      let start = payload.count
      var data = Data(repeating:0,count:count*2)
      data.withUnsafeMutableBytes { (bytes: UnsafeMutableRawBufferPointer) in
        for (index,value) in sparse[name] ?? [:] {
          let bits = UInt16(value.bitPattern >> 16).littleEndian
          bytes.storeBytes(of:bits,toByteOffset:index*2,as:UInt16.self)
        }
      }
      payload.append(data)
      header["duration_head."+name] = ["dtype":"BF16","shape":shape,"data_offsets":[start,payload.count]]
    }
    let encoded = try JSONSerialization.data(withJSONObject:header,options:.sortedKeys)
    var length = UInt64(encoded.count).littleEndian
    var bytes = withUnsafeBytes(of:&length) { Data($0) };bytes.append(encoded);bytes.append(payload)
    try bytes.write(to:path)
    return path
  }
  private static func streams(changedAudio: Bool = false) -> (MLXArray, MLXArray) {
    var video = [Float](repeating:0,count:3*4096)
    for (index,pair) in [(Float(0.25),Float(1)),(1.25,-0.5),(-0.75,0.25)].enumerated() {
      video[index*4096] = pair.0;video[index*4096+1] = pair.1
    }
    var audio = [Float](repeating:0,count:2*2048)
    audio[0] = changedAudio ? 2 : 0.5;audio[1] = 0.75;audio[2048] = -1.5;audio[2049] = -0.25
    return (MLXArray(video,[3,4096]),MLXArray(audio,[2,2048]))
  }
  func testOwnedIndependentNumericFixtureOnCPUUsesBothModalitiesAndBatchOne() throws {
    try Device.withDefaultDevice(.cpu) {
      let head = try MLXDurationHead(checkpoint:fixture()), (video,audio) = Self.streams()
      XCTAssertEqual(head.tensorBytes,3_806_722)
      // Independent scalar Python math oracle; nonuniform keys/values and both modalities.
      let result = try head.predict(video:video,audio:audio)
      XCTAssertEqual(result,1.8003680065354255,accuracy:0.00001)
      XCTAssertEqual(try head.predict(video:video.reshaped([1,3,4096]),audio:audio.reshaped([1,2,2048])),result,accuracy:0.000001)
      let changed = Self.streams(changedAudio:true)
      XCTAssertEqual(try head.predict(video:changed.0,audio:changed.1),1.8030944297386597,accuracy:0.00001)
    }
  }
  func testInvalidArchitectureAndMutationRejectBeforePayloadComputation() throws {
    XCTAssertThrowsError(try MLXDurationHead(checkpoint:fixture(wrongShape:true)))
    try Device.withDefaultDevice(.cpu) {
      let url = try fixture(), head = try MLXDurationHead(checkpoint:url)
      let handle = try FileHandle(forWritingTo:url);try handle.seekToEnd();try handle.write(contentsOf:Data([0]));try handle.close()
      let (video,audio) = Self.streams()
      XCTAssertThrowsError(try head.predict(video:video,audio:audio))
    }
  }
  func testChangedPinnedHeadHeaderRejectsBeforeEncoderConstruction() throws {
    let url = try fixture()
    let binding = try MLXTextPreparationBinding(originalRecipeSHA256:String(repeating:"a",count:64),
      prompt:"A room",negativePrompt:nil,gemmaRoot:"/nonexistent/gemma",connectorCheckpoint:"/nonexistent/connector")
    var admissionCalled = false
    XCTAssertThrowsError(try MLXAutomaticTextPreparation.prepare(binding:binding,
      policy:MLXAutomaticDurationPolicy(headCheckpointPath:url.path),fps:24,
      expectedHeadHeaderSHA256:String(repeating:"b",count:64),admitResolvedFrames:{ _ in admissionCalled=true })) {
      XCTAssertTrue(String(describing:$0).contains("Duration head changed before text encoding"))
    }
    XCTAssertFalse(admissionCalled)
    XCTAssertEqual(try MLXDurationHead(checkpoint:url).headerSHA256.count,64)
  }
  func testInvalidStreamBatchAndDtypeAreRejected() throws {
    try Device.withDefaultDevice(.cpu) {
      let head = try MLXDurationHead(checkpoint:fixture()), (video,audio) = Self.streams()
      XCTAssertThrowsError(try head.predict(video:video.asType(.bfloat16),audio:audio))
      XCTAssertThrowsError(try head.predict(video:video.reshaped([3,1,4096]),audio:audio))
      XCTAssertThrowsError(try head.predict(video:video,audio:audio.reshaped([1,4096])))
    }
  }
  func testCancellationBeforeHeadWeightsAreRead() async throws {
    let checkpoint = try fixture()
    let task = Task<Void, Error>.detached { @Sendable [checkpoint] in
      try Device.withDefaultDevice(.cpu) {
        let head = try MLXDurationHead(checkpoint:checkpoint)
        let (video,audio) = MLXDurationHeadTests.streams()
        withUnsafeCurrentTask { $0?.cancel() }
        _ = try head.predict(video:video,audio:audio)
      }
    }
    do { _ = try await task.value;XCTFail("Cancelled duration prediction must not proceed") }
    catch is CancellationError {} catch { XCTFail("Unexpected cancellation error: \(error)") }
  }
  func testOfficialFrameRoundingClampsAndUsesNearestEvenBeforeFloorGrid() throws {
    XCTAssertEqual(try MLXDurationHead.frames(seconds:0.5,fps:24),25)
    XCTAssertEqual(try MLXDurationHead.frames(seconds:3,fps:24),65)
    XCTAssertEqual(try MLXDurationHead.frames(seconds:5,fps:24),113)
    XCTAssertEqual(try MLXDurationHead.frames(seconds:100,fps:24),473)
    XCTAssertEqual(try MLXDurationHead.frames(seconds:2.03125,fps:16),25)
    XCTAssertEqual(try MLXDurationHead.frames(seconds:2.09375,fps:16),33)
    XCTAssertThrowsError(try MLXDurationHead.frames(seconds:2.5,fps:24,minimumSeconds:2.4,maximumSeconds:2.5))
    for seconds in [Double.nan,Double.infinity,0,-1] { XCTAssertThrowsError(try MLXDurationHead.frames(seconds:seconds,fps:24)) }
    XCTAssertEqual(try MLXDurationHead.frames(seconds:0.25,fps:24,minimumSeconds:0.25,maximumSeconds:30),9)
    XCTAssertEqual(try MLXDurationHead.frames(seconds:30,fps:24,minimumSeconds:0.25,maximumSeconds:30),713)
    XCTAssertThrowsError(try MLXDurationHead.frames(seconds:3,fps:24,minimumSeconds:0.24))
    XCTAssertThrowsError(try MLXDurationHead.frames(seconds:3,fps:24,maximumSeconds:30.01))
  }
}
