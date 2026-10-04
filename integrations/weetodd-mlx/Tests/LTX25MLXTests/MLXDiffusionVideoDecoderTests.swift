import Foundation
import MLX
import XCTest
@testable import LTX25MLX

final class MLXDiffusionVideoDecoderTests: XCTestCase {
  func testPatchSubpixelOrderAndInverseWithoutChannelPermutation() throws {
    try Device.withDefaultDevice(.cpu) {
      let image=MLXArray((0..<12).map(Float.init),[1,3,1,2,2])
      let packed=try MLXDiffusionVideoMath.patch(image,patch:2)
      XCTAssertEqual(packed.shape,[1,1,1,1,12])
      XCTAssertEqual(packed.asArray(Float.self),[0,2,1,3,4,6,5,7,8,10,9,11])
      XCTAssertEqual(try MLXDiffusionVideoMath.unpatch(packed,patch:2).asArray(Float.self),image.asArray(Float.self))
    }
  }
  func testLowPrecisionEulerMustNotShortcutPrediction() throws {
    try Device.withDefaultDevice(.cpu) {
      let noise=MLXArray([Float(1)]).asType(.bfloat16)
      let prediction=MLXArray([Float(0.010009765625)]).asType(.bfloat16)
      let result=MLXDiffusionVideoMath.oneStep(noise:noise,prediction:prediction)
      XCTAssertEqual(result.item(Float.self),0.01171875)
      XCTAssertNotEqual(result.item(Float.self),prediction.item(Float.self))
    }
  }
  func testShiftedWindowsKeepFullVolumeAtBordersAndAcrossQueryChunks() throws {
    let shape=[2,4,5,6,2,4],kernel=[3,3,5]
    let full=try MLXDiffusionVideoMath.neighborhoodIndices(shape:shape,kernel:kernel,queries:0..<120)
    let split=try MLXDiffusionVideoMath.neighborhoodIndices(shape:shape,kernel:kernel,queries:0..<37)
      + MLXDiffusionVideoMath.neighborhoodIndices(shape:shape,kernel:kernel,queries:37..<120)
    XCTAssertEqual(full,split)
    XCTAssertEqual(Array(full.prefix(5)),[0,1,2,3,4])
    XCTAssertEqual(Array(full.suffix(5)),[115,116,117,118,119])
    XCTAssertEqual(full.count,120*45)
    XCTAssertThrowsError(try MLXDiffusionVideoMath.neighborhoodIndices(shape:shape,kernel:[5,3,5],queries:0..<1))
  }
  func testReferenceAttentionAgainstIndependentScalarAcrossChunkBoundaries() throws {
    try Device.withDefaultDevice(.cpu) {
      let shape=[2,4,5,6,2,4],count=shape.reduce(1,*)
      let q=(0..<count).map { Float(($0*17)%29-14)/29 }
      let k=(0..<count).map { Float(($0*11)%31-15)/31 }
      let v=(0..<count).map { Float(($0*7)%37-18)/37 }
      let qm=MLXArray(q,shape),km=MLXArray(k,shape),vm=MLXArray(v,shape)
      let a=try MLXDiffusionVideoMath.referenceAttention(q:qm,k:km,v:vm,kernel:[3,3,5],queryChunk:37)
      let b=try MLXDiffusionVideoMath.referenceAttention(q:qm,k:km,v:vm,kernel:[3,3,5],queryChunk:113)
      let actual=a.asArray(Float.self),other=b.asArray(Float.self)
      for i in actual.indices { XCTAssertEqual(actual[i],other[i],accuracy:0.00001) }
      // Independent scalar loops, including batches, head-specific scores and inward-shifted borders.
      for flat in [0,36,37,119,120,239] {
        let batch=flat/120,local=flat%120,head=flat%2
        let indices=try MLXDiffusionVideoMath.neighborhoodIndices(shape:shape,kernel:[3,3,5],queries:local..<local+1)
        let scores=indices.map { row -> Double in
          (0..<4).reduce(0) { $0+Double(q[flat*8+head*4+$1])*Double(k[(batch*120+Int(row))*8+head*4+$1]) }/2
        }
        let maximum=scores.max()!,exponents=scores.map { exp($0-maximum) },total=exponents.reduce(0,+)
        for channel in 0..<4 {
          let expected=zip(indices,exponents).reduce(0.0) { $0+Double(v[(batch*120+Int($1.0))*8+head*4+channel])*$1.1/total }
          XCTAssertEqual(actual[flat*8+head*4+channel],Float(expected),accuracy:0.00002)
        }
      }
    }
  }
  func testPlanRejectsOversizedGatherBeforeAnyCheckpointOrArrayAllocation() throws {
    let options=try MLXDiffusionVideoOptions(queryChunkSize:512)
    let plan=try MLXDiffusionVideoPlan(shape:[1,128,12,24,42],options:options)
    XCTAssertEqual(plan.outputShape,[1,3,89,768,1344])
    XCTAssertEqual(plan.stageShapes.map { $0[1] },[14,14,27,53,105])
    XCTAssertEqual(plan.widthHalo,24)
    XCTAssertThrowsError(try MLXDiffusionVideoPlan(shape:[1,128,12,24,42],options:try MLXDiffusionVideoOptions(queryChunkSize:65_536)))
    XCTAssertThrowsError(try MLXDiffusionVideoPlan(shape:[1,128,1,7,7],options:options))
    XCTAssertThrowsError(try MLXDiffusionVideoOptions(optimization:.stage4WidthTiles,stage4TileWidth:0))
    XCTAssertThrowsError(try MLXDiffusionVideoOptions(optimization:.combined,stage4TileWidth:4))
  }
  func testWholeHeadRMSAndAdjacentThreeAxisRotary() throws {
    try Device.withDefaultDevice(.cpu) {
      let norm=MLXDiffusionVideoMath.rms(MLXArray([Float(1),2,3,4]),weight:MLXArray.ones([4]))
      let expected:[Float]=[0.3651483357,0.7302966714,1.0954450369,1.4605933428]
      for (a,b) in zip(norm.asArray(Float.self),expected) { XCTAssertEqual(a,b,accuracy:0.000001) }
      XCTAssertEqual(MLXDiffusionVideoMath.rotarySplit(64),[16,24,24])
      let values:[Float]=(0..<512).map { index in Float(index % 17)/Float(17) }
      let x=MLXArray(values,[1,2,2,2,1,64])
      let out=try MLXDiffusionVideoMath.rotary(x).asArray(Float.self),input=x.asArray(Float.self)
      XCTAssertEqual(Array(out.prefix(64)),Array(input.prefix(64)))
      // Position(t=0,h=0,w=1): T/H channels unchanged, W rotates adjacent0/1 by1rad.
      let start=64+40
      XCTAssertEqual(out[start],input[start]*Float(Foundation.cos(Double(1)))-input[start+1]*Float(Foundation.sin(Double(1))),accuracy:0.000001)
      XCTAssertEqual(out[start+1],input[start]*Float(Foundation.sin(Double(1)))+input[start+1]*Float(Foundation.cos(Double(1))),accuracy:0.000001)
    }
  }
}

extension MLXDiffusionVideoDecoderTests {
  private func headerFixture(wrongShape:Bool=false,wrongStep:Any=1) throws -> URL {
    let url=FileManager.default.temporaryDirectory.appendingPathComponent("diffvae-\(UUID().uuidString).safetensors")
    addTeardownBlock { try? FileManager.default.removeItem(at:url) }
    let decoder:[String:Any]=["_class_name":"NADiffusionDecoder","in_channels":128,"out_channels":3,"patch_size":4,"head_dim":64,
      "stage_channels":[2048,1024,512,512,256],"stage_depths":[4,6,4,2,8],
      "stage_kernels":[[3,7,7],[3,7,7],[3,5,5],[3,5,5],[11,11,11]],"stage5_kernel":[11,11,11],
      "upsamples":[[[1,2,2],2],[[2,1,1],2],[[2,2,2],1],[[2,2,2],2]],
      "timestep_scale_multiplier":1000,"default_num_inference_steps":wrongStep,
      "resampler_kind":"linear","spatial_padding_mode":"zeros"]
    let raw=try JSONSerialization.data(withJSONObject:["vae":["decoder":decoder,"model_output_type":"x0"]])
    var header:[String:Any]=["__metadata__":["model_version":"2.5.0","config":String(data:raw,encoding:.utf8)!]],offset=0
    var shapes=MLXDiffusionVideoCheckpoint.shapes
    if wrongShape { shapes["decoder.det_stages.0.0.attn.q_norm.weight"]=[128] }
    for name in shapes.keys.sorted() {
      let shape=shapes[name]!,bytes=shape.reduce(1,*)*2
      header[name]=["dtype":"BF16","shape":shape,"data_offsets":[offset,offset+bytes]];offset+=bytes
    }
    let encoded=try JSONSerialization.data(withJSONObject:header,options:.sortedKeys)
    var length=UInt64(encoded.count).littleEndian
    var data=withUnsafeBytes(of:&length) { Data($0) };data.append(encoded)
    try data.write(to:url,options:.withoutOverwriting)
    // Sparse synthetic payload: descriptor validation only. No model weights
    // are copied, generated or read by these architecture/mutation tests.
    let handle=try FileHandle(forWritingTo:url);try handle.truncate(atOffset:UInt64(data.count+offset));try handle.close()
    return url
  }
  func testHeaderAdmissionRejectsWrongHeadLayoutAndBooleanStepBeforePayload() throws {
    let valid=try MLXDiffusionVideoCheckpoint(checkpoint:headerFixture())
    XCTAssertEqual(MLXDiffusionVideoCheckpoint.shapes.count,312)
    XCTAssertEqual(valid.decoderTensorBytes,834_268_000)
    XCTAssertThrowsError(try MLXDiffusionVideoCheckpoint(checkpoint:headerFixture(wrongShape:true)))
    XCTAssertThrowsError(try MLXDiffusionVideoCheckpoint(checkpoint:headerFixture(wrongStep:true)))
    let url=valid.url,handle=try FileHandle(forWritingTo:url)
    try handle.seekToEnd();try handle.write(contentsOf:Data([0]));try handle.close()
    XCTAssertThrowsError(try valid.file.checkUnchanged(at:url))
  }
  func testSparseHeaderCancellationBeforeAnyTensorRead() async throws {
    let checkpoint=try headerFixture()
    let task=Task<Void,Error>.detached { @Sendable [checkpoint] in
      try Device.withDefaultDevice(.cpu) {
        let decoder=try MLXDiffusionVideoDecoder(checkpoint:checkpoint)
        let input=MLXArray.zeros([1,128,3,7,7])
        withUnsafeCurrentTask { $0?.cancel() }
        _=try decoder.decode(latent:input)
      }
    }
    do { try await task.value;XCTFail("Cancelled decode must not load weights") }
    catch is CancellationError {} catch { XCTFail("Unexpected cancellation error: \(error)") }
  }
  func testStrictDecodeControlsAndUnsupportedCombinationDoNotDisappear() throws {
    let good=Data(#"{"optimization":"deferred_stage4","query_chunk_size":512,"context_width_chunks":4,"stage4_tile_width":0}"#.utf8)
    let settings=try JSONDecoder().decode(MLXDiffusionVideoSettings.self,from:good)
    XCTAssertEqual(settings.optimization,.deferredStage4)
    XCTAssertEqual(try settings.options(maximumWorkspaceBytes:1024*1024*1024).contextWidthChunks,4)
    for text in [
      #"{"optimization":"deferred_stage4","query_chunk_size":512,"context_width_chunks":4,"stage4_tile_width":0,"unknown":true}"#,
      #"{"optimization":"combined","query_chunk_size":true,"context_width_chunks":4,"stage4_tile_width":0}"#,
      #"{"optimization":"combined","query_chunk_size":512,"context_width_chunks":4,"stage4_tile_width":8}"#] {
      XCTAssertThrowsError(try JSONDecoder().decode(MLXDiffusionVideoSettings.self,from:Data(text.utf8)))
    }
  }
  func testInstalledDiffusionHeaderWithoutLoadingWeightsWhenProvided() throws {
    guard let path=ProcessInfo.processInfo.environment["WEETODD_LTX_DIFFVAE_CHECKPOINT"] else { throw XCTSkip("Optional installed DiffVAE header") }
    let header=try MLXDiffusionVideoCheckpoint(checkpoint:URL(fileURLWithPath:path))
    XCTAssertEqual(header.file.tensors.keys.filter { $0.hasPrefix("decoder.") }.count,310)
    XCTAssertTrue(try MLXVideoDecoderSelection(checkpoint:header.url).isDiffusion)
  }
  func testInstalledFirstBlockObserverFailureReleasesScopedWeightsWhenProvided() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_LTX_DIFFVAE_WEIGHTED_FIRST_BLOCK"] == "1",
      let path=ProcessInfo.processInfo.environment["WEETODD_LTX_DIFFVAE_CHECKPOINT"] else { throw XCTSkip("Optional real DiffVAE component lifecycle witness; root schedules") }
    let decoder=try MLXDiffusionVideoDecoder(checkpoint:URL(fileURLWithPath:path))
    let oldCache=Memory.cacheLimit
    XCTAssertThrowsError(try decoder.decode(latent:MLXArray.zeros([1,128,3,7,7]).asType(.bfloat16),progress:{ phase,_,_ in
      if phase == "context_stage_1" { throw CancellationError() }
    })) { XCTAssertTrue($0 is CancellationError) }
    XCTAssertEqual(decoder.residentWeightBytes,0);XCTAssertEqual(Memory.cacheLimit,oldCache)
    XCTAssertGreaterThan(decoder.maximumResidentWeightBytes,0)
    XCTAssertLessThanOrEqual(decoder.maximumResidentWeightBytes,256*1024*1024)
  }
}

extension MLXDiffusionVideoDecoderTests {
  func testActualPublicationMetadataDoesNotMislabelConvolutionalTiling() throws {
    XCTAssertTrue(MLXNativeVideoDecoder.publicationMetadata(isDiffusion:false,settings:nil).isEmpty)
    let settings=try MLXDiffusionVideoSettings(optimization:.stage4WidthTiles,queryChunkSize:37,contextWidthChunks:3,stage4TileWidth:9)
    let actual=MLXNativeVideoDecoder.publicationMetadata(isDiffusion:true,settings:settings)
    XCTAssertEqual(actual["video_decoder_architecture"] as? String,"ltx25-one-step-diffusion-vae")
    XCTAssertEqual(actual["video_input_latent_dtype"] as? String,"float32")
    XCTAssertEqual(actual["video_stage_latent_dtype"] as? String,"bfloat16")
    XCTAssertEqual(actual["diffvae_noise_seed"] as? Int,0)
    XCTAssertEqual(actual["diffvae_query_chunk_size"] as? Int,37)
    XCTAssertEqual(actual["diffvae_stage4_tile_width"] as? Int,9)
    XCTAssertFalse(String(describing:actual).contains("32-frame"))
  }
  func testHeaderDispatchKeepsSparseConvolutionalAdmissionAndRejectsIgnoredControls() throws {
    let path=try MLXDecoderHeaderFixture.convolutional(test:self)
    XCTAssertFalse(try MLXVideoDecoderSelection(checkpoint:path).isDiffusion)
    XCTAssertThrowsError(try MLXVideoDecoderSelection(checkpoint:path,settings:MLXDiffusionVideoSettings(queryChunkSize:113)))
  }
  /// This numerical witness is separately scheduled; it performs no model load.
  /// Fused Metal rounds RMS+RoPE only at output and uses FP32 scores/probabilities.
  /// Reference preserves BF16 RMS, score and probability boundaries. Therefore
  /// qualification uses a PREDECLARED bounded error, never an exact-byte claim.
  func testExperimentalMetalHead64AgainstReferenceWhenExplicitlyScheduled() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_LTX_DIFFVAE_METAL_NUMERICS"] == "1" else {
      throw XCTSkip("Separate root-scheduled experimental Metal numerical qualification")
    }
    let shape=[2,3,3,5,2,64],count=shape.reduce(1,*),rows=2*3*3*5
    let values=(0..<count).map { Float(($0*17)%67-33)/33 }
    let keys=(0..<count).map { Float(($0*13)%71-35)/35 }
    let vals=(0..<count).map { Float(($0*7)%73-36)/36 }
    let weight=MLXArray((0..<64).map { Float(0.75)+Float($0%7)/28 }).asType(.bfloat16)
    let q=MLXArray(values,shape).asType(.bfloat16),k=MLXArray(keys,shape).asType(.bfloat16),v=MLXArray(vals,shape).asType(.bfloat16)
    let qr=try MLXDiffusionVideoMath.rotary(MLXDiffusionVideoMath.rms(q,weight:weight))
    let kr=try MLXDiffusionVideoMath.rotary(MLXDiffusionVideoMath.rms(k,weight:weight))
    let qf=q.reshaped([rows,2,64]),kf=k.reshaped([rows,2,64]);var qs:[MLXArray]=[],ks:[MLXArray]=[]
    for start in stride(from:0,to:rows,by:37) {
      let end=min(rows,start+37)
      qs.append(try MLXDiffusionVideoMetal.normRotary(qf[start..<end],weight:weight,shape:shape,queryStart:start))
      ks.append(try MLXDiffusionVideoMetal.normRotary(kf[start..<end],weight:weight,shape:shape,queryStart:start))
    }
    let qm=concatenated(qs,axis:0).reshaped(shape),km=concatenated(ks,axis:0).reshaped(shape)
    // RMS + adjacent RoPE fusion differs by at most a few BF16 rounding units
    // for this deliberately bounded input; limits fixed before any GPU run.
    for (a,b) in zip(qm.asArray(Float.self),qr.asArray(Float.self)) { XCTAssertEqual(a,b,accuracy:0.0625) }
    for (a,b) in zip(km.asArray(Float.self),kr.asArray(Float.self)) { XCTAssertEqual(a,b,accuracy:0.0625) }
    let expected=try MLXDiffusionVideoMath.referenceAttention(q:qr,k:kr,v:v,kernel:[3,3,5],queryChunk:37).asArray(Float.self)
    func actual(_ chunk:Int) throws -> [Float] {
      let flat=qm.reshaped([rows,2,64]);var pieces:[MLXArray]=[]
      for start in stride(from:0,to:rows,by:chunk) {
        pieces.append(try MLXDiffusionVideoMetal.attend(q:flat[start..<min(rows,start+chunk)],k:km,v:v,shape:shape,kernel:[3,3,5],queryStart:start))
      }
      return concatenated(pieces,axis:0).asArray(Float.self)
    }
    let first=try actual(37),second=try actual(113)
    XCTAssertEqual(first,second,"Query batching must not reset global batch/border coordinates")
    for (a,b) in zip(first,expected) { XCTAssertTrue(a.isFinite);XCTAssertEqual(a,b,accuracy:0.0625) }
  }
}
