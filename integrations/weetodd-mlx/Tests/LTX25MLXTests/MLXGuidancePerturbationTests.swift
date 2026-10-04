import XCTest
import MLX
@testable import LTX25MLX
import LTX25Engine

final class MLXGuidancePerturbationTests: XCTestCase {
  private func configuration() throws -> AVBlockConfiguration {
    try AVBlockConfiguration(videoDimension:32,audioDimension:16,heads:2,
      videoHeadDimension:16,audioHeadDimension:8,videoTokens:2,audioTokens:2,textTokens:1)
  }
  func testTargetsAreZeroBasedAndBoundedBeforeWeightedExecution() throws {
    XCTAssertNoThrow(try MLXGuidancePerturbation.none.validate(blockCount:48))
    let selected=MLXGuidancePerturbation(videoSelfAttentionBlocks:[0,47],audioSelfAttentionBlocks:[1],skipCrossModality:true)
    try selected.validate(blockCount:48)
    XCTAssertEqual(selected.block(0),.init(skipVideoSelfAttention:true,skipCrossModality:true))
    XCTAssertEqual(selected.block(1),.init(skipAudioSelfAttention:true,skipCrossModality:true))
    XCTAssertEqual(selected.block(2),.init(skipCrossModality:true))
    for bad in [-1,48] {
      XCTAssertThrowsError(try MLXGuidancePerturbation(videoSelfAttentionBlocks:[bad]).validate(blockCount:48))
      XCTAssertThrowsError(try MLXGuidancePerturbation(audioSelfAttentionBlocks:[bad]).validate(blockCount:48))
    }
    XCTAssertThrowsError(try selected.validate(blockCount:1))
  }
  func testInvalidTargetsReachNeitherFixedNorBlockProvider() throws {
    // Construction only admits shapes; the invalid target is rejected before
    // any MLX operation or weighted provider, including preparation of masks.
    let denoiser=try MLXDenoiser(configuration:configuration(),blockCount:2)
    var fixedCalls=0,blockCalls=0
    let fixed:MLXDenoiser.FixedProvider = { _,_ in
      fixedCalls += 1;throw LTXError.invalid("Fixed provider must not run")
    }
    let blocks:MLXDenoiser.BlockProvider = { _,_,_ in
      blockCalls += 1;throw LTXError.invalid("Block provider must not run")
    }
    let invalid=MLXGuidancePerturbation(videoSelfAttentionBlocks:[2])
    XCTAssertThrowsError(try denoiser.evaluate([:],sigma:1,videoDenoiseMask:[0,1],
      fixedWeights:fixed,blockWeights:blocks,perturbation:invalid))
    XCTAssertThrowsError(try denoiser.evaluatePrepared([:],sigma:1,preparation:nil,
      fixedWeights:fixed,blockWeights:blocks,fixedAdapters:{ _ in [] },blockAdapters:{ _ in [:] },
      progress:{ _ in },perturbation:invalid))
    XCTAssertEqual(fixedCalls,0);XCTAssertEqual(blockCalls,0)
    XCTAssertEqual(denoiser.residentWeightBytes,0)
  }
  func testEveryPerturbationCombinationHasItsOwnCompiledSignature() throws {
    let c=try configuration()
    var signatures=Set<[Int]>()
    for bits in 0..<8 {
      let flags=MLXGuidancePerturbation.Block(skipVideoSelfAttention:bits&1 != 0,
        skipAudioSelfAttention:bits&2 != 0,skipCrossModality:bits&4 != 0)
      let (_,arrays,signature)=MLXBlockGraph.bind(configuration:c,inputs:[:],weights:[:],adapters:[:],perturbation:flags)
      XCTAssertTrue(arrays.isEmpty)
      XCTAssertTrue(signatures.insert(signature).inserted)
    }
    let ordinary=MLXBlockGraph.bind(configuration:c,inputs:[:],weights:[:],adapters:[:]).2
    let explicit=MLXBlockGraph.bind(configuration:c,inputs:[:],weights:[:],adapters:[:],perturbation:.none).2
    XCTAssertEqual(ordinary,explicit)
  }

  /// The small forward fixture uses uniform attention (Q=K=0), identity V and
  /// output projections scaled by 0.5, and a learned head gate of 1.5. Expected values are
  /// calculated independently with scalar RMS normalization.
  private func fixture(_ c:AVBlockConfiguration,selfAttention:Bool,cross:Bool) -> [String:MLXArray] {
    var x:[String:MLXArray]=[:]
    for (stream,width,tokens) in [("video",32,2),("audio",16,2)] {
      var values=[Float](repeating:0,count:tokens*width)
      values[0]=1;values[width+1]=2
      x[stream]=MLXArray(values,[tokens,width])
      var mods=[Float](repeating:0,count:9*width)
      if selfAttention { for column in 0..<width { mods[2*width+column]=1 } }
      x[stream+"_modulation"]=MLXArray(mods,[1,9*width])
      x[stream+"_prompt_modulation"] = .zeros([1,2*width])
      x[stream+"_av_modulation"] = .zeros([1,4*width])
      x[stream+"_av_gate"] = cross ? .ones([1,width]) : .zeros([1,width])
      x[stream+"_text"] = .zeros([1,width])
    }
    for (stream,width) in [("video",16),("audio",8),("video_cross",8),("audio_cross",8)] {
      x[stream+"_rope_cos"] = .ones([4,width/2])
      x[stream+"_rope_sin"] = .zeros([4,width/2])
    }
    return x
  }
  private func parameter(_ name:String) -> MLXArray {
    let audio=name.hasPrefix("audio_") || name.hasSuffix("_audio")
    let width=audio ? 16 : 32
    if name.hasSuffix("norm.weight") {
      let inner=name.contains("to_video") || name.contains("to_audio") ? 16 : width
      return .ones([inner])
    }
    let rows=name.contains("a2v_ca") ? 5 : name.contains("prompt") ? 2 : 9
    return .zeros([rows,width])
  }
  private func linear(_ name:String,_ x:MLXArray) -> MLXArray {
    let cross=name.hasPrefix("audio_to_video") || name.hasPrefix("video_to_audio")
    let output=name.hasPrefix("audio_to_video") ? 32 : name.hasPrefix("video_to_audio") ? 16 : name.hasPrefix("audio_") ? 16 : 32
    let inner=cross ? 16 : output
    if name.hasSuffix("to_q") || name.hasSuffix("to_k") { return .zeros([x.shape[0],inner]) }
    if name.hasSuffix("to_v") { return x[0...,0..<inner] }
    if name.hasSuffix("to_gate_logits") { return .ones([x.shape[0],2])*Float(log(3.0)) }
    if name.hasSuffix("to_out") { return (output == inner ? x : concatenated([x,x],axis:1))*Float(0.5) }
    return .zeros([x.shape[0],name.hasSuffix("proj_in") ? 4*output : output])
  }
  private func normalized(_ values:[Float],width:Int) -> [Float] {
    (0..<2).flatMap { row -> [Float] in
      let chunk=Array(values[(row*width)..<((row+1)*width)])
      let rms=sqrt(chunk.reduce(Float(0)) { $0+$1*$1 }/Float(width)+1e-6)
      return chunk.map { $0/rms }
    }
  }
  private func uniform(_ values:[Float],width:Int) -> [Float] {
    let mean=(0..<width).map { (values[$0]+values[width+$0])/2 }
    return mean+mean
  }
  private func assertClose(_ actual:[Float],_ expected:[Float],file:StaticString=#filePath,line:UInt=#line) {
    XCTAssertEqual(actual.count,expected.count,file:file,line:line)
    XCTAssertLessThan(zip(actual,expected).map { abs($0-$1) }.max() ?? .infinity,0.000003,file:file,line:line)
  }
  func testSelfAttentionUsesValuesAndRetainsNonzeroResidualForEachStreamOnCPU() throws {
    try Device.withDefaultDevice(.cpu) {
      let c=try configuration(),x=fixture(c,selfAttention:true,cross:false)
      let ordinary=MLXAVBlock.forward(configuration:c,x,parameter:parameter,linear:linear)
      for flags in [MLXGuidancePerturbation.Block(skipVideoSelfAttention:true),
        .init(skipAudioSelfAttention:true),.init(skipVideoSelfAttention:true,skipAudioSelfAttention:true)] {
        let result=MLXAVBlock.forward(configuration:c,x,parameter:parameter,linear:linear,perturbation:flags)
        for (name,width,skip) in [("video",32,flags.skipVideoSelfAttention),("audio",16,flags.skipAudioSelfAttention)] {
          let original=x[name]!.asArray(Float.self),norm=normalized(original,width:width)
          let delta=(skip ? norm : uniform(norm,width:width)).map { $0*0.75 }
          assertClose(result[name]!.asArray(Float.self),zip(original,delta).map(+))
          XCTAssertGreaterThan(zip(result[name]!.asArray(Float.self),original).map { abs($0-$1) }.max()!,1)
          if !skip { XCTAssertEqual(result[name]!.asArray(Float.self),ordinary[name]!.asArray(Float.self)) }
        }
      }
    }
  }
  func testCrossSkipRemovesOnlyCrossResidualsAndUsesBothPreUpdateStreamsOnCPU() throws {
    try Device.withDefaultDevice(.cpu) {
      let c=try configuration(),x=fixture(c,selfAttention:false,cross:true)
      let ordinary=MLXAVBlock.forward(configuration:c,x,parameter:parameter,linear:linear)
      let skipped=MLXAVBlock.forward(configuration:c,x,parameter:parameter,linear:linear,perturbation:.init(skipCrossModality:true))
      let v=x["video"]!.asArray(Float.self),a=x["audio"]!.asArray(Float.self)
      let audioMix=uniform(normalized(a,width:16),width:16)
      let videoMix=uniform(normalized(v,width:32),width:32)
      var videoDelta:[Float]=[],audioDelta:[Float]=[]
      for row in 0..<2 {
        let audioRow=Array(audioMix[(row*16)..<((row+1)*16)])
        videoDelta.append(contentsOf:audioRow)
        videoDelta.append(contentsOf:audioRow)
        audioDelta.append(contentsOf:videoMix[(row*32)..<(row*32+16)])
      }
      assertClose(ordinary["video"]!.asArray(Float.self),zip(v,videoDelta.map { $0*0.75 }).map(+))
      assertClose(ordinary["audio"]!.asArray(Float.self),zip(a,audioDelta.map { $0*0.75 }).map(+))
      XCTAssertEqual(skipped["video"]!.asArray(Float.self),v)
      XCTAssertEqual(skipped["audio"]!.asArray(Float.self),a)
      // Removing cross outputs must still retain the nonzero self-attention path.
      let selfInputs=fixture(c,selfAttention:true,cross:true)
      let withoutCross=MLXAVBlock.forward(configuration:c,selfInputs,parameter:parameter,linear:linear,perturbation:.init(skipCrossModality:true))
      let selfOnly=MLXAVBlock.forward(configuration:c,fixture(c,selfAttention:true,cross:false),parameter:parameter,linear:linear)
      for name in ["video","audio"] { XCTAssertEqual(withoutCross[name]!.asArray(Float.self),selfOnly[name]!.asArray(Float.self)) }
    }
  }
}
