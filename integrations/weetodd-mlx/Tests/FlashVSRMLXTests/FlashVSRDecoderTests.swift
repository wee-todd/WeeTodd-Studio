import Foundation
import XCTest
import MLX
@testable import FlashVSRMLX

final class FlashVSRDecoderTests:XCTestCase {
  func testTinyPackingHasSourceBeforeLatentAndReplicatesOnlyInitialHistory()throws {
    let rgb=MLXArray((0..<3*8*8).map(Float.init),[1,3,1,8,8])
    let latent=MLXArray((0..<16).map { Float(1000+$0) },[1,16,1,1,1])
    let packed=try FlashVSRTinyDecoder.pack(latents:latent,sourceRGB:rgb,first:true)
    XCTAssertEqual(packed.shape,[1,1,1,784])
    let values=packed.asArray(Float.self)
    XCTAssertEqual(Array(values[0..<64]),(0..<64).map(Float.init))
    XCTAssertEqual(Array(values[64..<128]),(0..<64).map(Float.init))
    XCTAssertEqual(Array(values[256..<320]),(64..<128).map(Float.init))
    XCTAssertEqual(Array(values[768..<784]),(1000..<1016).map(Float.init))
    XCTAssertThrowsError(try FlashVSRTinyDecoder.pack(latents:latent,sourceRGB:rgb,first:false))
  }
  func testTinyPackingPreservesSpatialCellAndWithinPatchOrder()throws {
    let source=MLXArray((0..<1152).map(Float.init),[1,3,1,16,24])
    let packed=try FlashVSRTinyDecoder.pack(latents:MLXArray.ones([1,16,1,2,3]),sourceRGB:source,first:true)
    XCTAssertEqual(packed.shape,[1,2,3,784])
    XCTAssertEqual(packed[0,0,0,0..<16].asArray(Float.self),[0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31])
    XCTAssertEqual(packed[0,0,1,0..<8].asArray(Float.self),[8,9,10,11,12,13,14,15])
    XCTAssertEqual(packed[0,1,2,0..<8].asArray(Float.self),[208,209,210,211,212,213,214,215])
    XCTAssertEqual(packed[0,1,2,768..<784].asArray(Float.self),[Float](repeating:1,count:16))
  }
  func testSpatialKernelOrientationAndZeroHaloFor2DAndWan3D()throws {
    let pixels=MLXArray((1...12).map(Float.init),[1,3,4,1])
    let kernel=MLXArray([Float(0),1,0,0,2,10,0,0,0],[1,1,3,3])
    let expected:[Float]=[22,34,46,8,71,84,97,20,123,136,149,32]
    XCTAssertEqual(try FlashVSRDecoderOps.conv2D(pixels,weight:kernel).asArray(Float.self),expected)
    var carry:MLXArray?
    let volume=try FlashVSRDecoderOps.causal(pixels.reshaped([1,1,3,4,1]),weight:kernel.reshaped([1,1,1,3,3]),bias:nil,cache:&carry,spatialPadding:1)
    XCTAssertEqual(volume.asArray(Float.self),expected)
  }
  func testCausalConvolutionRetainsInputsAndMatchesSegmentedExecution()throws {
    let weight=MLXArray([Float(1),10,100],[1,1,3,1,1]),bias=MLXArray([Float(0)])
    let input=MLXArray([Float(1),2,3,4],[1,4,1,1,1])
    var whole:MLXArray?,carry:MLXArray?
    let full=try FlashVSRDecoderOps.causal(input,weight:weight,bias:bias,cache:&whole,spatialPadding:0)
    var pieces:[MLXArray]=[]
    for index in 0..<4 { pieces.append(try FlashVSRDecoderOps.causal(input[0...,index..<index+1],weight:weight,bias:bias,cache:&carry,spatialPadding:0)) }
    XCTAssertEqual(full.asArray(Float.self),[100,210,321,432])
    XCTAssertEqual(concatenated(pieces,axis:1).asArray(Float.self),[100,210,321,432])
    XCTAssertEqual(carry?.asArray(Float.self),[3,4])
  }
  func testTinyMemoryBlockCarriesPreviousInputInsteadOfPreviousOutput()throws {
    let weights:[String:MLXArray]=[
      "m.conv.0.weight":MLXArray([Float(1),1],[1,2,1,1]),"m.conv.0.bias":MLXArray([Float(0)]),
      "m.conv.2.weight":MLXArray.ones([1,1,1,1]),"m.conv.2.bias":MLXArray([Float(0)]),
      "m.conv.4.weight":MLXArray.ones([1,1,1,1]),"m.conv.4.bias":MLXArray([Float(0)])]
    var carry:MLXArray?
    let value=try FlashVSRDecoderOps.memoryBlock(MLXArray([Float(1),2],[2,1,1,1]),previous:&carry,weights:weights,prefix:"m")
    let next=try FlashVSRDecoderOps.memoryBlock(MLXArray([Float(3)],[1,1,1,1]),previous:&carry,weights:weights,prefix:"m")
    XCTAssertEqual(value.asArray(Float.self),[2,5]);XCTAssertEqual(next.asArray(Float.self),[8])
    XCTAssertEqual(carry?.asArray(Float.self),[3])
  }
  func testTinyTemporalGrowOrdersEachExpansionBeforeTheNextInputFrame()throws {
    let value=MLXArray([Float(1),2,3,4],[2,2,1,1]),weight=MLXArray([Float(1),10],[2,1,1,1])
    let output=try FlashVSRDecoderOps.temporalGrow(value,weight:weight)
    XCTAssertEqual(output.shape,[4,2,1,1]);XCTAssertEqual(output.asArray(Float.self),[1,2,10,20,3,4,30,40])
  }
  func testWanAttentionMatchesTwoTokenAnalyticSoftmax()throws {
    let input=MLXArray([Float(1),0,0,1],[1,1,1,2,2])
    let gamma=MLXArray.ones([2]),identity=MLXArray([Float(1),0,0,1],[2,2,1,1])
    let qkv=concatenated([identity,identity,identity],axis:0)
    let output=try FlashVSRDecoderOps.attention(input,gamma:gamma,qkv:qkv,qkvBias:MLXArray.zeros([6]),projection:identity,projectionBias:MLXArray.zeros([2]),queryChunk:1).asArray(Float.self)
    let root=sqrt(Float(2)),p=exp(root)/(exp(root)+1)
    XCTAssertEqual(output[0],1+root*p,accuracy:1e-6);XCTAssertEqual(output[1],root*(1-p),accuracy:1e-6)
    XCTAssertEqual(output[2],root*(1-p),accuracy:1e-6);XCTAssertEqual(output[3],1+root*p,accuracy:1e-6)
  }
  func testWanNormalizationUsesReleasedMeanAndStdAndNotAnotherCodec()throws {
    let zero=MLXArray.zeros([1,16,1,1,1]),one=MLXArray.ones([1,16,1,1,1])
    let mean=try FlashVSRWanDecoder.denormalize(zero).asArray(Float.self)
    let shifted=try FlashVSRWanDecoder.denormalize(one).asArray(Float.self)
    XCTAssertEqual(mean[0],-0.7571,accuracy:1e-6);XCTAssertEqual(mean[15],-0.2921,accuracy:1e-6)
    XCTAssertEqual(shifted[0],2.0613,accuracy:1e-6);XCTAssertEqual(shifted[15],1.6239,accuracy:1e-6)
    XCTAssertThrowsError(try FlashVSRWanDecoder.denormalize(MLXArray.zeros([1,32,1,1,1])))
  }
  func testWanTemporalShuffleInterleavesExpansionWithinEachFrame()throws {
    let input=MLXArray([Float(1),2,10,20,3,4,30,40],[1,2,1,1,4])
    XCTAssertEqual(try FlashVSRWanDecoder.temporalShuffle(input).asArray(Float.self),[1,2,10,20,3,4,30,40])
    XCTAssertEqual(try FlashVSRWanDecoder.temporalShuffle(input).shape,[1,4,1,1,2])
  }
  func testWanTileStartsCoverNonMultipleTailAndOverlapWeightsStayPositive()throws {
    XCTAssertEqual(try FlashVSRWanDecoder.tileStarts(length:11,size:6,stride:3),[0,3,5])
    XCTAssertThrowsError(try FlashVSRWanDecoder.tileStarts(length:11,size:3,stride:4))
    let mask=FlashVSRWanDecoder.blendMask(height:3,width:4,top:false,bottom:true,left:false,right:true,borderHeight:2,borderWidth:2)
    XCTAssertEqual(mask.asArray(Float.self),[0.5,0.5,0.5,0.5,0.5,1,1,1,0.5,1,1,1])
  }
  func testInstalledTinyAndWanCompleteVersusSegmentedStateAndVisibleTails()throws {
    guard let configured=ProcessInfo.processInfo.environment["WEETODD_FLASHVSR_DECODER_ROOT"] else { throw XCTSkip("Installed decoder qualification is opt-in.") }
    let root=URL(fileURLWithPath:configured,isDirectory:true)
    let latentValues:[Float]=(0..<48).map { index in Float(index%7-3)*Float(0.025) }
    let latent=MLXArray(latentValues,[1,16,3,1,1])
    let rgbValues:[Float]=(0..<1728).map { index in Float(index%23)/Float(11)-Float(1) }
    let rgb=MLXArray(rgbValues,[1,3,9,8,8])
    let tiny=try FlashVSRTinyDecoder(checkpoint:root.appendingPathComponent("TCDecoder.ckpt"),dtype:.float32)
    var complete=FlashVSRTinyDecoder.Carry(sourceID:"clip"),segmented=FlashVSRTinyDecoder.Carry(sourceID:"clip")
    var all:[MLXArray]=[],parts:[MLXArray]=[]
    try tiny.decode(latents:latent,sourceRGB:rgb,sourceID:"clip",carry:&complete) { _,group in all.append(group) }
    try tiny.decode(latents:latent[0...,0...,0..<1],sourceRGB:rgb[0...,0...,0..<1],sourceID:"clip",carry:&segmented) { _,group in parts.append(group) }
    try tiny.decode(latents:latent[0...,0...,1..<3],sourceRGB:rgb[0...,0...,1..<9],sourceID:"clip",carry:&segmented) { _,group in parts.append(group) }
    let joined=concatenated(all,axis:0),split=concatenated(parts,axis:0)
    XCTAssertEqual(joined.shape,[9,8,8,3]);XCTAssertLessThan(max(abs(joined-split)).item(Float.self),1e-5)
    XCTAssertEqual(complete.emittedFrames,9);XCTAssertEqual(segmented.emittedFrames,9)
    XCTAssertThrowsError(try tiny.decode(latents:latent[0...,0...,0..<1],sourceRGB:rgb[0...,0...,0..<1],sourceID:"other",carry:&segmented) { _,_ in })
    segmented.reset(sourceID:"tail");var tail:[MLXArray]=[]
    try tiny.decode(latents:latent[0...,0...,0..<2],sourceRGB:rgb[0...,0...,0..<3],sourceID:"tail",carry:&segmented) { _,g in tail.append(g) }
    XCTAssertEqual(concatenated(tail,axis:0).shape,[3,8,8,3]);XCTAssertEqual(segmented.emittedFrames,3)
    var clock=FlashVSRTinyDecoder.Carry(sourceID:"clock"),offsets:[Int]=[],counts:[Int]=[]
    let four=MLXArray.zeros([1,16,4,1,1]),initialRGB=MLXArray.zeros([1,3,13,8,8]),laterRGB=MLXArray.zeros([1,3,16,8,8])
    try tiny.decode(latents:four,sourceRGB:initialRGB,sourceID:"clock",carry:&clock) { offset,group in offsets.append(offset);counts.append(group.shape[0]) }
    try tiny.decode(latents:four,sourceRGB:laterRGB,sourceID:"clock",carry:&clock) { offset,group in offsets.append(offset);counts.append(group.shape[0]) }
    XCTAssertEqual(offsets,[0,1,5,9,13,17,21,25]);XCTAssertEqual(counts,[1,4,4,4,4,4,4,4]);XCTAssertEqual(clock.emittedFrames,29)
    complete.reset(sourceID:"released");segmented.reset(sourceID:"released");clock.reset(sourceID:"released")
    tiny.unload();XCTAssertFalse(tiny.isLoaded)

    let wan=try FlashVSRWanDecoder(checkpoint:root.appendingPathComponent("Wan2.1_VAE.pth"),dtype:.float32)
    var full=FlashVSRWanDecoder.Carry(sourceID:"clip"),chunks=FlashVSRWanDecoder.Carry(sourceID:"clip")
    var f:[MLXArray]=[],s:[MLXArray]=[]
    try wan.decode(latents:latent,sourceID:"clip",carry:&full) { _,g in f.append(g) }
    try wan.decode(latents:latent[0...,0...,0..<1],sourceID:"clip",carry:&chunks) { _,g in s.append(g) }
    try wan.decode(latents:latent[0...,0...,1..<3],sourceID:"clip",carry:&chunks) { _,g in s.append(g) }
    let a=concatenated(f,axis:0),b=concatenated(s,axis:0)
    XCTAssertEqual(a.shape,[9,8,8,3]);XCTAssertLessThan(max(abs(a-b)).item(Float.self),1e-5)
    XCTAssertTrue(isFinite(a).all().item(Bool.self))
    chunks.reset(sourceID:"tail");var terminal:[MLXArray]=[]
    try wan.decode(latents:latent[0...,0...,0..<2],sourceID:"tail",carry:&chunks,visibleFrames:3) { _,g in terminal.append(g) }
    XCTAssertEqual(concatenated(terminal,axis:0).shape,[3,8,8,3]);XCTAssertEqual(chunks.emittedFrames,3)
    full.reset(sourceID:"released");chunks.reset(sourceID:"released")
    wan.unload();XCTAssertFalse(wan.isLoaded)

    let spool=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:spool,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:spool) }
    let tiled=try FlashVSRWanDecoder(checkpoint:root.appendingPathComponent("Wan2.1_VAE.pth"),dtype:.float32,
      tiling:FlashVSRWanDecoder.Tiling(height:2,width:2,strideHeight:1,strideWidth:1))
    let tiledLatents=MLXArray.zeros([1,16,2,2,3])
    var joinedCarry=FlashVSRWanDecoder.Carry(sourceID:"tile",cacheDirectory:spool),splitCarry=FlashVSRWanDecoder.Carry(sourceID:"tile",cacheDirectory:spool)
    var tileWhole:[MLXArray]=[],tileSplit:[MLXArray]=[]
    try tiled.decode(latents:tiledLatents,sourceID:"tile",carry:&joinedCarry) { _,group in tileWhole.append(group) }
    try tiled.decode(latents:tiledLatents[0...,0...,0..<1],sourceID:"tile",carry:&splitCarry) { _,group in tileSplit.append(group) }
    try tiled.decode(latents:tiledLatents[0...,0...,1..<2],sourceID:"tile",carry:&splitCarry) { _,group in tileSplit.append(group) }
    let combined=concatenated(tileWhole,axis:0),streamed=concatenated(tileSplit,axis:0)
    XCTAssertEqual(combined.shape,[5,16,24,3]);XCTAssertLessThan(max(abs(combined-streamed)).item(Float.self),1e-5)
    XCTAssertGreaterThan(joinedCarry.spilledBytes,0);XCTAssertEqual(joinedCarry.cachedBytes,0)
    joinedCarry.reset(sourceID:"released");splitCarry.reset(sourceID:"released");tiled.unload()
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:spool.path),[])

  }
}
