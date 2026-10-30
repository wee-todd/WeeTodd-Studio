import CryptoKit
import Foundation
import XCTest
import MLX
import InferenceContracts
@testable import FlashVSRMLX

final class FlashVSRPipelineTests:XCTestCase {
  private func recipe(frames:Int=5,compression:Int=4,width:Int=64,height:Int=64,decoder:NativeFlashVSRRecipe.Decoder = .tiny,root:URL=URL(fileURLWithPath:"/qualification"),hash:String=String(repeating:"a",count:64))throws->NativeFlashVSRRecipe {
    try .init(sourceID:String(repeating:"b",count:64),inputRGBPath:root.appendingPathComponent("source.rgb").path,
      inputSHA256:hash,outputRGBPath:root.appendingPathComponent("result.rgb").path,scratchDirectory:root.appendingPathComponent("scratch").path,
      modelRoot:root.appendingPathComponent("models").path,width:width,height:height,frames:frames,fpsNumerator:30000,fpsDenominator:1001,
      temporalCompression:compression,variant:.turbo,decoder:decoder,seed:42,colorFix:.off)
  }
  func testTemporalCompressionPreservesUniqueFirstAndEOFOnlyPadding()throws {
    let four=try FlashVSRTemporalPlan(recipe:recipe(frames:3,compression:4))
    XCTAssertEqual(four.effectiveFrames,5);XCTAssertEqual(four.expandedFrames,5);XCTAssertEqual(four.groupCount,1)
    XCTAssertEqual(try four.projectionSourceIndices(group:0),[0,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
    let two=try FlashVSRTemporalPlan(recipe:recipe(frames:4,compression:2))
    XCTAssertEqual(two.expandedFrames,9)
    XCTAssertEqual((0..<9).map { two.sourceIndex(expandedIndex:$0) },[0,1,1,2,2,3,3,3,3])
    let one=try FlashVSRTemporalPlan(recipe:recipe(frames:4,compression:1))
    XCTAssertEqual(one.expandedFrames,13);XCTAssertEqual(one.gaussianWeights.count,4)
    XCTAssertEqual(one.gaussianWeights[0],0.13447072,accuracy:1e-7)
    XCTAssertEqual(one.gaussianWeights[1],0.36552929,accuracy:1e-7)
    let long=try FlashVSRTemporalPlan(recipe:recipe(frames:30,compression:4))
    XCTAssertEqual(long.groupCount,3);XCTAssertEqual(try long.decoderExpandedRange(group:0),0..<13)
    XCTAssertEqual(try long.decoderExpandedRange(group:1),13..<29);XCTAssertEqual(try long.decoderExpandedRange(group:2),29..<45)
  }
  func testGaussianMergerKeepsPaddingDistinctAndAcceptsSplitModelGroups()throws {
    let plan=try FlashVSRTemporalPlan(recipe:recipe(frames:4,compression:2)),merge=FlashVSRTemporalMerger(plan:plan)
    var values:[Float]=[],indices:[Int]=[]
    // Frame0 stays unique; real frame3 uses positions5/6 only. Padded effective
    // frame4 positions7/8 may read the same source, but must not enter frame3.
    let frames=MLXArray([Float(0),1,3,4,6,7,9,100,200,250],[10,1,1,1])
    let rgb=repeated(frames,count:3,axis:3)
    try merge.consume(offset:0,frames:rgb[0..<3]) { index,value in indices.append(index);values.append(value[0,0,0].item(Float.self)) }
    try merge.consume(offset:3,frames:rgb[3..<7]) { index,value in indices.append(index);values.append(value[0,0,0].item(Float.self)) }
    try merge.consume(offset:7,frames:rgb[7..<10]) { index,value in indices.append(index);values.append(value[0,0,0].item(Float.self)) }
    try merge.finish()
    XCTAssertEqual(indices,[0,1,2,3]);XCTAssertEqual(values,[0,2,5,8])
    XCTAssertThrowsError(try merge.consume(offset:10,frames:rgb[0..<1]) { _,_ in })
  }
  func testMemoryAdmissionIncludesAllCachesAndWanDiskCarry()throws {
    let tiny=try FlashVSRMemoryPlan(recipe:recipe(frames:100,width:2048,height:2048))
    XCTAssertGreaterThan(tiny.stageBytes["decoder"]!,UInt64(4*1024*1024*1024))
    XCTAssertGreaterThan(tiny.stageBytes["transformer"]!,UInt64(8*1024*1024*1024))
    XCTAssertGreaterThan(tiny.scratchBytes,tiny.outputBytes)
    XCTAssertThrowsError(try tiny.validate(maximumActiveBytes:1024*1024*1024,availableDiskBytes:UInt64.max,physicalMemoryBytes:128*1024*1024*1024))
    let wan=try FlashVSRMemoryPlan(recipe:recipe(frames:3,width:128,height:128,decoder:.wan))
    XCTAssertGreaterThan(wan.wanCarryDiskBytes,0)
    XCTAssertThrowsError(try wan.validate(maximumActiveBytes:UInt64.max,availableDiskBytes:155000000,physicalMemoryBytes:128*1024*1024*1024))
    XCTAssertThrowsError(try wan.validate(maximumActiveBytes:UInt64.max,availableDiskBytes:0,physicalMemoryBytes:128*1024*1024*1024))
    _ = try JSONDecoder().decode(FlashVSRMemoryPlan.self,from:JSONEncoder().encode(wan))
  }
  func testPreflightRejectsWrongSourceBeforeLoadingModelsAndLeavesNoArtifacts()throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    try Data([0]).write(to:root.appendingPathComponent("source.rgb"))
    let value=try recipe(root:root)
    XCTAssertThrowsError(try FlashVSRPipeline.preflight(recipe:value))
    XCTAssertFalse(FileManager.default.fileExists(atPath:value.outputRGBPath))
    XCTAssertFalse(FileManager.default.fileExists(atPath:value.scratchDirectory))
  }
  func testInstalledNativeTurboTinyPipelinePreservesRationalClockAndStageRelease()throws {
    guard let models=ProcessInfo.processInfo.environment["WEETODD_FLASHVSR_PIPELINE_MODELS"],
      let outputRoot=ProcessInfo.processInfo.environment["WEETODD_FLASHVSR_PIPELINE_OUTPUT"] else { throw XCTSkip("Installed native end-to-end qualification is opt-in.") }
    let root=URL(fileURLWithPath:outputRoot,isDirectory:true).appendingPathComponent(UUID().uuidString,isDirectory:true)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    var input=Data();for value:UInt8 in [64,96,128] { input.append(Data(repeating:value,count:128*128*3)) }
    let inputURL=root.appendingPathComponent("source.rgb");try input.write(to:inputURL)
    let hash=SHA256.hash(data:input).map { String(format:"%02x",$0) }.joined()
    let value=try NativeFlashVSRRecipe(sourceID:hash,inputRGBPath:inputURL.path,inputSHA256:hash,
      outputRGBPath:root.appendingPathComponent("output.rgb").path,scratchDirectory:root.appendingPathComponent("scratch").path,
      modelRoot:models,width:64,height:64,frames:3,fpsNumerator:30000,fpsDenominator:1001,temporalCompression:4,
      variant:.turbo,decoder:.tiny,seed:42,colorFix:.off)
    var events:[String]=[]
    let result=try FlashVSRPipeline.run(recipe:value) { events.append($0.stage) }
    print("NATIVE FLASH PIPELINE RESULT "+String(data:try JSONEncoder().encode(result),encoding:.utf8)!)
    XCTAssertEqual(result.visibleFrames,3);XCTAssertEqual(result.fpsNumerator,30000);XCTAssertEqual(result.fpsDenominator,1001)
    XCTAssertEqual(result.inputSHA256,hash)
    XCTAssertEqual(try Data(contentsOf:URL(fileURLWithPath:value.outputRGBPath)).count,3*64*64*3)
    XCTAssertEqual(result.outputSHA256,SHA256.hash(data:try Data(contentsOf:URL(fileURLWithPath:value.outputRGBPath))).map { String(format:"%02x",$0) }.joined())
    let ordered=["projection_released","transformer_released","decoder_released"]
    XCTAssertEqual(events.filter { ordered.contains($0) },ordered)
    XCTAssertEqual(Set(result.releasedStageActiveBytes.keys),["projection","transformer","decoder"])
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:value.scratchDirectory),[])
    let cancelledOutput=root.appendingPathComponent("cancelled.rgb")
    let cancelled=try NativeFlashVSRRecipe(sourceID:hash,inputRGBPath:inputURL.path,inputSHA256:hash,
      outputRGBPath:cancelledOutput.path,scratchDirectory:value.scratchDirectory,modelRoot:models,width:64,height:64,
      frames:3,fpsNumerator:30000,fpsDenominator:1001,temporalCompression:4,variant:.turbo,decoder:.tiny,seed:42,colorFix:.off)
    var cancelEvents:[String]=[]
    XCTAssertThrowsError(try FlashVSRPipeline.run(recipe:cancelled) { event in
      cancelEvents.append(event.stage)
      if event.stage=="decoder" && event.completed==1 { throw CancellationError() }
    }) { XCTAssertTrue($0 is CancellationError) }
    XCTAssertEqual(cancelEvents.filter { ordered.contains($0) },ordered)
    XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath:root.path).contains { $0.hasSuffix(".partial") })
    XCTAssertTrue(FileManager.default.fileExists(atPath:value.outputRGBPath))
    XCTAssertFalse(FileManager.default.fileExists(atPath:cancelledOutput.path))
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:value.scratchDirectory),[])
    _ = try JSONDecoder().decode(FlashVSRPipeline.Result.self,from:JSONEncoder().encode(result))
  }
}
