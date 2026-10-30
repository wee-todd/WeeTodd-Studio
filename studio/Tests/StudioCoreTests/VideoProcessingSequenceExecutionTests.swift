import Foundation
import InferenceContracts
import XCTest
@testable import StudioCore

private actor SequenceFixture {
  let runtime:FlashVSRRuntime,resources:FlashVSRResources,h3:H3RegenerateResources,pixel:NativePixelCQResources
  var counts=[String:Int](),failOnce:String?,waiting:String?
  init(root:URL,failOnce:String?=nil,waiting:String?=nil) throws {
    self.failOnce=failOnce;self.waiting=waiting
    let base=try VideoProcessingFixture(root:root),plan=try base.plan();runtime=plan.runtime;resources=try base.resources()
    let file=resources.engineFiles[0]
    func c(_ role:NativeComponentRole)->H3RegenerateJob.Component{.init(path:file.path,identity:.init(family:.h3,role:role,sourceSHA256:file.sha256,layout:"fixture"))}
    h3 = .init(components:.init(transformer:c(.transformer),refiner:c(.refiner),sharpCurve:c(.sharpCurve),sharpInput:c(.sharpInput),latentUpscaler:c(.latentUpscaler),videoVAE:c(.videoVAE),audioVAE:c(.audioVAE)),qwen:.init(checkpointRoot:root.path,tokenizer:.init(path:file.path,sha256:file.sha256),files:[.init(path:file.path,sha256:file.sha256)]),modelFiles:[file],engineFiles:[file],profile:file,runtime:.init(ffmpegPath:runtime.ffmpegPath,ffprobePath:runtime.ffprobePath,workerPath:"/usr/bin/true"))
    pixel = .init(runtime:.init(workerPath:"/usr/bin/true",ffmpegPath:runtime.ffmpegPath,ffprobePath:runtime.ffprobePath,components:[:],cqAdapterPath:nil),workerFiles:[file],componentIdentitySHA256:file.sha256)
  }
  func plan(changeLast:Bool=false) throws -> NativeVideoProcessingPlan {
    var flash=FlashVSRSettings();flash.backend = .swiftMLX;flash.width=128;flash.height=128
    var first=VideoProcessingStep(settings:flash);first.id="flash"
    var h3Settings=H3RegenerateSettings();h3Settings.targetWidth=256;h3Settings.targetHeight=256;h3Settings.encoderProfile = .nativePosteriorMeanF16;h3Settings.posteriorSeed=nil
    var middle=VideoProcessingStep(h3Regenerate:h3Settings);middle.id="h3";middle.input="flash"
    var pixelSettings=NativePixelCQSettings();if changeLast{pixelSettings.cqStrength=0.75}
    var last=VideoProcessingStep(pixelCQ:pixelSettings);last.id="pixel";last.input="h3"
    return .init(recipe:try .init(steps:[first,middle,last]),runtime:runtime,resources:[first.id:resources],h3Runtime:h3.runtime,h3Resources:[middle.id:h3],pixelResources:[last.id:pixel])
  }
  func resume(){waiting=nil}
  func run(step:VideoProcessingStep,input:FlashVSRFile,media:FlashVSRMediaInfo,directory:URL) async throws -> VideoProcessingStageExecution {
    counts[step.id,default:0]+=1
    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
    if waiting==step.id{try await Task.sleep(nanoseconds:30_000_000_000)}
    if failOnce==step.id{failOnce=nil;throw StudioError.invalid("Fixture stage failed")}
    let width=step.isNative ? media.width*2:step.isH3Regenerate ? step.h3Regenerate!.targetWidth:step.settings.width
    let height=step.isNative ? media.height*2:step.isH3Regenerate ? step.h3Regenerate!.targetHeight:step.settings.height
    let movie=directory.appendingPathComponent("restored.mkv")
    try VideoProcessingFixture.ffmpeg(["-i",input.path,"-map","0:v:0","-an","-vf","scale=\(width):\(height):flags=neighbor","-c:v","ffv1","-pix_fmt","bgr0","-fps_mode","passthrough",movie.path])
    let file=try FlashVSRFile.capture(movie),receiptURL=directory.appendingPathComponent("receipt.json")
    let payload:VideoProcessingStagePayload
    if step.isFlashVSR {
      var request=VideoUpscaleRequest(sourcePath:input.path,settings:step.settings,runtime:runtime);request.sourceSHA256=input.sha256
      let plan=FlashVSRResolvedPlan(request:request,source:input,media:media,components:resources.components,engineFiles:resources.engineFiles,frames:media.frames,temporaryBytes:1)
      let receipt=FlashVSRReceipt(requestFingerprint:try plan.fingerprint,plan:plan,video:file,restored:file,audio:nil,elapsedSeconds:0.01)
      payload = .flashVSR(receipt);try JSONEncoder().encode(receipt).write(to:receiptURL)
    } else if step.isH3Regenerate {
      let request=H3RegenerateRequest(sourcePath:input.path,settings:step.h3Regenerate!,runtime:h3.runtime)
      let padded=5+((max(0,media.frames-5)+16)/17)*17
      let plan=H3RegenerateResolvedPlan(request:request,source:input,media:media,resources:h3,frames:media.frames,paddedFrames:padded,sourceWidth:media.width,sourceHeight:media.height,temporaryBytes:1,preparationPolicy:"fixture")
      let metadata=directory.appendingPathComponent("metadata.json");try Data("{}".utf8).write(to:metadata)
      let rows=directory.appendingPathComponent("rows.f32");try Data([0,0,0,0]).write(to:rows)
      let receipt=H3RegenerateReceipt(requestFingerprint:try plan.fingerprint,plan:plan,source:input,video:file,restored:file,audio:nil,workerResult:try .capture(metadata),workerJob:try .capture(metadata),refinedLatents:try .capture(rows),refinedLatentsShape:[1,1,96],elapsedSeconds:0.01)
      payload = .h3Regenerate(receipt);try JSONEncoder().encode(receipt).write(to:receiptURL)
    } else {
      let metadata=directory.appendingPathComponent("metadata.json");try Data("{}".utf8).write(to:metadata)
      let actual=FlashVSRMediaInfo(width:width,height:height,frames:media.frames,fps:media.fps,firstTimestamp:0,hasAudio:false,audioRate:nil,audioChannels:nil)
      let receipt=NativePixelCQReceipt(requestFingerprint:"fixture",source:input,video:file,workerMetadata:try .capture(metadata),workerRecipe:try .capture(metadata),media:actual,elapsedSeconds:0.01,sourceAudioSHA256:nil,outputAudioSHA256:nil,restored:file)
      payload = .pixelCQ(receipt);try JSONEncoder().encode(receipt).write(to:receiptURL)
    }
    return .init(payload:payload,restored:file,receiptFile:try .capture(receiptURL))
  }
}
final class VideoProcessingSequenceExecutionTests:XCTestCase {
  private func root() throws -> URL {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);addTeardownBlock{try? FileManager.default.removeItem(at:root)};return root
  }
  func testThreeFamilyFailureResumeAndDownstreamChangePreservePrefixAndPCM() async throws {
    let root=try root(),fixture=try SequenceFixture(root:root,failOnce:"pixel"),source=try VideoProcessingFixture(root:root).source()
    let plan=try await fixture.plan(),cache=root.appendingPathComponent("cache")
    let worker:VideoProcessingSequenceExecutor.Worker={step,input,media,directory in try await fixture.run(step:step,input:input,media:media,directory:directory)}
    do{_ = try await NativeVideoProcessingExecutor.run(plan:plan,source:source,directory:cache,sequenceWorker:worker);XCTFail("Expected final stage failure")}
    catch{XCTAssertTrue(error.localizedDescription.contains("Fixture"),error.localizedDescription)}
    let resumed=try await NativeVideoProcessingExecutor.run(plan:plan,source:source,directory:cache,sequenceWorker:worker)
    XCTAssertEqual(resumed.reusedSteps,2);XCTAssertEqual(resumed.receipt.frames,24);XCTAssertEqual(resumed.receipt.fps.value,24)
    let receipt=try XCTUnwrap(resumed.receipt.sequence)
    XCTAssertEqual(receipt.sourcePCMHash,receipt.outputPCMHash);XCTAssertNotNil(receipt.sourcePCMHash)
    XCTAssertEqual(receipt.stages.map{ $0.receipt.payload.kind },["flashvsr-turbo","h3-regenerate","native-pixel-cq"])
    let read=try VideoProcessingReceipt.read(from:URL(fileURLWithPath:resumed.receiptPath));XCTAssertEqual(read.video,resumed.receipt.video)
    let reused=try await NativeVideoProcessingExecutor.run(plan:plan,source:source,directory:cache,sequenceWorker:worker)
    XCTAssertEqual(reused.reusedSteps,3);XCTAssertEqual(reused.receipt.video,resumed.receipt.video)
    let changed=try await NativeVideoProcessingExecutor.run(plan:fixture.plan(changeLast:true),source:source,directory:cache,sequenceWorker:worker)
    XCTAssertEqual(changed.reusedSteps,2)
    let counts=await fixture.counts;XCTAssertEqual(counts,["flash":1,"h3":1,"pixel":3]);try source.verify()
  }
  func testDownstreamGeometryAndPixelClockRejectBeforePrefixWorker() async throws {
    for mode in ["h3-large-source","pixel-90fps","pixel-off-grid","pixel-wide","pixel-tall"] {
      let root=try root(),fixture=try SequenceFixture(root:root,failOnce:"flash"),base=try await fixture.plan()
      let source=try VideoProcessingFixture(root:root).source(fps:mode == "pixel-90fps" ? 90:24)
      var first=base.recipe.steps[0]
      var next=base.recipe.steps[mode == "h3-large-source" ? 1:2];next.input=first.id
      if mode == "h3-large-source" {first.settings.width=1280;first.settings.height=1024;next.h3Regenerate!.targetWidth=2560;next.h3Regenerate!.targetHeight=2048}
      if mode == "pixel-off-grid" {first.settings.width=130}
      if mode == "pixel-wide" {first.settings.width=2560;first.settings.height=1440}
      if mode == "pixel-tall" {first.settings.width=1440;first.settings.height=2560}
      let plan=NativeVideoProcessingPlan(recipe:try .init(steps:[first,next]),runtime:base.runtime,resources:base.resources,h3Runtime:base.h3Runtime,h3Resources:base.h3Resources,pixelResources:base.pixelResources)
      let worker:VideoProcessingSequenceExecutor.Worker={step,input,media,directory in try await fixture.run(step:step,input:input,media:media,directory:directory)}
      do {_ = try await NativeVideoProcessingExecutor.run(plan:plan,source:source,directory:root.appendingPathComponent("cache"),sequenceWorker:worker);XCTFail("Known downstream constraint must reject")}
      catch {XCTAssertFalse(error.localizedDescription.contains("Fixture stage failed"),mode+": "+error.localizedDescription)}
      let counts=await fixture.counts;XCTAssertTrue(counts.isEmpty,mode)
    }
  }
  func testPixelSourceAxisLimitIncludes2048Boundary() throws {
    let recipe=try VideoProcessingRecipe(steps:[VideoProcessingStep(pixelCQ:NativePixelCQSettings())])
    let media=FlashVSRMediaInfo(width:2048,height:2048,frames:24,fps:try .init(numerator:24,denominator:1),firstTimestamp:0,hasAudio:false,audioRate:nil,audioChannels:nil)
    XCTAssertNoThrow(try VideoProcessingSequenceExecutor.validateGeometry(recipe:recipe,sourceMedia:media))
  }
  func testDownstreamH3RejectsSourceClockBeforeAnyWorker() async throws {
    let root=try root(),fixture=try SequenceFixture(root:root),movie=root.appendingPathComponent("source-30.mkv")
    try VideoProcessingFixture.ffmpeg(["-f","lavfi","-i","testsrc2=size=64x64:rate=30:duration=1","-an","-c:v","ffv1",movie.path])
    let plan=try await fixture.plan(),source=try FlashVSRFile.capture(movie)
    let worker:VideoProcessingSequenceExecutor.Worker={step,input,media,directory in try await fixture.run(step:step,input:input,media:media,directory:directory)}
    do {_ = try await NativeVideoProcessingExecutor.run(plan:plan,source:source,directory:root.appendingPathComponent("cache"),sequenceWorker:worker);XCTFail("Downstream H3 cannot change source FPS")}
    catch {XCTAssertTrue(error.localizedDescription.contains("24 FPS"),error.localizedDescription)}
    let counts=await fixture.counts;XCTAssertTrue(counts.isEmpty)
  }
  func testCancellationRestartsOnlyInterruptedStatefulStage() async throws {
    let root=try root(),fixture=try SequenceFixture(root:root,waiting:"h3"),source=try VideoProcessingFixture(root:root).source()
    let plan=try await fixture.plan(),cache=root.appendingPathComponent("cache")
    let worker:VideoProcessingSequenceExecutor.Worker={step,input,media,directory in try await fixture.run(step:step,input:input,media:media,directory:directory)}
    let task=Task{try await NativeVideoProcessingExecutor.run(plan:plan,source:source,directory:cache,sequenceWorker:worker)}
    for _ in 0..<300 {if await fixture.counts["h3"] != nil{break};try await Task.sleep(nanoseconds:20_000_000)}
    task.cancel();do{_ = try await task.value;XCTFail("Cancellation ignored")}catch{XCTAssertTrue(error is CancellationError)}
    await fixture.resume()
    let result=try await NativeVideoProcessingExecutor.run(plan:plan,source:source,directory:cache,sequenceWorker:worker)
    XCTAssertEqual(result.reusedSteps,1);let counts=await fixture.counts
    XCTAssertEqual(counts,["flash":1,"h3":2,"pixel":1])
  }
}
