import Foundation
import XCTest
@testable import StudioCore

final class VideoProcessingSequenceJobTests:XCTestCase {
  func testVersion3RoundTripKeepsGenerationIdentityIndependentOfDownstreamH3AndCodec() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer{try? FileManager.default.removeItem(at:root)}
    let fixture=try VideoProcessingFixture(root:root),source=try fixture.source()
    var project=StudioProject();project.settings.width=64;project.settings.height=64
    var clip=Clip(engine:.movie);clip.sourcePath=source.path;clip.duration=1
    var h3=H3RegenerateSettings();h3.targetWidth=128;h3.targetHeight=128
    clip.videoProcessing = .init(h3Regenerate:h3);project.clips=[clip]
    let runtime=H3RegenerateRuntime(ffmpegPath:"/opt/homebrew/bin/ffmpeg",ffprobePath:"/opt/homebrew/bin/ffprobe",workerPath:"/usr/bin/true")
    var job=try NativeHeadlessJob(project:project,recipes:[:],workers:[:],ffmpeg:runtime.ffmpegPath,h3ProcessingRuntime:runtime)
    let identity=try job.generationIdentity(),url=root.appendingPathComponent("job.json")
    try job.write(to:url)
    let wrapper=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as! [String:Any]
    XCTAssertEqual(wrapper["format"] as? String,NativeHeadlessJob.sequenceFormat)
    let reread=try NativeHeadlessJob.read(from:url);XCTAssertEqual(reread.h3ProcessingRuntime,runtime)
    job.project.clips[0].videoProcessing!.steps[0].h3Regenerate!.seed=99
    job.h3ProcessingRuntime!.workerPath="/different/downstream/worker";job.project.settings.format = .proRes
    XCTAssertEqual(try job.generationIdentity(),identity)
    var wrong=wrapper;wrong["format"]=NativeHeadlessJob.processingFormat
    let old=root.appendingPathComponent("wrong-version.json");try JSONSerialization.data(withJSONObject:wrong).write(to:old)
    XCTAssertThrowsError(try NativeHeadlessJob.read(from:old))
  }
  func testCapturedDestinationRejectsDocumentClipAndSelectedSourceMutation() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer{try? FileManager.default.removeItem(at:root)}
    let fixture=try VideoProcessingFixture(root:root),source=try fixture.source(),base=try fixture.plan()
    var project=StudioProject(),clip=Clip(engine:.movie);clip.sourcePath=source.path;clip.videoProcessing=base.recipe;project.clips=[clip]
    let token=try VideoProcessingAcceptanceToken(project:project,clipID:clip.id)
    let media=FlashVSRMediaInfo(width:128,height:128,frames:24,fps:try .init(numerator:24,denominator:1),firstTimestamp:0,hasAudio:false,audioRate:nil,audioChannels:nil)
    let plan=FlashVSRResolvedPlan(request:.init(sourcePath:source.path,settings:base.recipe.steps[0].settings,runtime:base.runtime),source:source,media:media,components:base.resources["restore"]!.components,engineFiles:[],frames:24,temporaryBytes:1)
    let receipt=FlashVSRReceipt(requestFingerprint:try plan.fingerprint,plan:plan,video:source,restored:source,audio:nil,elapsedSeconds:0)
    let result=NativeVideoProcessingResult(source:source,recipe:base.recipe,receipt:receipt,receiptPath:root.appendingPathComponent("receipt.json").path,reusedSteps:0)
    var changed=project;changed.id=UUID()
    XCTAssertThrowsError(try VideoProcessingAcceptance.apply(result,to:clip.id,project:&changed,expected:token));XCTAssertTrue(changed.clips[0].versions.isEmpty)
    changed=project;changed.clips[0].prompt="edited while running"
    XCTAssertThrowsError(try VideoProcessingAcceptance.apply(result,to:clip.id,project:&changed,expected:token))
    changed=project;changed.clips[0].sourcePath="/new/source.mov"
    XCTAssertThrowsError(try VideoProcessingAcceptance.apply(result,to:clip.id,project:&changed))
    XCTAssertNoThrow(try VideoProcessingAcceptance.apply(result,to:clip.id,project:&project,expected:token))
    XCTAssertEqual(project.clips[0].versions.count,1)
  }
}
