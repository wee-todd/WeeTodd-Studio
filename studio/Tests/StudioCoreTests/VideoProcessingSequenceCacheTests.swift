import Foundation
import XCTest
@testable import StudioCore

final class VideoProcessingSequenceCacheTests:XCTestCase {
  func testEachPixelStageKeepsItsExplicitAdapterSelection() throws {
    let runtime=NativePixelCQRuntime(workerPath:"/worker",ffmpegPath:"/ffmpeg",ffprobePath:"/ffprobe",components:["pixel_spatial_adapter":"/old-pixel"],cqAdapterPath:"/old-cq")
    var settings=NativePixelCQSettings();settings.pixelAdapterPath="/new-pixel";settings.cqAdapterPath="/new-cq"
    let selected=try NativeVideoProcessingPlan.pixelRuntime(runtime,settings:settings)
    XCTAssertEqual(selected.components["pixel_spatial_adapter"],"/new-pixel");XCTAssertEqual(selected.cqAdapterPath,"/new-cq")
    XCTAssertEqual(runtime.components["pixel_spatial_adapter"],"/old-pixel")
  }

  func testStageIdentityExcludesDownstreamSettingsAndIncludesOwnResourcesAndClock() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer{try? FileManager.default.removeItem(at:root)}
    let fixture=try VideoProcessingFixture(root:root),base=try fixture.plan()
    var first=base.recipe.steps[0],second=first;first.id="first";second.id="second";second.input="first"
    let recipe=try VideoProcessingRecipe(steps:[first,second])
    let resources=try fixture.resources()
    let plan=NativeVideoProcessingPlan(recipe:recipe,runtime:base.runtime,resources:["first":resources,"second":resources])
    let media=FlashVSRMediaInfo(width:64,height:64,frames:24,fps:try .init(numerator:24,denominator:1),firstTimestamp:0,hasAudio:false,audioRate:nil,audioChannels:nil)
    let hash=String(repeating:"a",count:64)
    let key=try VideoProcessingSequenceExecutor.stageKey(plan:plan,step:first,inputRGBSHA256:hash,media:media)
    var changedRecipe=recipe;changedRecipe.steps[1].settings.decoder = .vae
    let changed=NativeVideoProcessingPlan(recipe:changedRecipe,runtime:plan.runtime,resources:plan.resources)
    XCTAssertEqual(key,try VideoProcessingSequenceExecutor.stageKey(plan:changed,step:first,inputRGBSHA256:hash,media:media))
    var own=first;own.settings.decoder = .vae
    XCTAssertNotEqual(key,try VideoProcessingSequenceExecutor.stageKey(plan:plan,step:own,inputRGBSHA256:hash,media:media))
    XCTAssertNotEqual(key,try VideoProcessingSequenceExecutor.stageKey(plan:plan,step:first,inputRGBSHA256:String(repeating:"b",count:64),media:media))
    let otherClock=FlashVSRMediaInfo(width:64,height:64,frames:24,fps:try .init(numerator:25,denominator:1),firstTimestamp:0,hasAudio:false,audioRate:nil,audioChannels:nil)
    XCTAssertNotEqual(key,try VideoProcessingSequenceExecutor.stageKey(plan:plan,step:first,inputRGBSHA256:hash,media:otherClock))
    try Data("new engine".utf8).write(to:root.appendingPathComponent("fixture-engine"))
    let updated=NativeVideoProcessingPlan(recipe:recipe,runtime:base.runtime,resources:["first":try fixture.resources(),"second":resources])
    XCTAssertNotEqual(key,try VideoProcessingSequenceExecutor.stageKey(plan:updated,step:first,inputRGBSHA256:hash,media:media))
  }
}
