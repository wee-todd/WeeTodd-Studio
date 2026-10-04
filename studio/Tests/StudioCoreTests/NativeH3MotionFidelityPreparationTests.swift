import CryptoKit
import Foundation
import XCTest
@testable import StudioCore

final class NativeH3MotionFidelityPreparationTests:XCTestCase {
  private func clip() -> Clip {
    var clip=Clip(engine:.h3);clip.duration=2.5;clip.seed=42;clip.generationWidth=64;clip.generationHeight=32
    clip.generationSelection = .init(task:"t2v");clip.generationSelection?.steps=19;return clip
  }
  private func ordinary(_ clip:Clip) -> [String:Any] {
    ["format":"weetodd-headless-v2","engine":"h3","components":["task":"t2va","loras":[]],
      "conditioning":["version":1,"task":"t2v","inputs":[],"audio_policy":"generated"],
      "config":["width":clip.generationWidth,"height":clip.generationHeight,"duration_seconds":clip.duration,"steps":20,"seed":clip.seed,"projection_backend":"mlx","transformer_backend":"mlx"],
      "ffmpeg":"/opt/homebrew/bin/ffmpeg","prompt":"Continue the action."]
  }
  func testExplicitRepairFreezesSourceAndDoesNotRelabelOrdinaryContinuity() throws {
    let source=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".mov")
    try Data([1,2,3]).write(to:source);defer { try? FileManager.default.removeItem(at:source) }
    let clip=clip(),settings=H3MotionFidelitySettings(sourceVideo:source.path,experimentalEnabled:true)
    let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:directory) }
    try FileManager.default.createSymbolicLink(at:directory.appendingPathComponent("ffprobe"),withDestinationURL:URL(fileURLWithPath:"/usr/bin/true"))
    var baseline=ordinary(clip);baseline["ffmpeg"]=directory.appendingPathComponent("ffmpeg").path
    let recipe=try NativeH3MotionFidelityPreparation.apply(settings,to:baseline,clip:clip)
    let fields=try XCTUnwrap(recipe["motion_fidelity"] as? [String:Any])
    XCTAssertEqual(fields["source_sha256"] as? String,SHA256.hash(data:Data([1,2,3])).map { String(format:"%02x",$0) }.joined())
    XCTAssertEqual(fields["source_in"] as? Double,0);XCTAssertEqual(fields["duration_seconds"] as? Double,2.5)
    XCTAssertEqual(fields["seed"] as? Int,42);XCTAssertNil(recipe["continuation"])
    let nilRecipe=try NativeH3MotionFidelityPreparation.apply(nil,to:ordinary(clip),clip:clip)
    XCTAssertEqual(try JSONSerialization.data(withJSONObject:nilRecipe,options:[.sortedKeys]),try JSONSerialization.data(withJSONObject:ordinary(clip),options:[.sortedKeys]))
  }
  func testRepairRejectsUnalignedTrimUniformOverflowAndSharedModeConflictsBeforeSourceRead() throws {
    var clip=clip(),settings=H3MotionFidelitySettings(sourceVideo:"/missing/source.mov",experimentalEnabled:true)
    settings.sourceIn=0.01
    XCTAssertThrowsError(try NativeH3MotionFidelityPreparation.apply(settings,to:ordinary(clip),clip:clip))
    settings.sourceIn=0;settings.mode = .uniform;settings.maxFrames=73
    XCTAssertThrowsError(try NativeH3MotionFidelityPreparation.apply(settings,to:ordinary(clip),clip:clip))
    settings.mode = .adaptive;settings.maxFrames=345;clip.continuity = .init(mode:"motion")
    XCTAssertThrowsError(try NativeH3MotionFidelityPreparation.apply(settings,to:ordinary(clip),clip:clip)) { error in XCTAssertTrue(error.localizedDescription.contains("independent")) }
    clip.continuity=nil;var low=ordinary(clip);low["config"]=["steps":5]
    XCTAssertThrowsError(try NativeH3MotionFidelityPreparation.apply(settings,to:low,clip:clip)) { error in XCTAssertTrue(error.localizedDescription.contains("15 evaluations")) }
  }
  func testActualCompressedCFRClockPreservesEditedOriginalCanvas() async throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false);defer { try? FileManager.default.removeItem(at:root) }
    let ffmpeg=URL(fileURLWithPath:"/opt/homebrew/bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath:ffmpeg.path) else { throw XCTSkip("FFmpeg CFR fixture unavailable") }
    let movie=root.appendingPathComponent("source.mov")
    try NativeMovieFrozenMedia.run(ffmpeg,["-v","error","-f","lavfi","-i","color=red:s=64x32:r=24","-frames:v","72","-an","-c:v","qtrle",movie.path],log:root.appendingPathComponent("fixture.log"))
    let clip=clip();var settings=H3MotionFidelitySettings(sourceVideo:movie.path,sourceIn:0.5,experimentalEnabled:true)
    try await NativeH3MotionFidelityPreparation.inspect(settings,clip:clip)
    settings.sourceIn=13.0/24
    do { try await NativeH3MotionFidelityPreparation.inspect(settings,clip:clip);XCTFail("Must reject incomplete source interval") }
    catch { XCTAssertTrue(error.localizedDescription.contains("exact edited")) }
    var wrong=clip;wrong.generationWidth=32
    do { try await NativeH3MotionFidelityPreparation.inspect(.init(sourceVideo:movie.path,experimentalEnabled:true),clip:wrong);XCTFail("Must preserve original canvas") }
    catch { XCTAssertTrue(error.localizedDescription.contains("original generation canvas")) }
  }
}
