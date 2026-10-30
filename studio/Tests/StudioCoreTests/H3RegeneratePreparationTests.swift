import Foundation
import InferenceContracts
import XCTest
@testable import StudioCore

final class H3RegeneratePreparationTests: XCTestCase {
  func testCatalogAndPreinstalledAssetsUseAbsentOnlyFallback() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
    let alternate=root.appendingPathComponent("catalog.bin");try Data([1]).write(to:alternate)
    XCTAssertEqual(try H3RegeneratePreparation.installedAsset(root:root,relativePaths:["installed.bin","catalog.bin"]),alternate)
    let preferred=root.appendingPathComponent("installed.bin")
    try FileManager.default.createDirectory(at:preferred,withIntermediateDirectories:false)
    XCTAssertThrowsError(try H3RegeneratePreparation.installedAsset(root:root,relativePaths:["installed.bin","catalog.bin"]))
    try FileManager.default.removeItem(at:preferred);try Data([2]).write(to:preferred)
    XCTAssertEqual(try H3RegeneratePreparation.installedAsset(root:root,relativePaths:["installed.bin","catalog.bin"]),preferred)
  }
  func testPublisherPaddedResizeKeepsIndependentAxesForDefault2K() throws {
    let media=FlashVSRMediaInfo(width:1280,height:720,frames:124,fps:try .init(numerator:24,denominator:1),firstTimestamp:0,hasAudio:false,audioRate:nil,audioChannels:nil)
    var settings=H3RegenerateSettings();settings.targetWidth=2560;settings.targetHeight=1440
    for fit in [FlashVSRSettings.Fit.fit,.fill] {
      settings.fit=fit
      let grid=try H3RegeneratePreparation.geometry(media:media,settings:settings,firstFrame:0,frameCount:nil)
      XCTAssertEqual(grid.sourceWidth,1280);XCTAssertEqual(grid.sourceHeight,736)
      XCTAssertEqual(grid.frames,124);XCTAssertEqual(grid.paddedFrames,124)
      // These serialized legacy names select publisher source preparation.
      // The learned target resize is intentionally independent on each axis.
      XCTAssertNotEqual(settings.targetWidth*grid.sourceHeight,settings.targetHeight*grid.sourceWidth)
      XCTAssertEqual(settings.targetWidth,2560);XCTAssertEqual(settings.targetHeight,1440)
    }
  }
  func testClockAndSourceGridRejectBeforeAssetDiscovery() throws {
    var settings = H3RegenerateSettings(); settings.targetWidth = 1280; settings.targetHeight = 768
    let wrong = FlashVSRMediaInfo(width: 640, height: 360, frames: 120,
      fps: try .init(numerator: 30, denominator: 1), firstTimestamp: 0, hasAudio: false, audioRate: nil, audioChannels: nil)
    XCTAssertThrowsError(try H3RegeneratePreparation.geometry(media: wrong, settings: settings, firstFrame: 0, frameCount: nil)) {
      XCTAssertTrue($0.localizedDescription.contains("24 FPS"))
    }
    let valid = FlashVSRMediaInfo(width: 640, height: 360, frames: 120,
      fps: try .init(numerator: 24, denominator: 1), firstTimestamp: 0, hasAudio: false, audioRate: nil, audioChannels: nil)
    let grid = try H3RegeneratePreparation.geometry(media: valid, settings: settings, firstFrame: 3, frameCount: 22)
    XCTAssertEqual(grid.sourceWidth, 640); XCTAssertEqual(grid.sourceHeight, 384)
    XCTAssertEqual(grid.frames, 22); XCTAssertEqual(grid.paddedFrames, 22)
    XCTAssertThrowsError(try H3RegeneratePreparation.geometry(media: valid, settings: settings, firstFrame: 119, frameCount: 2))
    settings.targetWidth = 2560
    XCTAssertThrowsError(try H3RegeneratePreparation.geometry(media: valid, settings: settings, firstFrame: 0, frameCount: 22))
  }
  func testProfileRejectsDistilledOrConditionalAmbiguityBeforeWeights() throws {
    let ordinary: [String: Any] = ["format":"weetodd-headless-v2", "engine":"h3", "components":["task":"ref2va", "transformer":"/abs/base", "text_encoder":"/abs/text", "tokenizer":"/abs/tokenizer", "video_vae":"/abs/video", "audio_vae":"/abs/audio"], "config":[:]]
    XCTAssertNoThrow(try H3RegeneratePreparation.profileComponents(ordinary))
    for key in ["fasth3", "fasth3_omni", "vdn", "loras", "joint_refinement", "motion_fidelity"] {
      var candidate = ordinary; candidate[key] = [:]
      XCTAssertThrowsError(try H3RegeneratePreparation.profileComponents(candidate))
    }
    var candidate = ordinary; var components = candidate["components"] as! [String: Any]
    components["fun_controlnet"] = "/control"; candidate["components"] = components
    XCTAssertThrowsError(try H3RegeneratePreparation.profileComponents(candidate))
  }
  func testExplicitLatentsCannotSilentlyFallBackOrCrossIntervals() throws {
    let hash = String(repeating:"a", count:64)
    let file = FlashVSRFile(path:"/rows",sha256:hash,bytes:12*2*2*96*4)
    let latent = H3RegenerateLatentSource(file:file,sourceSHA256:hash,firstFrame:0,visibleFrames:39,paddedFrames:39,width:64,height:64)
    XCTAssertNoThrow(try latent.validate(sourceSHA256:hash,firstFrame:0,frames:39))
    XCTAssertThrowsError(try latent.validate(sourceSHA256:hash,firstFrame:1,frames:39))
    XCTAssertThrowsError(try latent.validate(sourceSHA256:String(repeating:"b",count:64),firstFrame:0,frames:39))
    var settings = H3RegenerateSettings(); settings.sourceMode = .movie
    XCTAssertThrowsError(try H3RegeneratePreparation.validateLatentSelection(latent,settings:settings,sourceSHA256:hash,firstFrame:0,frames:39))
  }
}
