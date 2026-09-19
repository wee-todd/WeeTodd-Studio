import XCTest
@testable import StudioCore

final class MusicVideoProductionTests: XCTestCase {
  func testProductionMetadataPersistsWithoutChangingInputIdentity() throws {
    var project = StudioProject(); project.clips = [Clip(name: "Shot")]
    let original = try project.productionInputFingerprint()
    project.production = MusicVideoProduction(jobDirectory: "/job", inputFingerprint: original)
    XCTAssertEqual(try project.productionInputFingerprint(), original)
    let restored = try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(project))
    XCTAssertEqual(restored.production?.jobDirectory, "/job")
    project.clips[0].prompt = "Changed"
    XCTAssertNotEqual(try project.productionInputFingerprint(), original)
  }
  func testApplyingTakesRequiresUnchangedProjectAndPreservesSong() throws {
    var project = StudioProject(); project.clips = [Clip(name: "Shot")]
    project.audio = [AudioRegion(assetID: UUID(), path: "/song.wav")]
    let fingerprint = try project.productionInputFingerprint()
    project.production = MusicVideoProduction(jobDirectory: "/job", inputFingerprint: fingerprint)
    var resolved = project
    resolved.clips[0].sourcePath = "/take.mp4"
    resolved.clips[0].versions = [RenderVersion(path: "/take.mp4", seed: 42, prompt: "", recipePath: "/recipe")]
    let song = project.audio
    try project.applyProduction(resolved, expectedFingerprint: fingerprint)
    XCTAssertEqual(project.clips[0].sourcePath, "/take.mp4")
    XCTAssertEqual(project.audio, song)
    XCTAssertTrue(project.production?.applied == true)
    XCTAssertThrowsError(try project.applyProduction(resolved, expectedFingerprint: fingerprint))
  }
  func testApplyRejectsLateOrReorderedResults() throws {
    var project = StudioProject(); project.clips = [Clip(name: "One"), Clip(name: "Two")]
    let fingerprint = try project.productionInputFingerprint()
    var resolved = project; resolved.clips.reverse()
    XCTAssertThrowsError(try project.applyProduction(resolved, expectedFingerprint: fingerprint))
    resolved = project; project.name = "New intent"
    XCTAssertThrowsError(try project.applyProduction(resolved, expectedFingerprint: fingerprint))
  }
}
