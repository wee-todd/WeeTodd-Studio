import Foundation
import XCTest
@testable import StudioCore

final class ClipContinuityTests: XCTestCase {
  func testLegacyClipAndVersionDecodeWithoutContinuity() throws {
    let clip = Clip()
    let encoded = try JSONEncoder().encode(clip)
    XCTAssertNil(try JSONDecoder().decode(Clip.self, from: encoded).continuity)
    let version = RenderVersion(path: "/movie.mp4", seed: 1, prompt: "", recipePath: "")
    XCTAssertNil(try JSONDecoder().decode(RenderVersion.self, from: JSONEncoder().encode(version)).continuationArtifact)
  }

  func testSourceSelectionOnlyPermitsEarlierClips() throws {
    var project = StudioProject()
    let a = Clip(), b = Clip()
    var c = Clip(); c.continuity = ClipContinuity(mode: "frame")
    project.clips = [a, b, c]
    XCTAssertEqual(try project.continuitySource(for: c)?.id, b.id)
    c.continuity?.sourceClipID = a.id
    XCTAssertEqual(try project.continuitySource(for: c)?.id, a.id)
    c.continuity?.sourceClipID = c.id
    XCTAssertThrowsError(try project.continuitySource(for: c))
    var first = a; first.continuity = ClipContinuity(mode: "motion", sourceClipID: c.id)
    XCTAssertThrowsError(try project.continuitySource(for: first))
    first.continuity?.sourceClipID = nil
    XCTAssertThrowsError(try project.continuitySource(for: first))
  }

  func testDependencyTracksVisibleTrimAcceptedTakePathAndArtifact() throws {
    var project = StudioProject()
    var source = Clip(engine: .h3)
    source.sourcePath = "/old.mp4"
    source.versions = [RenderVersion(path: source.sourcePath, seed: 1, prompt: "", recipePath: "")]
    var next = Clip(engine: .h3); next.continuity = ClipContinuity(mode: "motion")
    project.clips = [source, next]
    let original = project.continuityDependencyFingerprint(for: next)
    project.clips[0].sourceIn = 1
    XCTAssertNotEqual(original, project.continuityDependencyFingerprint(for: next))
    project.clips[0] = source; project.clips[0].duration += 1
    XCTAssertNotEqual(original, project.continuityDependencyFingerprint(for: next))
    project.clips[0] = source; project.clips[0].sourcePath = "/new.mp4"
    XCTAssertNotEqual(original, project.continuityDependencyFingerprint(for: next))
    project.clips[0] = source; project.clips[0].versions[0].id = UUID()
    XCTAssertNotEqual(original, project.continuityDependencyFingerprint(for: next))
    project.clips[0] = source
    project.clips[0].versions[0].continuationArtifact = ContinuationArtifact(manifest: "/context/manifest.json", manifestSHA256: "a", payloadSHA256: "b")
    XCTAssertNotEqual(original, project.continuityDependencyFingerprint(for: next))
    XCTAssertTrue(project.shouldSaveContinuityContext(for: source))
    let saving = project.continuityDependencyFingerprint(for: source)
    project.clips[1].continuity = nil
    XCTAssertFalse(project.shouldSaveContinuityContext(for: source))
    XCTAssertNotEqual(saving, project.continuityDependencyFingerprint(for: source))
  }

  func testDependencyTracksSourceFileReplacement() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: url) }
    try Data([1]).write(to: url)
    var source = Clip(); source.sourcePath = url.path
    var next = Clip(); next.continuity = ClipContinuity(mode: "frame")
    var project = StudioProject(); project.clips = [source, next]
    let old = project.continuityDependencyFingerprint(for: next)
    try Data([1, 2]).write(to: url)
    XCTAssertNotEqual(old, project.continuityDependencyFingerprint(for: next))
  }

  func testArtifactPathAndHashRoundTrip() throws {
    var project = StudioProject(); var clip = Clip()
    var version = RenderVersion(path: "Media/movie.mp4", seed: 1, prompt: "", recipePath: "")
    version.continuationArtifact = ContinuationArtifact(manifest: "Media/context/manifest.json", manifestSHA256: "abc", payloadSHA256: "def")
    clip.versions = [version]; project.clips = [clip]
    let data = try JSONEncoder().encode(project)
    XCTAssertTrue(String(decoding: data, as: UTF8.self).contains("manifest_sha256"))
    project = try JSONDecoder().decode(StudioProject.self, from: data)
    ProjectStorage.mapPaths(&project) { $0.isEmpty ? $0 : "/project/" + $0 }
    XCTAssertEqual(project.clips[0].versions[0].continuationArtifact?.manifest, "/project/Media/context/manifest.json")
    XCTAssertEqual(project.clips[0].versions[0].continuationArtifact?.manifestSHA256, "abc")
  }
}

extension ClipContinuityTests {
  func testPortableArtifactCollectionPreservesManifestAndSiblingPayload() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let source = directory.appendingPathComponent("original")
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    let manifest = Data("{\"payload\":{\"file\":\"latents.safetensors\"}}".utf8)
    try manifest.write(to: source.appendingPathComponent("manifest.json"))
    try Data([1, 2, 3]).write(to: source.appendingPathComponent("latents.safetensors"))
    let original = ContinuationArtifact(manifest: source.appendingPathComponent("manifest.json").path, manifestSHA256: "a", payloadSHA256: "b")
    let target = directory.appendingPathComponent("portable")
    let copied = try ProjectStorage.collectContinuationArtifact(original, to: target)
    XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: copied.manifest)), manifest)
    XCTAssertEqual(try Data(contentsOf: target.appendingPathComponent("latents.safetensors")), Data([1, 2, 3]))
    XCTAssertEqual(copied.manifestSHA256, original.manifestSHA256)
    XCTAssertEqual(copied.payloadSHA256, original.payloadSHA256)
  }
}

extension ClipContinuityTests {
  func testH3FractionalDurationWithinTerminalFrameAcceptsContextButTrimmedFrameRejects() throws {
    let movie = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try Data([0]).write(to: movie)
    defer { try? FileManager.default.removeItem(at: movie) }
    var source = Clip(engine: .h3)
    source.sourcePath = movie.path
    source.duration = 3.04
    source.versions = [RenderVersion(path: movie.path, seed: 1, prompt: "", recipePath: "",
      usableSourceIn: 0, usableDuration: 73.0 / 24,
      continuationArtifact: ContinuationArtifact(manifest: "/context/manifest.json", manifestSHA256: "a", payloadSHA256: "b"))]
    var next = Clip(engine: .h3); next.continuity = ClipContinuity(mode: "motion")
    var project = StudioProject(); project.clips = [source, next]
    XCTAssertTrue(project.continuityIssues(for: next).isEmpty)
    project.clips[0].duration = 72.0 / 24
    XCTAssertTrue(project.continuityIssues(for: next).contains { $0.contains("untrimmed terminal frame") })
  }

  func testH3SaveContextPreferenceStaysInactiveForLTXEngine() {
    var source = Clip(engine: .h3)
    source.continuity = ClipContinuity(saveContext: true)
    var project = StudioProject(); project.clips = [source]
    XCTAssertTrue(project.shouldSaveContinuityContext(for: source))
    source.selectGenerationEngine(.ltx25)
    project.clips[0] = source
    XCTAssertFalse(project.shouldSaveContinuityContext(for: source))
    XCTAssertEqual(source.continuity?.saveContext, true)
  }
}

extension ClipContinuityTests {
  func testIndependentClipsWithoutContextPreserveEmptyDependency() {
    var project = StudioProject()
    for engine in [Engine.h3, .ltx23, .ltx25, .drawThings] {
      var clip = Clip(engine: engine)
      project.clips = [clip]
      XCTAssertEqual(project.continuityDependencyFingerprint(for: clip), "")
      clip.continuity = ClipContinuity()
      project.clips = [clip]
      XCTAssertEqual(project.continuityDependencyFingerprint(for: clip), "")
    }
  }
}
