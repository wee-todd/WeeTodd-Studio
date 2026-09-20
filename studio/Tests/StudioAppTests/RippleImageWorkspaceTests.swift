import AppKit
import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class RippleImageWorkspaceTests: XCTestCase {
  @MainActor private func fixture(invocation: Bridge.Invocation? = nil) throws -> (StudioStore, RippleImageContext, String) {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
    let image = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 2, pixelsHigh: 2,
      bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
      colorSpaceName: .deviceRGB, bytesPerRow: 8, bitsPerPixel: 32)!
    let url = directory.appendingPathComponent("frame.png")
    try image.representation(using: .png, properties: [:])!.write(to: url)
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: invocation)
    var clip = Clip(name: "Movie", engine: .movie)
    clip.sourcePath = "/source.mp4"; clip.duration = 5
    clip.rippleDraft = RippleDraft(clip: clip, frameRate: 24)
    clip.rippleDraft!.references[0].originalPath = url.path
    store.project.clips = [clip]; store.selectedClipID = clip.id
    let context = RippleImageContext(clipID: clip.id, draft: clip.rippleDraft!, reference: clip.rippleDraft!.references[0])
    return (store, context, url.path)
  }

  @MainActor func testFrameCanvasAndCandidateRemainSeparateFromOrdinaryWorkspace() throws {
    let (store, context, path) = try fixture()
    store.drawThingsConnections = [DrawThingsConnection(id: "local")]
    var ordinary = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: store.project.id))
    ordinary.prompt = "Ordinary work"; ordinary.profileID = "local"; ordinary.modelID = "image-editor"
    store.imageDraft = ordinary; store.imagePreviewPath = "/ordinary.png"
    let lease = ReferenceWorkspaceLease(store: store)
    let draft = store.makeRippleImageDraft(context, width: 768, height: 512, previousDraft: lease.previousDraft)
    store.imageDraft = draft
    XCTAssertEqual(draft.canvas?.path, path)
    XCTAssertEqual(draft.canvas?.enabled, true)
    XCTAssertEqual(draft.modelID, ordinary.modelID)
    XCTAssertEqual(draft.width, 768)
    XCTAssertEqual(draft.height, 512)
    XCTAssertTrue(draft.prompt.isEmpty)
    try store.adoptRippleImage(path, context: context)
    XCTAssertEqual(store.project.clips[0].rippleDraft?.references[0].path, path)
    XCTAssertEqual(store.project.clips[0].sourcePath, "/source.mp4")
    XCTAssertTrue(lease.restore(store: store))
    XCTAssertEqual(store.imageDraft, ordinary)
    XCTAssertEqual(store.imagePreviewPath, "/ordinary.png")
  }

  @MainActor func testChangedSourceOrRemovedFrameRejectsCandidate() throws {
    let (store, context, path) = try fixture()
    store.imageDraft = store.makeRippleImageDraft(context, width: 768, height: 512, previousDraft: nil)
    store.project.clips[0].sourceIn += 1
    XCTAssertThrowsError(try store.adoptRippleImage(path, context: context))
    store.project.clips[0].sourceIn -= 1
    store.project.clips[0].rippleDraft?.references = []
    XCTAssertThrowsError(try store.adoptRippleImage(path, context: context))
  }

  @MainActor func testReferenceConnectionPreferenceAndErrorsAreScoped() throws {
    let (store, context, _) = try fixture()
    store.drawThingsConnections = [DrawThingsConnection(id: "cloud")]
    store.imageWorkspaceLibrary.referenceConnectionID = "removed-local"
    var ordinary = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: store.project.id))
    ordinary.profileID = "cloud"
    let draft = store.makeRippleImageDraft(context, width: 768, height: 512, previousDraft: ordinary)
    XCTAssertEqual(draft.profileID, "")
    store.imageDraft = draft; store.error = "Unrelated error"
    let key = store.beginImageAttempt(draft)
    store.recordImageFailure("Reference failed", referenceKey: key)
    XCTAssertEqual(store.referenceImageError, "Reference failed")
    XCTAssertEqual(store.error, "Unrelated error")
    store.imageDraft = ordinary
    store.recordImageFailure("Late failure", referenceKey: key)
    XCTAssertNil(store.referenceImageError)
    XCTAssertEqual(store.error, "Unrelated error")
  }

  @MainActor func testLateGeneratedCandidateCannotLandInReopenedMovie() async throws {
    var active: StudioStore?
    let (store, context, path) = try fixture { _, _, _, _ in
      if let active { try active.replaceDocument(active.project, url: nil, isDirty: false) }
      return ["asset": ["path": "/saved-candidate.png"]]
    }
    active = store
    store.drawThingsConnections = [DrawThingsConnection(id: "local")]
    store.imageDraft = store.makeRippleImageDraft(context, width: 768, height: 512, previousDraft: nil)
    store.imageDraft?.profileID = "local"; store.imageDraft?.modelID = "test-model"
    XCTAssertEqual(store.imageDraft?.canvas?.path, path)
    await store.generateImageAsset()
    XCTAssertNil(store.error)
    XCTAssertTrue(store.project.assets.isEmpty)
    XCTAssertTrue(store.notice.contains("destination movie changed"))
  }
}
