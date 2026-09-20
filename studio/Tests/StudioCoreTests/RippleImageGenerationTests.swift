import Foundation
import XCTest
@testable import StudioCore

final class RippleImageGenerationTests: XCTestCase {
  func testFrameEditorIdentityCannotMoveToAnotherFrameOrSource() throws {
    var clip = Clip(engine: .movie)
    clip.sourcePath = "/source.mov"; clip.sourceIn = 2; clip.duration = 5
    var ripple = RippleDraft(clip: clip, frameRate: 24)
    ripple.references[0].frame = 37
    ripple.references[0].originalPath = "/frame.png"
    clip.rippleDraft = ripple
    let context = RippleImageContext(clipID: clip.id, draft: ripple, reference: ripple.references[0])
    XCTAssertTrue(context.matches(clip))
    clip.rippleDraft?.references[0].frame = 38
    XCTAssertFalse(context.matches(clip))
    clip.rippleDraft = ripple; clip.sourceIn = 3
    XCTAssertFalse(context.matches(clip))
  }

  func testFrameEditorsHaveSeparateSavedWorkspacesFromOrdinaryImagesAndEachOther() throws {
    var clip = Clip(engine: .movie); clip.sourcePath = "/source.mov"
    let ripple = RippleDraft(clip: clip, frameRate: 24)
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .clip, projectID: UUID(), owner: clip.id))
    let ordinary = draft.storageKey
    draft.rippleReference = RippleImageContext(clipID: clip.id, draft: ripple, reference: ripple.references[0])
    let first = draft.storageKey
    XCTAssertNotEqual(first, ordinary)
    draft.rippleReference = RippleImageContext(clipID: clip.id, draft: ripple, reference: RippleReference(frame: 12))
    XCTAssertNotEqual(draft.storageKey, first)
    XCTAssertEqual(try JSONDecoder().decode(DrawThingsImageDraft.self, from: JSONEncoder().encode(draft)), draft)
  }
}
