import Foundation
import StudioCore
import ImageLayout
import XCTest
@testable import WeeToddStudio

final class MOSSModelDiscoveryTests: XCTestCase {
  @MainActor func testManualChoiceWinsOverAutomaticFillAndCannotOverwriteANewerChoice() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.imageDraft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    store.selectImageProvider(.nativeMOSS)
    let attempt = UUID(), session = store.documentSessionID, key = store.imageDraft?.storageKey
    store.mossAnalyzerSelectionAttempt = attempt
    await store.resolveMOSSModelLinks(libraries: [], legacyAnalyzer: "/automatically-filled/model")
    XCTAssertTrue(store.completeMOSSAnalyzerSelection("/manual/qwen", attempt: attempt, session: session, draftKey: key))
    XCTAssertEqual(store.mossModelLinks.analyzerDirectory, "/manual/qwen")
    store.selectMOSSAnalyzer("/newer/selection")
    XCTAssertFalse(store.completeMOSSAnalyzerSelection("/old/gemma", attempt: attempt, session: session, draftKey: key))
    XCTAssertEqual(store.mossModelLinks.analyzerDirectory, "/newer/selection")
  }
  @MainActor func testLinkingOfficialLibraryDoesNotReplaceTheSelectedQuantizedTransformer() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.restoringImageWorkspace = false
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    draft.selectProvider(.nativeMOSS); draft.mossImage?.quantizationBits = 8
    store.imageDraft = draft
    var official = MOSSImageSettings(); official.libraryDirectory = "/official/library"
    official.sharedDirectory = "/official/Shared"; official.transformerDirectory = "/official/bf16"
    store.applyMOSSComponentLinks(official)
    XCTAssertEqual(store.imageDraft?.mossImage?.quantizationBits, 8)
    XCTAssertEqual(store.imageDraft?.mossImage?.transformerDirectory, "")
    XCTAssertNil(store.mossModelLinks.transformerDirectories["sft:q8"])
    XCTAssertEqual(store.mossModelLinks.transformerDirectories["sft:bf16"], "/official/bf16")
    store.imageDraft?.mossImage?.transformerDirectory = "/custom/q8"
    store.applyMOSSComponentLinks(official)
    XCTAssertEqual(store.imageDraft?.mossImage?.transformerDirectory, "/custom/q8")
  }
  @MainActor func testPrecisionTransitionsRestoreTheMatchingTransformerWithoutPoisoningDefaults() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.restoringImageWorkspace = false
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    draft.selectProvider(.nativeMOSS); draft.mossImage?.transformerDirectory = "/manual/bf16"
    store.imageDraft = draft
    store.selectMOSSPrecision(8)
    XCTAssertEqual(store.imageDraft?.mossImage?.transformerDirectory, "")
    XCTAssertNil(store.mossModelLinks.transformerDirectories["sft:q8"])
    store.imageDraft?.mossImage?.transformerDirectory = "/manual/q8"
    store.selectMOSSPrecision(nil)
    XCTAssertEqual(store.imageDraft?.mossImage?.transformerDirectory, "/manual/bf16")
    XCTAssertEqual(store.mossModelLinks.transformerDirectories["sft:bf16"], "/manual/bf16")
    store.selectMOSSPrecision(8)
    XCTAssertEqual(store.imageDraft?.mossImage?.transformerDirectory, "/manual/q8")
  }
  @MainActor func testComponentLinksSurviveNewDraftProjectAndRelaunch() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.restoringImageWorkspace = false
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    draft.selectProvider(.nativeMOSS)
    draft.mossImage?.libraryDirectory = "/offline/model-library/MOSS-Image"
    draft.mossImage?.sharedDirectory = "/offline/model-library/MOSS-Image/Shared"
    draft.mossImage?.transformerDirectory = "/offline/model-library/MOSS-Image/SFT/transformer"
    store.imageDraft = draft
    for target in [store, StudioStore(dataDirectory: directory, restoreSession: false)] {
      target.project = StudioProject()
      target.imageDraft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: target.project.id))
      target.selectImageProvider(.nativeMOSS)
      XCTAssertEqual(target.imageDraft?.mossImage?.sharedDirectory, "/offline/model-library/MOSS-Image/Shared")
      XCTAssertEqual(target.imageDraft?.mossImage?.transformerDirectory, "/offline/model-library/MOSS-Image/SFT/transformer")
    }
  }

  @MainActor func testAnalyzerSelectionAndLegacyPreferenceSurviveRelaunch() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.imageDraft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    store.selectImageProvider(.nativeMOSS)
    await store.resolveMOSSModelLinks(libraries: [], legacyAnalyzer: "/offline/chosen-qwen")
    XCTAssertEqual(store.mossModelLinks.analyzerDirectory, "/offline/chosen-qwen")
    store.selectMOSSAnalyzer("/offline/chosen-gemma")
    let reopened = StudioStore(dataDirectory: directory, restoreSession: false)
    reopened.imageDraft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: reopened.project.id))
    reopened.selectImageProvider(.nativeMOSS)
    await reopened.resolveMOSSModelLinks(libraries: [], legacyAnalyzer: "/offline/chosen-qwen")
    XCTAssertEqual(reopened.mossModelLinks.analyzerDirectory, "/offline/chosen-gemma")
  }

  @MainActor func testInstalledLibraryAutomaticallyLinksFreshDraftWhenProvided() async throws {
    guard let root = ProcessInfo.processInfo.environment["WEETODD_MOSS_TEST_LIBRARY"] else { throw XCTSkip("Opt-in installed library metadata discovery") }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.restoringImageWorkspace = false
    store.imageDraft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    store.selectImageProvider(.nativeMOSS)
    store.imageDraft?.steps = 17; store.imageDraft?.guidance = 1.5; store.imageDraft?.mossImage?.weightCacheGB = 8
    await store.resolveMOSSModelLinks()
    XCTAssertEqual(store.imageDraft?.mossImage?.sharedDirectory, root + "/WeeTodd/MOSS-Image/Shared")
    XCTAssertEqual(store.imageDraft?.mossImage?.transformerDirectory, root + "/WeeTodd/MOSS-Image/SFT/transformer")
    XCTAssertEqual(store.mossModelLinks.analyzerDirectory, root + "/ImgGen/mlx-vlm/gemma-4-12B-it-8bit")
    XCTAssertEqual(store.mossAnalyzerFamily, .gemma4_12b)
    XCTAssertTrue(store.mossSharedTokenizerAvailable)
    XCTAssertEqual(store.imageDraft?.steps, 17); XCTAssertEqual(store.imageDraft?.guidance, 1.5)
    XCTAssertEqual(store.imageDraft?.mossImage?.weightCacheGB, 8)
    let reopened = StudioStore(dataDirectory: directory, restoreSession: false)
    reopened.imageDraft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: reopened.project.id))
    reopened.selectImageProvider(.nativeMOSS)
    XCTAssertEqual(reopened.imageDraft?.mossImage?.sharedDirectory, root + "/WeeTodd/MOSS-Image/Shared")
    XCTAssertEqual(reopened.mossModelLinks.analyzerDirectory, root + "/ImgGen/mlx-vlm/gemma-4-12B-it-8bit")
  }
}
