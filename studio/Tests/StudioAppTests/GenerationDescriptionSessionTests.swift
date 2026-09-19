import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class GenerationDescriptionSessionTests: XCTestCase {
  @MainActor func testReopeningIdenticalMovieRestartsDescriptionWithoutInvalidatingRenderInputs() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    let clip = Clip(name: "Same selected shot", engine: .ltx25)
    store.project.clips = [clip]
    store.selectedClipID = clip.id
    store.generationDescriptions[clip.id] = ["studioInput": store.generationRequestKey(for: clip)]
    let request = store.generationRequestKey(for: clip)
    let task = store.generationDescriptionTaskKey(for: clip)
    let document = store.project

    try store.replaceDocument(document, url: nil, isDirty: false)

    XCTAssertEqual(store.selectedClipID, clip.id)
    XCTAssertNil(store.generationDescriptions[clip.id])
    XCTAssertEqual(store.generationRequestKey(for: clip), request)
    XCTAssertNotEqual(store.generationDescriptionTaskKey(for: clip), task,
      "A cleared description must reload even when the movie and selected clip are identical")
  }

  @MainActor func testDescriptionTaskChangesForClipInputsButNotPlaybackPosition() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.project.clips = [Clip(name: "Shot", engine: .ltx25)]
    let task = store.generationDescriptionTaskKey(for: store.project.clips[0])
    store.playhead = 2
    XCTAssertEqual(store.generationDescriptionTaskKey(for: store.project.clips[0]), task)
    store.project.clips[0].prompt = "A different physical action."
    XCTAssertNotEqual(store.generationDescriptionTaskKey(for: store.project.clips[0]), task)
  }
}
