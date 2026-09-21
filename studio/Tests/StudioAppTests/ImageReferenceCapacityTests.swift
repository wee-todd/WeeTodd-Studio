import AppKit
import StudioCore
import XCTest
@testable import WeeToddStudio

final class ImageReferenceCapacityTests: XCTestCase {
  @MainActor func testImportTenReferencesAndPreserveOverflowWhenAddingCanvas() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let pixels = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 2, pixelsHigh: 2,
      bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
      colorSpaceName: .deviceRGB, bytesPerRow: 8, bitsPerPixel: 32)!
    let file = root.appendingPathComponent("reference.png")
    try pixels.representation(using: .png, properties: [:])!.write(to: file)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.imageDraft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: store.project.id))
    store.imageDraft?.selectProvider(.nativeMLX)
    store.loadImageInputs(Array(repeating: file, count: 10), canvas: false)
    XCTAssertEqual(store.imageDraft?.moodboard.count, 10)
    store.loadImageInputs([file], canvas: true)
    XCTAssertEqual(store.imageDraft?.moodboard.count, 10)
    XCTAssertNotNil(store.imageDraft?.imageInputIssue)
    store.imageDraft?.selectProvider(.drawThings)
    XCTAssertEqual(store.imageDraft?.moodboard.count, 10)
    XCTAssertNotNil(store.imageDraft?.imageInputIssue)
  }
}
