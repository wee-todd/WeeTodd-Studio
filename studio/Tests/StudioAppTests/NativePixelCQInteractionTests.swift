import Foundation
import XCTest
@testable import StudioCore
@testable import WeeToddStudio

@MainActor final class NativePixelCQInteractionTests: XCTestCase {
  func testAppendingFlashPreservesSavedNativePrefixAndStrengths() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var clip = Clip(engine: .h3)
    var native = VideoProcessingRecipe(pixelCQ: .init()); native.steps[0].id = "upscale"
    native.steps[0].pixelCQ?.cqStrength = 0.65
    clip.videoProcessing = native; store.project.clips = [clip]; store.select(clip.id)
    store.setVideoProcessingEngine("native-pixel-cq-flashvsr")
    let chain = try XCTUnwrap(store.selectedClip?.videoProcessing)
    XCTAssertEqual(chain.nativePrefix?.steps, native.steps)
    XCTAssertTrue(chain.isSequence)
    XCTAssertEqual(chain.steps[1].input, "upscale")
    XCTAssertNoThrow(try chain.validate())
  }
  func testH3AndLTXKeepGenerationWhenNativeStrengthsChangeAndSaveReopen() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    for engine in [Engine.h3, .ltx23, .ltx25] {
      var clip = Clip(engine: engine); clip.sourcePath = "/accepted/original.mp4"
      store.project.clips = [clip]; store.select(clip.id)
      let key = store.generationRequestKey(for: clip)
      store.setVideoProcessingEnabled(true)
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.count, 1)
      store.setVideoProcessingEngine("native-pixel-cq-flashvsr")
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.map(\.kind), ["native-pixel-cq", "flashvsr-turbo"])
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.last?.settings.width, 1920)
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.last?.settings.height, 1080)
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.last?.settings.decoder, .tiny)
      store.editPixelCQProcessing { $0.pixelStrength = 0.8; $0.cqStrength = 0.6 }
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.first?.pixelCQ?.pixelStrength, 0.8)
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.first?.pixelCQ?.cqStrength, 0.6)
      XCTAssertEqual(store.generationRequestKey(for: store.selectedClip!), key)
      XCTAssertEqual(store.videoProcessingOriginal(for: store.selectedClip!).path, clip.sourcePath)
      store.editVideoProcessing { $0.decoder = .vae }
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.first?.pixelCQ?.cqStrength, 0.6)
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.last?.settings.decoder, .vae)
      XCTAssertEqual(store.generationRequestKey(for: store.selectedClip!), key)
      let reopened = try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(store.project))
      XCTAssertEqual(reopened.clips[0].videoProcessing, store.selectedClip?.videoProcessing)
      store.setVideoProcessingEngine("flashvsr-turbo")
      XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.first?.settings.decoder, .tiny)
      store.setVideoProcessingEngine("native-pixel-cq-flashvsr")
      XCTAssertEqual(store.generationRequestKey(for: store.selectedClip!), key)
      XCTAssertEqual(store.project.clips[0].sourcePath, clip.sourcePath)
    }
  }
}
