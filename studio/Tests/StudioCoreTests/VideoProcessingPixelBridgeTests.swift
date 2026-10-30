import Foundation
import XCTest
@testable import StudioCore

final class VideoProcessingPixelBridgeTests:XCTestCase {
  func testLosslessMOVBridgePreservesPixelsAndIsReadableByPixelMediaAdmission() async throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer{try? FileManager.default.removeItem(at:root)}
    let fixture=try VideoProcessingFixture(root:root),source=try fixture.source(),plan=try fixture.plan(),cancel=NativeHeadlessCancellation()
    let media=try FlashVSRMediaInfo.inspect(URL(fileURLWithPath:source.path),runtime:plan.runtime,directory:root,cancellation:cancel)
    let (canonical,checked,rgb)=try VideoProcessingSequenceExecutor.canonical(source,media:media,output:root.appendingPathComponent("canonical.mkv"),runtime:plan.runtime,directory:root,cancellation:cancel)
    let bridge=try await VideoProcessingSequenceExecutor.pixelBridge(canonical,media:checked,runtime:plan.runtime,directory:root,cancellation:cancel)
    let actual=try await NativeMovieFrozenMedia.videoClock(URL(fileURLWithPath:bridge.path))
    XCTAssertEqual(actual.times.count,24);XCTAssertEqual(actual.fps,24,accuracy:0.00001)
    XCTAssertEqual(try VideoProcessingSequenceExecutor.hashMedia(bridge,video:true,runtime:plan.runtime,directory:root,cancellation:cancel),rgb)
  }
}
