import AVFoundation
import AppKit
import StudioCore
import XCTest
@testable import WeeToddStudio

final class TimelinePreviewQualificationTests: XCTestCase {
  @MainActor func testRealProjectProducesVisiblePlayerPixels() async throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_PREVIEW_TEST_PROJECT"] else {
      throw XCTSkip("Optional real-project decoder qualification")
    }
    let project = try ProjectStorage.read(URL(fileURLWithPath: path))
    let media = try await TimelinePlaybackBuilder.build(TimelinePlaybackPlan(project: project))
    XCTAssertTrue(media.unavailableClips.isEmpty, "\(media.unavailableClips)")
    if let composition = media.videoComposition {
      let valid = try await composition.isValid(for: media.composition, timeRange: CMTimeRange(start: .zero, duration: media.composition.duration), validationDelegate: nil)
      XCTAssertTrue(valid)
    }
    let item = AVPlayerItem(asset: media.composition)
    item.videoComposition = media.videoComposition
    item.audioMix = media.audioMix
    let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    item.add(output)
    let player = AVPlayer(playerItem: item)
    defer { player.pause() }
    for _ in 0..<200 where item.status == .unknown { try await Task.sleep(nanoseconds: 20_000_000) }
    XCTAssertEqual(item.status, .readyToPlay, item.error?.localizedDescription ?? "")
    for second in [0.5, 26.0, 63.0].filter({ $0 < project.duration }) {
      await player.seek(to: TimelinePlaybackBuilder.time(second), toleranceBefore: .zero, toleranceAfter: .zero)
      player.play()
      var frame: CVPixelBuffer?
      var displayed = CMTime.invalid
      for _ in 0..<100 {
        let candidate = output.copyPixelBuffer(forItemTime: player.currentTime(), itemTimeForDisplay: &displayed)
        // AVPlayer may expose the previous seek's buffered frame until the new one arrives.
        if let candidate, displayed.seconds >= second - 0.05, displayed.seconds < second + 1 {
          frame = candidate; break
        }
        try await Task.sleep(nanoseconds: 20_000_000)
      }
      player.pause()
      let buffer = try XCTUnwrap(frame, "No decoded frame at \(second)s; \(item.error?.localizedDescription ?? "no item error")")
      CVPixelBufferLockBaseAddress(buffer, .readOnly)
      let bytes = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
      let stride = CVPixelBufferGetBytesPerRow(buffer)
      var brightness = 0.0
      var count = 0
      for y in Swift.stride(from: 0, to: CVPixelBufferGetHeight(buffer), by: 8) {
        for x in Swift.stride(from: 0, to: CVPixelBufferGetWidth(buffer), by: 8) {
          let offset = y * stride + x * 4
          brightness += Double(bytes[offset]) + Double(bytes[offset + 1]) + Double(bytes[offset + 2])
          count += 3
        }
      }
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
      print("PLAYER FRAME requested=\(second)s displayed=\(displayed.seconds)s mean RGB=\(brightness / Double(count))")
      XCTAssertGreaterThan(brightness / Double(count), 3, "Black player frame at \(second)s")
    }
  }
}
