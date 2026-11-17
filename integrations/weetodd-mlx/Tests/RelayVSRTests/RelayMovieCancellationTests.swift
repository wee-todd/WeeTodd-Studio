import AVFoundation
import MLX
import XCTest
@testable import RelayVSR

final class RelayMovieCancellationTests: XCTestCase {
  func testFourTimesWriterScalesOrientationTranslation() async throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mov")
    defer { try? FileManager.default.removeItem(at: url) }
    let geometry = try RelayGeometry(width: 32, height: 24, scale: .x4)
    let writer = try RelayMovieWriter(url: url, geometry: geometry,
      transform: CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 24, ty: 0), timeScale: 24)
    defer { writer.cancelIfWriting() }
    try Device.withDefaultDevice(.cpu) {
      try writer.append(MLXArray.zeros([96, 128, 3]), at: .zero, duration: CMTime(value: 1, timescale: 24))
    }
    try await writer.finish(at: CMTime(value: 1, timescale: 24))
    let tracks = try await AVURLAsset(url: url).loadTracks(withMediaType: .video)
    let track = try XCTUnwrap(tracks.first)
    let transform = try await track.load(.preferredTransform)
    let size = try await track.load(.naturalSize)
    XCTAssertEqual(size, CGSize(width: 128, height: 96))
    XCTAssertEqual(transform, CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 96, ty: 0))
  }

  func testCancelledFinalizationCannotFinishTheMovie() async throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mov")
    defer { try? FileManager.default.removeItem(at: url) }
    let finalization = Task.detached {
      let geometry = try RelayGeometry(width: 32, height: 24, scale: .x4)
      let writer = try RelayMovieWriter(url: url, geometry: geometry, transform: .identity, timeScale: 24)
      defer { writer.cancelIfWriting() }
      try Device.withDefaultDevice(.cpu) {
        try writer.append(MLXArray.zeros([96, 128, 3]), at: .zero, duration: CMTime(value: 1, timescale: 24))
      }
      withUnsafeCurrentTask { $0?.cancel() }
      do {
        try await writer.finish(at: CMTime(value: 1, timescale: 24))
        return false
      } catch is CancellationError {
        return writer.writer.status != .completed
      }
    }
    let cancelled = try await finalization.value
    XCTAssertTrue(cancelled, "A cancelled render must not finish its movie successfully.")
  }
}
