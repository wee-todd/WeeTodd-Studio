import AVFoundation
import Foundation
import StudioCore

extension StudioStore {
  /// AVFoundation reads the published movie asynchronously; no Python inspection subprocess.
  static func inspectNativeMovie(_ path: String) async throws -> [String: Any] {
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    let duration = try await asset.load(.duration).seconds
    let tracks = try await asset.loadTracks(withMediaType: .video)
    guard let track = tracks.first else { throw StudioError.invalid("Native renderer returned no video track.") }
    let fps = Double(try await track.load(.nominalFrameRate))
    let size = try await track.load(.naturalSize)
    let audio = try await asset.loadTracks(withMediaType: .audio)
    guard duration.isFinite, duration > 0, fps.isFinite, fps > 0, !audio.isEmpty else {
      throw StudioError.invalid("Native renderer returned an incomplete audiovisual movie.")
    }
    return ["duration": duration, "fps": fps, "width": Int(size.width), "height": Int(size.height)]
  }
}
