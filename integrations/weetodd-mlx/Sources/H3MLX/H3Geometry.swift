import Foundation

public enum H3GeometryError: Error, Equatable {
  case invalid(String)
}

/// H3's fixed 24 fps / 40 Hz audiovisual grid. This describes shapes only;
/// memory admission and model loading happen after the complete request passes.
public struct H3Geometry: Sendable {
  /// Include the released 1 MP, 16:9 canvas on H3's 32-pixel grid.
  /// Packed-row admission separately bounds duration and conditioning.
  public static let maximumCanvasPixels = 1376 * 768

  public let width: Int
  public let height: Int
  public let frames: Int
  public let videoLatentFrames: Int
  public let audioLatentFrames: Int

  public var videoRows: Int { videoLatentFrames * (height / 32) * (width / 32) }
  public var audioRows: Int { audioLatentFrames * 2 }

  public init(width: Int, height: Int, durationSeconds: Double) throws {
    guard (32...4096).contains(width), (32...4096).contains(height),
      width.isMultiple(of: 32), height.isMultiple(of: 32),
      durationSeconds.isFinite, (2.5...15).contains(durationSeconds) else {
      throw H3GeometryError.invalid("H3 requires 32-pixel canvas multiples and a finite 2.5–15 second duration.")
    }
    self.width = width
    self.height = height
    var aligned = Int((durationSeconds * 24).rounded(.toNearestOrEven))
    while aligned % 17 != 5 { aligned += 1 }
    frames = aligned
    videoLatentFrames = ((aligned - 5) / 17) * 5 + 2
    audioLatentFrames = Int((Double(aligned) / 24 * 40).rounded(.toNearestOrEven))
  }

  public func packedRows(textRows: Int, conditionVideoRows: Int,
    conditionAudioRows: Int) throws -> Int {
    guard textRows > 0, conditionVideoRows >= 0, conditionAudioRows >= 0 else {
      throw H3GeometryError.invalid("H3 requires text rows and nonnegative conditioning rows.")
    }
    var result = 0
    for value in [textRows, conditionVideoRows, conditionAudioRows, audioRows, videoRows] {
      let (sum, overflow) = result.addingReportingOverflow(value)
      guard !overflow else { throw H3GeometryError.invalid("H3 packed row count overflows Int.") }
      result = sum
    }
    return result
  }
}
