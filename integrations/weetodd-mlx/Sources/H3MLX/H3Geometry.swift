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
  /// Short still windows admit a bounded two-megapixel image without expanding video.
  public static let maximumStillCanvasPixels = 1920 * 1088

  public let canvasAdmission: H3CanvasAdmission
  public var maximumPackedRows: Int { canvasAdmission.maximumPackedRows }
  public let width: Int
  public let height: Int
  public let frames: Int
  public let videoLatentFrames: Int
  public let audioLatentFrames: Int
  public let stillImageMode: H3StillImageMode?
  public var maximumPixels: Int {
    stillImageMode == nil ? canvasAdmission.maximumPixels : Self.maximumStillCanvasPixels
  }

  public var videoRows: Int { videoLatentFrames * (height / 32) * (width / 32) }
  public var audioRows: Int { audioLatentFrames * 2 }

  public init(width: Int, height: Int, durationSeconds: Double) throws {
    try self.init(width: width, height: height, durationSeconds: durationSeconds, canvasAdmission: .ordinary)
  }

  public init(width: Int, height: Int, durationSeconds: Double,
    canvasAdmission: H3CanvasAdmission) throws {
    guard (32...4096).contains(width), (32...4096).contains(height),
      width.isMultiple(of: 32), height.isMultiple(of: 32),
      durationSeconds.isFinite, (2.5...15).contains(durationSeconds) else {
      throw H3GeometryError.invalid("H3 requires 32-pixel canvas multiples and a finite 2.5–15 second duration.")
    }
    if canvasAdmission == .spatialRefinement {
      try canvasAdmission.validate(width: width, height: height)
    }
    self.canvasAdmission = canvasAdmission
    self.stillImageMode = nil
    self.width = width
    self.height = height
    var aligned = Int((durationSeconds * 24).rounded(.toNearestOrEven))
    while aligned % 17 != 5 { aligned += 1 }
    frames = aligned
    videoLatentFrames = ((aligned - 5) / 17) * 5 + 2
    audioLatentFrames = Int((Double(aligned) / 24 * 40).rounded(.toNearestOrEven))
  }

  public init(width: Int, height: Int, stillImageMode: H3StillImageMode) throws {
    guard (32...1920).contains(width), (32...1920).contains(height),
      width.isMultiple(of: 32), height.isMultiple(of: 32),
      width * height <= Self.maximumStillCanvasPixels else {
      throw H3GeometryError.invalid("H3 stills require 32-pixel multiples, sides at most 1920 and a 1920 × 1088 pixel budget.")
    }
    self.canvasAdmission = .ordinary
    self.stillImageMode = stillImageMode
    self.width = width; self.height = height
    frames = stillImageMode.frames
    videoLatentFrames = stillImageMode.latentFrames
    audioLatentFrames = stillImageMode.audioLatentFrames
  }

  /// Exact source timeline for the explicit regeneration path. No duration or
  /// cadence resampling occurs; padded frames are separate from visible frames.
  public init(regenerateWidth width:Int,height:Int,paddedFrames:Int) throws {
    try H3CanvasAdmission.regenerate.validate(width:width,height:height)
    guard (5...362).contains(paddedFrames),(paddedFrames-5).isMultiple(of:17) else {
      throw H3GeometryError.invalid("Regeneration requires the exact 5+17*n padded source timeline.")
    }
    self.canvasAdmission = .regenerate;self.stillImageMode=nil
    self.width=width;self.height=height;frames=paddedFrames
    videoLatentFrames=(paddedFrames-5)/17*5+2
    audioLatentFrames=(try H3RegenerateAdmission.silenceSampleCount(paddedFrames:paddedFrames)+799)/800
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
    if canvasAdmission != .ordinary { try canvasAdmission.validatePackedRows(result) }
    return result
  }
}
