import Foundation

/// A decoded source tail followed by generated frames on the same causal grid.
/// Publication omits every context frame; the accepted source keeps its own copy.
public struct LTX25ExtensionWindow: Sendable {
  public let contextFrames: Int
  public let additionalFrames: Int
  public let geometry: AVGeometry

  public var totalFrames: Int { geometry.frames }
  public var contextLatentFrames: Int { (contextFrames - 1) / 8 + 1 }
  public var outputRange: Range<Int> { contextFrames..<totalFrames }
  public var additionalDuration: Double { Double(additionalFrames) / geometry.fps }

  public init(contextFrames: Int, additionalFrames: Int,
    width: Int, height: Int, fps: Double) throws {
    guard (9...241).contains(contextFrames), (contextFrames - 1) % 8 == 0,
      (8...720).contains(additionalFrames), additionalFrames % 8 == 0,
      width % 64 == 0, height % 64 == 0,
      fps.isFinite, Double(contextFrames) / fps <= 20 else {
      throw LTXError.invalid("LTX 2.5 extension needs 8n+1 context frames in 9–241, a positive multiple of eight additional frames, a 64-pixel two-stage grid, and at most 20 seconds of source audio.")
    }
    let geometry = try AVGeometry(width: width, height: height,
      frames: contextFrames + additionalFrames, fps: fps)
    self.contextFrames = contextFrames
    self.additionalFrames = additionalFrames
    self.geometry = geometry
  }

  public func sourceRange(sourceFrames: Int) throws -> Range<Int> {
    guard sourceFrames >= contextFrames else {
      throw LTXError.invalid("LTX 2.5 extension source must contain at least its context window.")
    }
    return (sourceFrames - contextFrames)..<sourceFrames
  }
}
