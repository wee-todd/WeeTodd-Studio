import Foundation
import LTX25Engine

/// Describes untouched publication PCM separately from model-conditioning audio.
/// Media extraction must preserve these values; this type never resamples PCM.
public struct MLXMovieSourceAudioContract: Sendable {
  public let videoFrames: Int, fps: Double
  public let publicationSampleRate: Int, publicationSamples: Int
  public let sourceChannels: Int, publicationChannels: Int = 2
  public let sourceSupplied: Bool
  public let videoSeconds: Double, audioSeconds: Double, driftSeconds: Double
  public var conditioningSampleRate: Int { 16000 }
  public var duplicateMono: Bool { sourceChannels == 1 }
  public init(videoFrames: Int, fps: Double, sampleRate: Int? = nil,
    samples: Int? = nil, channels: Int? = nil, maximumDriftSeconds: Double = 0.05) throws {
    guard videoFrames > 0, fps.isFinite, (1...60).contains(fps),
      maximumDriftSeconds.isFinite, (0...0.5).contains(maximumDriftSeconds),
      (sampleRate == nil) == (samples == nil), (samples == nil) == (channels == nil) else {
      throw LTXError.invalid("Movie audio requires a complete PCM description and finite 1–60 fps/drift settings.")
    }
    self.videoFrames = videoFrames; self.fps = fps
    videoSeconds = Double(videoFrames) / fps
    if let sampleRate, let samples, let channels {
      guard sampleRate > 0, samples > 0, (1...2).contains(channels) else {
        throw LTXError.invalid("Movie source audio must contain mono or stereo samples at a positive sample rate.")
      }
      publicationSampleRate = sampleRate; publicationSamples = samples
      sourceChannels = channels; sourceSupplied = true
    } else {
      let count = (videoSeconds * 48000).rounded(.toNearestOrEven)
      guard count.isFinite, count >= 1, count < Double(Int.max) else { throw LTXError.invalid("Movie silence duration overflows.") }
      publicationSampleRate = 48000; publicationSamples = Int(count)
      sourceChannels = 2; sourceSupplied = false
    }
    audioSeconds = Double(publicationSamples) / Double(publicationSampleRate)
    driftSeconds = abs(audioSeconds - videoSeconds)
    guard driftSeconds <= maximumDriftSeconds + 1e-9 else {
      throw LTXError.invalid("Movie source audio/video duration drift exceeds the allowed bound; source PCM will not be stretched, truncated or padded.")
    }
  }
  /// Half-open sample ranges are contiguous at shared frame boundaries.
  public func publicationSampleBounds(startFrame: Int, endFrame: Int, fps: Double) throws -> Range<Int> {
    guard startFrame >= 0, endFrame > startFrame, endFrame <= videoFrames,
      fps == self.fps else {
      throw LTXError.invalid("Movie audio interval needs ordered nonnegative frame bounds and 1–60 fps.")
    }
    func index(_ frame: Int) throws -> Int {
      let value = (Double(frame) * Double(publicationSampleRate) / fps).rounded(.toNearestOrEven)
      guard value.isFinite, value >= 0 else { throw LTXError.invalid("Movie audio sample index is invalid.") }
      return value >= Double(publicationSamples) ? publicationSamples : Int(value)
    }
    return try index(startFrame)..<index(endFrame)
  }
}
