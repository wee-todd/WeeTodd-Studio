@preconcurrency import AVFoundation
import CoreVideo
import Foundation

public enum NativeUpscaleMovieError: LocalizedError {
  case invalid(String)
  public var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}

/// Exact source clocks and passthrough audio, shared by native movie upscalers.
public enum NativeUpscaleMovieIO {
  public static func videoContentRange(_ track: AVAssetTrack, fallback: CMTimeRange) async throws
    -> CMTimeRange
  {
    // AVAssetReader synthesizes a frame for a leading empty edit when left unrestricted.
    let media = try await track.load(.segments).filter { !$0.isEmpty }
    guard let first = media.first, let last = media.last else { return fallback }
    return CMTimeRange(start: first.timeMapping.target.start, end: last.timeMapping.target.end)
  }

  public static func finalVideoEnd(_ asset: AVAsset, track: AVAssetTrack, last: CMTime,
    editEnd: CMTime) -> CMTime
  {
    // Decoded samples can omit duration, and an MP4 edit can round its end to
    // the coarser movie timescale. Query the last compressed sample for its tail.
    guard editEnd > last, let reader = try? AVAssetReader(asset: asset) else { return editEnd }
    reader.timeRange = CMTimeRange(start: last, end: editEnd)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
    guard reader.canAdd(output) else { return editEnd }
    reader.add(output)
    guard reader.startReading() else { return editEnd }
    defer { reader.cancelReading() }
    while let sample = output.copyNextSampleBuffer() {
      let duration = sample.outputDuration
      if sample.numSamples > 0, sample.outputPresentationTimeStamp == last,
        duration.isNumeric, duration > .zero {
        return min(editEnd, last + duration)
      }
    }
    return editEnd
  }

  @discardableResult
  public static func mux(_ videoFile: URL, audio: [AVAssetTrack], sourceStart: CMTime,
    audioStart: CMTime? = nil, previewEnd: CMTime?, output: URL) async throws -> CMTime
  {
    let video = AVURLAsset(url: videoFile, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
    guard let track = try await video.loadTracks(withMediaType: .video).first else {
      throw NativeUpscaleMovieError.invalid("The encoded video could not be opened for audio muxing.")
    }
    let composition = AVMutableComposition()
    guard let movieTrack = composition.addMutableTrack(withMediaType: .video,
      preferredTrackID: kCMPersistentTrackID_Invalid) else {
      throw NativeUpscaleMovieError.invalid("Cannot allocate a video track for muxing.")
    }
    var audioSegments: [(AVAssetTrack, CMTimeRange)] = []
    for audioTrack in audio {
      let original = try await audioTrack.load(.timeRange)
      let end = previewEnd.map { min($0, original.end) } ?? original.end
      let start = max(audioStart ?? original.start, original.start)
      guard end > start else { continue }
      audioSegments.append((audioTrack, CMTimeRange(start: start, end: end)))
    }
    let origin = audioStart ?? audioSegments.reduce(sourceStart) { min($0, $1.1.start) }
    let range = try await track.load(.timeRange)
    try movieTrack.insertTimeRange(range, of: track, at: .zero)
    if sourceStart > origin {
      movieTrack.insertEmptyTimeRange(CMTimeRange(start: .zero, end: sourceStart - origin))
    }
    movieTrack.preferredTransform = try await track.load(.preferredTransform)
    for (track, range) in audioSegments {
      guard range.isValid, !range.isEmpty else { continue }
      guard let destination = composition.addMutableTrack(withMediaType: .audio,
        preferredTrackID: kCMPersistentTrackID_Invalid) else {
        throw NativeUpscaleMovieError.invalid("Cannot allocate a source audio track for muxing.")
      }
      try destination.insertTimeRange(range, of: track, at: .zero)
      if range.start > origin {
        destination.insertEmptyTimeRange(CMTimeRange(start: .zero, end: range.start - origin))
      }
    }
    guard let export = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough) else {
      throw NativeUpscaleMovieError.invalid("Cannot create an audio passthrough export session.")
    }
    if #available(macOS 15.0, *) {
      try await export.export(to: output, as: .mov)
    } else {
      export.outputURL = output
      export.outputFileType = .mov
      await export.export()
      guard export.status == .completed else {
        throw export.error ?? NativeUpscaleMovieError.invalid("Audio passthrough failed.")
      }
    }
    return sourceStart - origin
  }}
