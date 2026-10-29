import AVFoundation
import CryptoKit
import Foundation

/// Stores the selected decoded audio losslessly. Original media is never the durable voice.
public enum Drama3AudioPreparation {
  public static func prepare(
    source: URL, start: Double, duration: Double, channel: String = "mix", in root: URL
  ) async throws -> Drama3PreparedSample {
    guard start.isFinite, start >= 0, duration.isFinite, duration > 0, duration <= 600,
      ["mix", "left", "right"].contains(channel)
    else { throw Drama3Error.invalid("Choose an audio range up to 600 seconds.") }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let temporary = root.appendingPathComponent(".\(UUID().uuidString).wav")
    defer { try? FileManager.default.removeItem(at: temporary) }
    let asset = AVURLAsset(url: source)
    guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
      throw Drama3Error.invalid("The voice sample has no audio track.")
    }
    let descriptions = try await track.load(.formatDescriptions)
    let sourceChannels =
      descriptions.first.flatMap {
        CMAudioFormatDescriptionGetStreamBasicDescription($0)?.pointee.mChannelsPerFrame
      } ?? 1
    let channels = min(Int(sourceChannels), 2)
    guard channel != "right" || channels == 2 else {
      throw Drama3Error.invalid("The selected sample has no right channel.")
    }
    let reader = try AVAssetReader(asset: asset)
    reader.timeRange = CMTimeRange(
      start: CMTime(seconds: start, preferredTimescale: 44100),
      duration: CMTime(seconds: duration, preferredTimescale: 44100))
    let output = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 44100,
        AVNumberOfChannelsKey: channels, AVLinearPCMIsFloatKey: true, AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsNonInterleaved: false,
      ])
    output.alwaysCopiesSampleData = false
    reader.add(output)
    guard reader.startReading() else {
      throw reader.error ?? Drama3Error.invalid("Could not decode the voice sample.")
    }
    defer { reader.cancelReading() }
    let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
    let wanted = Int64((duration * 44100).rounded())
    var frames: Int64 = 0
    do {
      let writer = try AVAudioFile(forWriting: temporary, settings: format.settings)
      while let sample = output.copyNextSampleBuffer() {
        try Task.checkCancellation()
        guard let block = CMSampleBufferGetDataBuffer(sample) else { continue }
        let n = CMBlockBufferGetDataLength(block) / 4
        var data = [Float](repeating: 0, count: n)
        let status = data.withUnsafeMutableBytes {
          CMBlockBufferCopyDataBytes(
            block, atOffset: 0, dataLength: n * 4, destination: $0.baseAddress!)
        }
        guard status == kCMBlockBufferNoErr else {
          throw Drama3Error.invalid("Could not read sample PCM.")
        }
        let count = min(n / channels, Int(max(0, wanted - frames)))
        if count > 0 {
          let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count))!
          pcm.frameLength = AVAudioFrameCount(count)
          for i in 0..<count {
            let value: Float
            if channels == 1 || channel == "left" {
              value = data[i * channels]
            } else if channel == "right" {
              value = data[i * channels + 1]
            } else {
              value = (data[i * channels] + data[i * channels + 1]) * 0.5
            }
            guard value.isFinite else {
              throw Drama3Error.invalid("The reference contains invalid audio samples.")
            }
            pcm.floatChannelData![0][i] = value
          }
          try writer.write(from: pcm)
          frames += Int64(count)
        }
      }
    }
    guard reader.status == .completed, frames > 0, abs(frames - wanted) <= 1 else {
      throw reader.error
        ?? Drama3Error.invalid(
          "The selected reference range extends beyond its audio. Shorten it and match the transcript."
        )
    }
    let digest = try fileDigest(temporary.path)
    let destination = root.appendingPathComponent(digest + ".wav")
    if !FileManager.default.fileExists(atPath: destination.path) {
      try FileManager.default.moveItem(at: temporary, to: destination)
    } else if try fileDigest(destination.path) != digest {
      throw Drama3Error.invalid(
        "Retained audio is damaged. Restore or remove the damaged sample before saving this voice again."
      )
    }
    return Drama3PreparedSample(
      path: destination.path, duration: Double(frames) / 44100, sha256: digest)
  }
  public static func fileDigest(_ path: String) throws -> String {
    let h = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
    defer { try? h.close() }
    var hash = SHA256()
    while true {
      try Task.checkCancellation()
      let consumed = try autoreleasepool {
        let bytes = try h.read(upToCount: 1024 * 1024) ?? Data()
        hash.update(data: bytes)
        return bytes.count
      }
      if consumed == 0 { break }
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
public struct Drama3PreparedSample: Sendable {
  public let path: String
  public let duration: Double
  public let sha256: String
}
