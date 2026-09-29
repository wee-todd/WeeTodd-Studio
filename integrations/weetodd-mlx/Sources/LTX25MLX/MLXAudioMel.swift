import AVFoundation
import Foundation
import MLX
import LTX25Engine

/// Released LTX audio-encoder front end: centered reflect-padded Hann STFT,
/// magnitude spectrum, and area-normalized Slaney mel bands at 16 kHz.
public enum MLXAudioMel {
  private static let fftSize = 1024
  private static let hop = 160
  private static let bands = 64
  private static func hzToMel(_ hz: Double) -> Double {
    hz < 1000 ? 3 * hz / 200 : 15 + 27 * log(hz / 1000) / log(6.4)
  }
  private static func melToHz(_ mel: Double) -> Double {
    mel < 15 ? 200 * mel / 3 : 1000 * exp((mel - 15) * log(6.4) / 27)
  }
  private static func filterbank() -> MLXArray {
    let end = hzToMel(8000)
    let points = (0..<(bands + 2)).map { melToHz(Double($0) * end / Double(bands + 1)) }
    var values = [Float](repeating: 0, count: (fftSize / 2 + 1) * bands)
    for frequency in 0...(fftSize / 2) {
      let hz = Double(frequency) * 16000 / Double(fftSize)
      for band in 0..<bands {
        let lower = points[band], center = points[band + 1], upper = points[band + 2]
        let slope = hz <= center ? max(0, (hz - lower) / (center - lower))
          : max(0, (upper - hz) / (upper - center))
        values[frequency * bands + band] = Float(slope * 2 / (upper - lower))
      }
    }
    return MLXArray(values, [fftSize / 2 + 1, bands])
  }
  private static func reflected(_ index: Int, count: Int) -> Int {
    if index < 0 { return -index }
    if index >= count { return 2 * count - index - 2 }
    return index
  }
  public static func encode(wav: URL) throws -> MLXArray {
    guard wav.isFileURL else { throw LTXError.invalid("Audio conditioning must use a local WAV file.") }
    let file = try AVAudioFile(forReading: wav)
    let format = file.processingFormat
    guard format.sampleRate == 16000, format.channelCount == 2,
      format.commonFormat == .pcmFormatFloat32, !format.isInterleaved,
      (513...321600).contains(file.length) else {
      throw LTXError.invalid("Audio conditioning WAV must be finite 16 kHz stereo Float32 PCM of at most 20 seconds.")
    }
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length))!
    try file.read(into: buffer)
    guard buffer.frameLength == file.length, let channels = buffer.floatChannelData else {
      throw LTXError.invalid("Audio conditioning WAV could not be read completely.")
    }
    let count = Int(buffer.frameLength)
    let planar = Array(UnsafeBufferPointer(start: channels[0], count: count))
      + Array(UnsafeBufferPointer(start: channels[1], count: count))
    return try encode(planar: planar, sampleRate: 16000)
  }
  /// Planar stereo [left, right] Float32 PCM. This is a bounded conditioning
  /// stage; the original publication waveform stays outside this transform.
  public static func encode(planar: [Float], sampleRate: Int) throws -> MLXArray {
    guard sampleRate == 16000, planar.count % 2 == 0,
      (fftSize / 2 + 1...321600).contains(planar.count / 2),
      planar.allSatisfy(\.isFinite) else {
      throw LTXError.invalid("LTX audio mel input requires finite 16 kHz stereo PCM of at least 513 samples and at most 20 seconds.")
    }
    try Task.checkCancellation()
    let samples = planar.count / 2, frames = samples / hop + 1
    let window = (0..<fftSize).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(fftSize))) }
    let basis = filterbank()
    var channels: [MLXArray] = []
    for channel in 0..<2 {
      try Task.checkCancellation()
      var framed = [Float](repeating: 0, count: frames * fftSize)
      for frame in 0..<frames {
        let offset = frame * hop - fftSize / 2
        for bin in 0..<fftSize {
          framed[frame * fftSize + bin] = planar[channel * samples + reflected(offset + bin, count: samples)] * window[bin]
        }
      }
      let spectrum = abs(MLXFFT.rfft(MLXArray(framed, [frames, fftSize])))
      let mel = log(maximum(matmul(spectrum, basis), 1e-5))
      eval(mel); channels.append(mel)
    }
    let result = stacked(channels, axis: 0).expandedDimensions(axis: 0)
    eval(result)
    return result
  }
}
