import AVFoundation
import Foundation

enum FlashVSRAudioVerification {
  static func verify(source: URL, decoded: URL) throws {
    let reference = try AVAudioFile(forReading: source)
    let output = try AVAudioFile(forReading: decoded)
    guard reference.length > 0, output.length > 0,
      reference.fileFormat.sampleRate == output.fileFormat.sampleRate,
      reference.fileFormat.channelCount == output.fileFormat.channelCount,
      abs(reference.length - output.length) <= 2048 else {
      throw StudioError.invalid("Delivered audio changed its selected sample interval beyond AAC encoder padding.")
    }
  }
}
