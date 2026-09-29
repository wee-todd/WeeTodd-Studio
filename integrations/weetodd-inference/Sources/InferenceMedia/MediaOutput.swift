import Foundation
import Darwin
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

public enum MediaOutputError: Error { case invalid(String) }

/// Shared lossless frame/audio publication, independent of inference backend.
public enum MediaOutput {
  private static func newFile(_ url: URL) throws -> FileHandle {
    let fd = Darwin.open(url.path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, mode_t(0o600))
    guard fd >= 0 else { throw MediaOutputError.invalid("Cannot create new file \(url.lastPathComponent): \(String(cString: strerror(errno)))") }
    return FileHandle(fileDescriptor: fd, closeOnDealloc: true)
  }

  public static func writePNG(_ rgb: [Float], width: Int, height: Int, to url: URL) throws {
    guard (1...16384).contains(width), (1...16384).contains(height), rgb.count == width * height * 3, rgb.allSatisfy(\.isFinite),
      !FileManager.default.fileExists(atPath: url.path) else { throw MediaOutputError.invalid("Invalid or existing PNG output.") }
    let bytes = Data(rgb.map { UInt8((min(1, max(-1, $0)) + 1) * 127.5) })
    let png = NSMutableData()
    guard let provider = CGDataProvider(data: bytes as CFData), let color = CGColorSpace(name: CGColorSpace.sRGB),
      let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
        bytesPerRow: width * 3, space: color, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
      let destination = CGImageDestinationCreateWithData(png, UTType.png.identifier as CFString, 1, nil) else {
      throw MediaOutputError.invalid("Cannot create RGB PNG output.")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw MediaOutputError.invalid("Cannot finalize RGB PNG output.") }
    try (png as Data).write(to: url, options: .withoutOverwriting)
  }

  /// IEEE Float32 stereo WAV with explicit fact sample count. H3's native
  /// audio rate is 32 kHz; LTX's is 48 kHz. The count comes from the decoder.
  public static func writeWAV(samples: [Float], sampleRate: Int, channels: Int, to url: URL) throws {
    let frames = samples.count / 2
    guard channels == 2, [32_000, 48_000].contains(sampleRate), frames > 0,
      samples.count == frames * 2, samples.allSatisfy(\.isFinite),
      samples.count <= (Int(UInt32.max) - 50) / 4 else { throw MediaOutputError.invalid("Invalid or oversized stereo WAV.") }
    let count = frames, dataBytes = UInt32(samples.count * 4)
    var header = Data("RIFF".utf8)
    header.appendLE(dataBytes + 50); header.append(Data("WAVEfmt ".utf8)); header.appendLE(UInt32(18))
    header.appendLE(UInt16(3)); header.appendLE(UInt16(2)); header.appendLE(UInt32(sampleRate))
    header.appendLE(UInt32(sampleRate * 8)); header.appendLE(UInt16(8)); header.appendLE(UInt16(32)); header.appendLE(UInt16(0))
    header.append(Data("fact".utf8)); header.appendLE(UInt32(4)); header.appendLE(UInt32(count))
    header.append(Data("data".utf8)); header.appendLE(dataBytes)
    let file = try newFile(url)
    defer { try? file.close() }
    try file.write(contentsOf: header)
    for start in stride(from: 0, to: count, by: 4096) {
      try Task.checkCancellation()
      var bytes = Data(); bytes.reserveCapacity(min(4096, count - start) * 8)
      for sample in start..<min(count, start + 4096) {
        bytes.appendLE(samples[sample].bitPattern)
        bytes.appendLE(samples[count + sample].bitPattern)
      }
      try file.write(contentsOf: bytes)
    }
    try file.synchronize()
  }
}

private extension Data {
  mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
    var little = value.littleEndian
    Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
  }
}
