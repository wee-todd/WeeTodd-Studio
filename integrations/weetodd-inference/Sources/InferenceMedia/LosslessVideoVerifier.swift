import Foundation
import Darwin

/// Independently decodes the lossless publication. Frame hashes establish the
/// frame count/grid/constant clock; a whole-stream hash proves exact RGB bytes.
public enum LosslessVideoVerifier {
  public struct Proof: Codable, Sendable {
    public let frames: Int, width: Int, height: Int, fpsNumerator: Int, fpsDenominator: Int
    public let rgbSHA256: String
  }
  public static func verify(ffmpeg: URL, video: URL, width: Int, height: Int, frames: Int,
    fps: Double, expectedRGBSHA256: String? = nil) throws -> Proof {
    guard video.isFileURL, ffmpeg.isFileURL, FileManager.default.isExecutableFile(atPath: ffmpeg.path),
      (2...16384).contains(width), (2...16384).contains(height), (1...1_000_000).contains(frames), fps.isFinite, (1...240).contains(fps),
      expectedRGBSHA256.map({ $0.count == 64 && $0.allSatisfy { $0.isHexDigit && !$0.isUppercase } }) ?? true else {
      throw MediaOutputError.invalid("Invalid lossless verification contract.")
    }
    let values = try video.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
    guard values.isRegularFile == true, values.isSymbolicLink != true else { throw MediaOutputError.invalid("Lossless video must be a regular file.") }
    let root = video.deletingLastPathComponent().appendingPathComponent(".lossless-proof-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let hashes = root.appendingPathComponent("frames.sha256"), whole = root.appendingPathComponent("rgb.sha256"), log = root.appendingPathComponent("decode.log")
    let fd = Darwin.open(log.path, O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC, 0o600)
    guard fd >= 0 else { throw MediaOutputError.invalid("Cannot create lossless verification log.") }
    let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true); defer { try? handle.close() }
    let process = Process(); process.executableURL = ffmpeg
    process.arguments = ["-v","error","-nostdin","-n","-i",video.path,
      "-map","0:v:0","-an","-pix_fmt","rgb24","-fps_mode","passthrough","-f","framehash","-hash","sha256",hashes.path,
      "-map","0:v:0","-an","-pix_fmt","rgb24","-fps_mode","passthrough","-f","hash","-hash","sha256",whole.path]
    process.standardOutput = handle; process.standardError = handle
    try Task.checkCancellation(); try process.run()
    defer {
      if process.isRunning { process.terminate(); usleep(100000); if process.isRunning { kill(process.processIdentifier, SIGKILL) } }
      process.waitUntilExit()
    }
    while process.isRunning { try Task.checkCancellation(); usleep(10000) }
    guard process.terminationStatus == 0 else { throw MediaOutputError.invalid("Cannot decode lossless video for verification.") }
    guard (try whole.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 1000) < 128 else { throw MediaOutputError.invalid("Invalid RGB proof.") }
    let digest = try String(contentsOf: whole, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
    guard digest.hasPrefix("SHA256="), digest.count == 71 else { throw MediaOutputError.invalid("Missing decoded RGB hash.") }
    let rgb = String(digest.dropFirst(7))
    guard expectedRGBSHA256 == nil || expectedRGBSHA256 == rgb else { throw MediaOutputError.invalid("Lossless publication changed decoded RGB bytes.") }
    let input = try FileHandle(forReadingFrom: hashes); defer { try? input.close() }
    var buffer = Data(), count = 0, numerator = 0, denominator = 0, dimensions = false
    func line(_ bytes: Data) throws {
      guard bytes.count < 4096, let text = String(data: bytes, encoding: .utf8) else { throw MediaOutputError.invalid("Invalid frame proof.") }
      if text.hasPrefix("#tb 0: ") {
        let pair = text.dropFirst(7).split(separator: "/")
        guard pair.count == 2, let d = Int(pair[0]), let n = Int(pair[1]), n > 0, d > 0,
          abs(Double(n)/Double(d) - fps) < 0.000001 else { throw MediaOutputError.invalid("Lossless publication changed source cadence.") }
        numerator = n; denominator = d
      } else if text.hasPrefix("#dimensions 0: ") { dimensions = text == "#dimensions 0: \(width)x\(height)" }
      else if !text.hasPrefix("#") && !text.isEmpty {
        let fields = text.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        guard fields.count == 6, fields[0] == "0", Int(fields[1]) == count, Int(fields[2]) == count,
          Int(fields[3]) == 1, Int(fields[4]) == width*height*3, count < frames else {
          throw MediaOutputError.invalid("Lossless publication changed frame coverage or clock.")
        }
        count += 1
      }
    }
    while true {
      let consumed = try autoreleasepool {
        try Task.checkCancellation()
        let chunk = try input.read(upToCount: 65536) ?? Data()
        buffer.append(chunk)
        while let end = buffer.firstIndex(of: 10) { try line(Data(buffer[..<end])); buffer.removeSubrange(...end) }
        guard buffer.count < 4096 else { throw MediaOutputError.invalid("Frame proof exceeds its line bound.") }
        return chunk.count
      }
      if consumed == 0 { break }
    }
    if !buffer.isEmpty { try line(buffer) }
    guard count == frames, dimensions, numerator > 0 else { throw MediaOutputError.invalid("Lossless publication is incomplete or changed its grid.") }
    return Proof(frames: frames, width: width, height: height, fpsNumerator: numerator, fpsDenominator: denominator, rgbSHA256: rgb)
  }
}
