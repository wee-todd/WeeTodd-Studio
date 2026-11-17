import Foundation
import CoreMedia
import CryptoKit

/// Opt-in diagnostic copy of the owning RGB24 bytes at the encoder boundary.
/// Never retains MLX tensors, pixel buffers or a caller's mutable storage.
public final class NativeRGBFrameCapture {
  private let directory: URL, width: Int, height: Int, frameBytes: Int
  private let metadata: [String:String]
  private let handle: FileHandle
  private let started = ProcessInfo.processInfo.systemUptime
  private var frames: [[String:Any]] = []
  private var totalBytes: UInt64 = 0
  private var digest = SHA256()
  private var last: CMTime?
  private var finished = false

  public init(directory: URL, width: Int, height: Int, metadata: [String:String] = [:]) throws {
    guard directory.isFileURL, directory.path.hasPrefix("/"),
      (1...16384).contains(width), (1...16384).contains(height),
      !FileManager.default.fileExists(atPath:directory.path) else {
      throw NativeUpscaleMovieError.invalid("RGB capture requires supported dimensions and a new absolute directory.")
    }
    self.directory = directory; self.width = width; self.height = height
    frameBytes = width * height * 3; self.metadata = metadata
    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
    let raw = directory.appendingPathComponent("frames.rgb")
    guard FileManager.default.createFile(atPath:raw.path,contents:nil) else {
      throw NativeUpscaleMovieError.invalid("Cannot create RGB diagnostic stream.")
    }
    handle = try FileHandle(forWritingTo:raw)
    try saveManifest(complete:false)
  }
  public func appendRGB(_ bytes: [UInt8], at timestamp: CMTime, duration: CMTime) throws {
    guard !finished, bytes.count == frameBytes, timestamp.isNumeric, timestamp >= .zero,
      duration.isNumeric, duration > .zero, last.map({ timestamp > $0 }) ?? true else {
      throw NativeUpscaleMovieError.invalid("RGB diagnostic dimensions or frame clock changed.")
    }
    let elapsed = ProcessInfo.processInfo.systemUptime - started
    let data = Data(bytes)
    try handle.write(contentsOf:data)
    digest.update(data:data)
    frames.append(["index":frames.count,"offset":totalBytes,"bytes":data.count,
      "ptsValue":timestamp.value,"ptsTimescale":timestamp.timescale,
      "durationValue":duration.value,"durationTimescale":duration.timescale,
      "sha256":Self.hex(SHA256.hash(data:data)),"appendSeconds":elapsed])
    totalBytes += UInt64(data.count); last = timestamp
    try saveManifest(complete:false)
  }
  public func finish() throws {
    guard !finished, !frames.isEmpty else { throw NativeUpscaleMovieError.invalid("RGB capture is empty or already closed.") }
    try handle.synchronize(); try handle.close()
    finished = true
    try saveManifest(complete:true)
  }
  private static func hex<D: Sequence>(_ digest: D) -> String where D.Element == UInt8 {
    digest.map { String(format:"%02x",$0) }.joined()
  }
  private func saveManifest(complete: Bool) throws {
    let copy = digest
    let body: [String:Any] = ["format":"weetodd-native-rgb-v1","pixelFormat":"rgb24",
      "width":width,"height":height,"rawFile":"frames.rgb","frameBytes":frameBytes,
      "frameCount":frames.count,"totalBytes":totalBytes,"complete":complete,
      "sha256":Self.hex(copy.finalize()),"metadata":metadata,"frames":frames]
    try JSONSerialization.data(withJSONObject:body,options:[.prettyPrinted,.sortedKeys])
      .write(to:directory.appendingPathComponent("manifest.json"),options:.atomic)
  }
  deinit { if !finished { try? handle.close() } }
}
