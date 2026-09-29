import AVFoundation
import CoreImage
import Darwin
import Foundation
import ImageIO

/// Resolved editorial dependency, separate from the user's stored attachments.
/// Source metadata is checked on both sides of media extraction.
struct NativeLTXFrameSource {
  let clipID: UUID
  let takeID: UUID?
  let url: URL
  let start: Double
  let duration: Double
  let signature: [Int64]
  init(project: StudioProject, clip: Clip) throws {
    guard Set(project.clips.map(\.id)).count == project.clips.count,
      let source = try project.continuitySource(for: clip), !source.sourcePath.isEmpty,
      source.sourceIn.isFinite, source.sourceIn >= 0, source.duration.isFinite, source.duration > 0,
      (source.sourceIn + source.duration).isFinite else {
      throw StudioError.invalid("Match previous frame needs an accepted earlier movie and a finite visible interval.")
    }
    let path = (source.sourcePath as NSString).expandingTildeInPath
    guard path.hasPrefix("/"), !path.utf8.contains(0) else { throw StudioError.invalid("Relink the continuity source movie.") }
    url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
    signature = try Self.signature(url)
    clipID = source.id; takeID = source.activeRenderVersion?.id; start = source.sourceIn; duration = source.duration
  }
  private static func signature(_ url: URL) throws -> [Int64] {
    var status = stat()
    guard url.path.withCString({ Darwin.lstat($0, &status) }) == 0, status.st_mode & S_IFMT == S_IFREG,
      status.st_size > 0, FileManager.default.isReadableFile(atPath: url.path) else {
      throw StudioError.invalid("Render and accept the continuity source first, or relink its movie.")
    }
    return [Int64(status.st_dev), Int64(bitPattern: UInt64(status.st_ino)), status.st_size,
      Int64(status.st_mtimespec.tv_sec), Int64(status.st_mtimespec.tv_nsec),
      Int64(status.st_ctimespec.tv_sec), Int64(status.st_ctimespec.tv_nsec)]
  }
  func verify() throws {
    guard try Self.signature(url) == signature else { throw StudioError.invalid("The continuity source changed. Prepare the clip again.") }
  }
  var report: [String: Any] {
    var result: [String: Any] = ["version": 1, "mode": "frame", "engine": "ltx25", "saveContext": false,
      "sourceClipID": clipID.uuidString, "sourcePath": url.path, "sourceIn": start,
      "duration": duration, "sourceTimeEnd": start + duration, "sourceStat": signature]
    if let takeID { result["sourceTakeID"] = takeID.uuidString }
    return result
  }
  /// Decode only the selected interval, retaining one final pixel buffer. Packet
  /// timestamps select the endpoint; nominal FPS never guesses a VFR frame.
  func extract(to destination: URL) async throws -> Double {
    try verify(); try Task.checkCancellation()
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw StudioError.invalid("The continuity source has no video track.")
    }
    let mediaDuration = try await asset.load(.duration).seconds
    let end = start + duration
    guard mediaDuration.isFinite, end <= mediaDuration + 0.000001 else {
      throw StudioError.invalid("The visible continuity interval extends beyond the source movie.")
    }
    let transform = try await track.load(.preferredTransform)
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else { throw StudioError.invalid("Cannot decode the continuity source.") }
    reader.add(output)
    let begin = CMTime(seconds: start, preferredTimescale: 1_000_000_000)
    let finish = CMTime(seconds: end, preferredTimescale: 1_000_000_000)
    reader.timeRange = CMTimeRange(start: begin, end: finish)
    guard reader.startReading() else { throw reader.error ?? StudioError.invalid("Cannot start reading the continuity source.") }
    defer { reader.cancelReading() }
    var last: CVPixelBuffer?, lastTime: Double?
    while let sample = output.copyNextSampleBuffer() {
      try Task.checkCancellation()
      let time = CMSampleBufferGetPresentationTimeStamp(sample)
      if time >= begin, time < finish, let pixels = CMSampleBufferGetImageBuffer(sample) {
        last = pixels; lastTime = time.seconds
      }
    }
    if reader.status == .failed { throw reader.error ?? StudioError.invalid("Continuity source decoding failed.") }
    guard let last, let lastTime else { throw StudioError.invalid("The selected visible interval contains no video frame.") }
    try Task.checkCancellation(); try verify()
    let image = CIImage(cvPixelBuffer: last).transformed(by: transform)
    let context = CIContext(options: [.cacheIntermediates: false])
    guard let rendered = context.createCGImage(image, from: image.extent, format: .RGBA8,
      colorSpace: CGColorSpace(name: CGColorSpace.sRGB)) else { throw StudioError.invalid("Cannot create the continuity reference image.") }
    let data = NSMutableData()
    guard let writer = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil) else {
      throw StudioError.invalid("Cannot write the continuity reference image.")
    }
    CGImageDestinationAddImage(writer, rendered, nil)
    guard CGImageDestinationFinalize(writer) else { throw StudioError.invalid("Cannot finish the continuity reference image.") }
    try Task.checkCancellation(); try verify()
    try (data as Data).write(to: destination, options: .withoutOverwriting)
    return lastTime
  }
}
