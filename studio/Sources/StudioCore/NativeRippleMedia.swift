import AVFoundation
import CoreImage
import Darwin
import Foundation
import ImageIO

/// Weight-free Ripple source inspection and exact editorial-frame extraction.
/// The source is decoded in presentation order; nominal FPS only checks cadence.
public enum NativeRippleMedia {
  private struct Request {
    let url: URL
    let start: Double
    let duration: Double
    let fps: Double
    let width: Int
    let height: Int
    let frame: Int?
    let signature: [Int64]
    let frameCount: Int

    init(_ raw: [String: Any], requireFrame: Bool) throws {
      guard let path = raw["source_path"] as? String, path.hasPrefix("/"), !path.utf8.contains(0),
        let start = (raw["source_start"] as? NSNumber)?.doubleValue, start.isFinite, start >= 0,
        let duration = (raw["duration"] as? NSNumber)?.doubleValue, duration.isFinite, (0.001...30).contains(duration),
        let fps = (raw["frame_rate"] as? NSNumber)?.doubleValue, fps.isFinite, (1...60).contains(fps),
        let width = raw["width"] as? Int, let height = raw["height"] as? Int,
        [width, height].allSatisfy({ (32...1920).contains($0) && $0 % 32 == 0 }) else {
        throw StudioError.invalid("Ripple needs an existing source movie and valid finite interval, frame rate, and dimensions.")
      }
      let frameCount = Int(ceil(duration * fps - 1e-7))
      let frame = raw["frame"] as? Int
      guard !requireFrame || (frame != nil && (0..<frameCount).contains(frame!)) else {
        throw StudioError.invalid("Select a source frame inside the Ripple interval.")
      }
      self.url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
      self.signature = try Self.fileSignature(url)
      self.start = start; self.duration = duration; self.fps = fps
      self.width = width; self.height = height; self.frame = frame; self.frameCount = frameCount
    }

    func verify() throws {
      guard try Self.fileSignature(url) == signature else {
        throw StudioError.invalid("The Ripple source changed during inspection. Start a new draft.")
      }
    }

    private static func fileSignature(_ url: URL) throws -> [Int64] {
      var status = stat()
      guard url.path.withCString({ Darwin.lstat($0, &status) }) == 0,
        status.st_mode & S_IFMT == S_IFREG, status.st_size > 0,
        FileManager.default.isReadableFile(atPath: url.path) else {
        throw StudioError.invalid("Select an existing readable source movie for Ripple.")
      }
      return [Int64(status.st_dev), Int64(bitPattern: UInt64(status.st_ino)), status.st_size,
        Int64(status.st_mtimespec.tv_sec), Int64(status.st_mtimespec.tv_nsec),
        Int64(status.st_ctimespec.tv_sec), Int64(status.st_ctimespec.tv_nsec)]
    }
  }

  private struct Inspection {
    let request: Request
    let source: [String: Any]
    let firstTime: Double
    let decodedFrames: Int
    let transform: CGAffineTransform
    let asset: AVURLAsset
    let track: AVAssetTrack
  }

  private static func inspectSource(_ request: Request) async throws -> Inspection {
    try request.verify(); try Task.checkCancellation()
    let asset = AVURLAsset(url: request.url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw StudioError.invalid("Ripple requires a movie with a video track.")
    }
    let duration = try await asset.load(.duration).seconds
    let fps = Double(try await track.load(.nominalFrameRate))
    let natural = try await track.load(.naturalSize)
    let transform = try await track.load(.preferredTransform)
    let rect = CGRect(origin: .zero, size: natural).applying(transform).standardized
    guard duration.isFinite, duration > 0, fps.isFinite, fps > 0,
      request.start + request.duration <= duration + 0.001,
      rect.width.isFinite, rect.height.isFinite, rect.width > 0, rect.height > 0 else {
      throw StudioError.invalid("The Ripple source interval or video track is invalid.")
    }
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else { throw StudioError.invalid("Cannot inspect Ripple source frames.") }
    reader.add(output)
    let start = CMTime(seconds: request.start, preferredTimescale: 1_000_000_000)
    let end = CMTime(seconds: request.start + request.duration, preferredTimescale: 1_000_000_000)
    reader.timeRange = CMTimeRange(start: start, end: end)
    guard reader.startReading() else { throw reader.error ?? StudioError.invalid("Cannot read Ripple source frames.") }
    defer { reader.cancelReading() }
    var first: Double?, count = 0
    let tolerance = max(0.001, 0.01 / fps)
    while let sample = output.copyNextSampleBuffer() {
      try Task.checkCancellation()
      let time = CMSampleBufferGetPresentationTimeStamp(sample).seconds
      guard time.isFinite else { throw StudioError.invalid("Ripple source timestamps must be finite.") }
      if time < request.start - 1e-7 || time >= request.start + request.duration - 1e-7 { continue }
      if first == nil { first = time }
      guard abs(time - (first! + Double(count) / fps)) <= tolerance else {
        throw StudioError.invalid("Ripple requires a constant frame rate source. Convert this movie before editing.")
      }
      count += 1
    }
    if reader.status == .failed { throw reader.error ?? StudioError.invalid("Ripple source inspection failed.") }
    guard let first else {
      throw StudioError.invalid("The Ripple interval contains no decoded source frames.")
    }
    try request.verify()
    let audio = try await asset.loadTracks(withMediaType: .audio)
    let source: [String: Any] = ["kind": "video", "path": request.url.path, "duration": duration,
      "width": Int(rect.width.rounded()), "height": Int(rect.height.rounded()), "fps": fps,
      "hasAudio": !audio.isEmpty, "start_time": 0.0, "rotation": 0.0]
    return Inspection(request: request, source: source, firstTime: first,
      decodedFrames: count, transform: transform, asset: asset, track: track)
  }

  public static func inspect(_ raw: [String: Any]) async throws -> [String: Any] {
    let value = try await inspectSource(Request(raw, requireFrame: false))
    let request = value.request
    let minimumIntervals = Int(ceil(0.25 * request.fps / 8))
    let modelFrames = 1 + 8 * max(1, Int(ceil(Double(request.frameCount - 1) / 8)), minimumIntervals)
    return ["source": value.source, "frames": request.frameCount, "frame_count": request.frameCount,
      "model_frames": modelFrames, "frame_rate": request.fps,
      "source_frame_rate": value.source["fps"]!, "duration": request.duration,
      "source_start": request.start, "source_preview_start": value.firstTime,
      "width": request.width, "height": request.height,
      "has_audio": value.source["hasAudio"]!]
  }

  public static func extractFrame(_ raw: [String: Any], into directory: URL) async throws -> [String: Any] {
    let value = try await inspectSource(Request(raw, requireFrame: true))
    let request = value.request, frame = request.frame!
    let sourceFPS = value.source["fps"] as! Double
    guard abs(request.fps - sourceFPS) <= max(1, sourceFPS) * 0.00001 else {
      throw StudioError.invalid("Ripple frame rate must match the source movie. Use \(sourceFPS) fps.")
    }
    guard value.decodedFrames >= request.frameCount else {
      throw StudioError.invalid("The Ripple interval contains too few decoded source frames.")
    }
    let reader = try AVAssetReader(asset: value.asset)
    let output = AVAssetReaderTrackOutput(track: value.track, outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else { throw StudioError.invalid("Cannot decode the Ripple source frame.") }
    reader.add(output)
    let start = CMTime(seconds: request.start, preferredTimescale: 1_000_000_000)
    let end = CMTime(seconds: request.start + request.duration, preferredTimescale: 1_000_000_000)
    reader.timeRange = CMTimeRange(start: start, end: end)
    guard reader.startReading() else { throw reader.error ?? StudioError.invalid("Cannot decode the Ripple source frame.") }
    defer { reader.cancelReading() }
    var index = 0, image: CGImage?
    let context = CIContext(options: [.cacheIntermediates: false])
    while let sample = output.copyNextSampleBuffer() {
      try Task.checkCancellation()
      let time = CMSampleBufferGetPresentationTimeStamp(sample).seconds
      if time < request.start - 1e-7 || time >= request.start + request.duration - 1e-7 { continue }
      if index == frame, let pixels = CMSampleBufferGetImageBuffer(sample) {
        let source = CIImage(cvPixelBuffer: pixels).transformed(by: value.transform)
        image = context.createCGImage(source, from: source.extent, format: .RGBA8,
          colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        break
      }
      index += 1
    }
    if reader.status == .failed { throw reader.error ?? StudioError.invalid("Ripple source frame decoding failed.") }
    guard let image else { throw StudioError.invalid("The selected Ripple source frame could not be decoded.") }
    try request.verify(); try Task.checkCancellation()
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let target = directory.appendingPathComponent(String(format: "source-frame-%06d.png", frame))
    let bytes = NSMutableData()
    guard let writer = CGImageDestinationCreateWithData(bytes, "public.png" as CFString, 1, nil) else {
      throw StudioError.invalid("Cannot encode the Ripple source frame.")
    }
    CGImageDestinationAddImage(writer, image, nil)
    guard CGImageDestinationFinalize(writer) else { throw StudioError.invalid("Cannot finish the Ripple source frame.") }
    try (bytes as Data).write(to: target, options: .withoutOverwriting)
    return ["image_path": target.path, "path": target.path, "frame": frame,
      "source_time": value.firstTime + Double(frame) / request.fps]
  }

  /// Streams the author's causal guide order to packed RGB24 storage: the
  /// edited frame, then the unchanged source interval, then terminal clones.
  /// Only one decoded and one resized frame are resident at a time. The worker
  /// can consume this file in bounded windows when its video VAE is streamed.
  public static func prepareGuide(_ raw: [String: Any], editedFirstFrame: URL,
    into directory: URL, progress: (Int, Int) throws -> Void = { _, _ in }) async throws -> [String: Any] {
    let value = try await inspectSource(Request(raw, requireFrame: false))
    let request = value.request, sourceFPS = value.source["fps"] as! Double
    guard abs(request.fps - sourceFPS) <= max(1, sourceFPS) * 0.00001,
      value.decodedFrames >= request.frameCount else {
      throw StudioError.invalid("Ripple guide needs the source movie's exact frame rate and full selected interval.")
    }
    let modelFrames = 1 + 8 * max(1, Int(ceil(Double(request.frameCount - 1) / 8)),
      Int(ceil(0.25 * request.fps / 8)))
    let (pixels, pixelOverflow) = request.width.multipliedReportingOverflow(by: request.height)
    let (frameBytes, frameOverflow) = pixels.multipliedReportingOverflow(by: 3)
    let (totalBytes, totalOverflow) = frameBytes.multipliedReportingOverflow(by: modelFrames)
    guard !pixelOverflow, !frameOverflow, !totalOverflow, totalBytes > 0 else {
      throw StudioError.invalid("Ripple guide dimensions exceed the native output bound.")
    }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let filesystem = try FileManager.default.attributesOfFileSystem(forPath: directory.path)
    guard let available = filesystem[.systemFreeSize] as? NSNumber,
      available.int64Value >= Int64(totalBytes) + 512 * 1024 * 1024 else {
      throw StudioError.invalid("The Ripple guide needs enough free disk space for its streamed frames.")
    }
    guard editedFirstFrame.isFileURL,
      let edited = CIImage(contentsOf: editedFirstFrame,
        options: [.applyOrientationProperty: true]) else {
      throw StudioError.invalid("Select a readable edited first-frame image for Ripple.")
    }
    let context = CIContext(options: [.cacheIntermediates: false])
    let color = CGColorSpace(name: CGColorSpace.sRGB)
    func rgb(_ image: CIImage) throws -> Data {
      let extent = image.extent.standardized
      guard extent.width.isFinite, extent.height.isFinite,
        extent.width > 0, extent.height > 0 else {
        throw StudioError.invalid("Ripple guide contains an invalid image extent.")
      }
      let scale = max(CGFloat(request.width) / extent.width,
        CGFloat(request.height) / extent.height)
      let scaled = image.applyingFilter("CILanczosScaleTransform", parameters: [
        "inputScale": scale, "inputAspectRatio": 1])
      let outputWidth = CGFloat(request.width), outputHeight = CGFloat(request.height)
      let cropX = scaled.extent.midX - outputWidth / 2
      let cropY = scaled.extent.midY - outputHeight / 2
      let crop = CGRect(x: cropX, y: cropY, width: outputWidth, height: outputHeight)
      let centered = scaled.cropped(to: crop).transformed(by:
        CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
      var rgba = [UInt8](repeating: 0, count: pixels * 4)
      rgba.withUnsafeMutableBytes { bytes in
        context.render(centered, toBitmap: bytes.baseAddress!, rowBytes: request.width * 4,
          bounds: CGRect(x: 0, y: 0, width: request.width, height: request.height),
          format: .RGBA8, colorSpace: color)
      }
      var output = Data(count: frameBytes)
      output.withUnsafeMutableBytes { destination in
        rgba.withUnsafeBytes { source in
          let dst = destination.bindMemory(to: UInt8.self)
          let src = source.bindMemory(to: UInt8.self)
          for index in 0..<pixels {
            dst[index * 3] = src[index * 4]
            dst[index * 3 + 1] = src[index * 4 + 1]
            dst[index * 3 + 2] = src[index * 4 + 2]
          }
        }
      }
      return output
    }
    let target = directory.appendingPathComponent("guide.rgb24")
    let descriptor = Darwin.open(target.path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
      S_IRUSR | S_IWUSR)
    guard descriptor >= 0 else { throw StudioError.invalid("Cannot create the Ripple guide stream.") }
    let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    var finished = false
    defer {
      try? file.close()
      if !finished { try? FileManager.default.removeItem(at: target) }
    }
    try file.write(contentsOf: rgb(edited))
    try progress(1, modelFrames)
    let reader = try AVAssetReader(asset: value.asset)
    let output = AVAssetReaderTrackOutput(track: value.track, outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else { throw StudioError.invalid("Cannot decode the Ripple guide movie.") }
    reader.add(output)
    reader.timeRange = CMTimeRange(
      start: CMTime(seconds: request.start, preferredTimescale: 1_000_000_000),
      end: CMTime(seconds: request.start + request.duration, preferredTimescale: 1_000_000_000))
    guard reader.startReading() else { throw reader.error ?? StudioError.invalid("Cannot read Ripple guide frames.") }
    defer { reader.cancelReading() }
    var written = 1, last: Data?
    while written < modelFrames, let sample = output.copyNextSampleBuffer() {
      try Task.checkCancellation()
      let time = CMSampleBufferGetPresentationTimeStamp(sample).seconds
      if time < request.start - 1e-7 || time >= request.start + request.duration - 1e-7 { continue }
      guard let pixels = CMSampleBufferGetImageBuffer(sample) else {
        throw StudioError.invalid("A Ripple source frame could not be decoded.")
      }
      let frame = try rgb(CIImage(cvPixelBuffer: pixels).transformed(by: value.transform))
      try file.write(contentsOf: frame)
      last = frame; written += 1
      try progress(written, modelFrames)
    }
    if reader.status == .failed { throw reader.error ?? StudioError.invalid("Ripple guide decoding failed.") }
    guard let last else { throw StudioError.invalid("Ripple guide has no source frame.") }
    while written < modelFrames {
      try Task.checkCancellation()
      try file.write(contentsOf: last)
      written += 1; try progress(written, modelFrames)
    }
    try request.verify()
    try file.close()
    finished = true
    return ["rgb_path": target.path, "frames": modelFrames, "editorial_frames": request.frameCount,
      "width": request.width, "height": request.height, "frame_rate": request.fps,
      "bytes": totalBytes, "format": "rgb24", "order": "edited-first,source-interval,terminal-clone"]
  }

  /// Verify the published editorial take, not just the worker's declared result.
  /// Count decoded presentation frames: compressed packet counts are not frames.
  public static func verifyPublishedTake(_ path: String, draft: RippleDraft,
    hasAudio: Bool) async throws {
    let url = URL(fileURLWithPath: path)
    guard url.isFileURL, FileManager.default.fileExists(atPath: url.path) else {
      throw StudioError.invalid("Ripple did not publish a movie at the reported path.")
    }
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw StudioError.invalid("Ripple take has no video track.")
    }
    let duration = try await asset.load(.duration).seconds
    let fps = Double(try await track.load(.nominalFrameRate))
    let dimensions = try await track.load(.naturalSize)
    let audio = try await asset.loadTracks(withMediaType: .audio)
    guard duration.isFinite, abs(duration - draft.duration) <= 1 / draft.frameRate + 0.002,
      fps.isFinite, abs(fps - draft.frameRate) <= max(1, draft.frameRate) * 0.00001,
      Int(dimensions.width.rounded()) == draft.width,
      Int(dimensions.height.rounded()) == draft.height,
      !audio.isEmpty == hasAudio else {
      throw StudioError.invalid("Ripple published movie timing, dimensions, or audio differs from the draft.")
    }
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else { throw StudioError.invalid("Cannot inspect Ripple take frames.") }
    reader.add(output)
    guard reader.startReading() else {
      throw reader.error ?? StudioError.invalid("Cannot read the Ripple take.")
    }
    defer { reader.cancelReading() }
    var count = 0
    while output.copyNextSampleBuffer() != nil {
      try Task.checkCancellation()
      count += 1
    }
    guard reader.status == .completed, count == draft.frameCount else {
      throw StudioError.invalid("Ripple published \(count) frames; the draft requires \(draft.frameCount).")
    }
  }
}
