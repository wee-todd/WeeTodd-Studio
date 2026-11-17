@preconcurrency import AVFoundation
import CoreVideo
import Foundation
import InferenceContracts
import InferenceMedia
import Metal
import MLX
import TensorIO

public struct RelayMovieOptions: Sendable {
  public var scale: NativeRelayVSRScale = .x2
  public var steps: Int
  public var seed: UInt64
  public var keyframeInterval: Int
  public var detail: Float
  public var frameLimit: Int?
  public var sourceStart: Double?
  public var sourceDuration: Double?

  public init(steps: Int = 4, seed: UInt64 = 42, keyframeInterval: Int = 15,
    detail: Float = 1, frameLimit: Int? = nil)
  {
    self.steps = steps
    self.seed = seed
    self.keyframeInterval = keyframeInterval
    self.detail = detail
    self.frameLimit = frameLimit
  }

  func validate() throws {
    _ = try RelaySchedule.sigmas(steps: steps)
    guard (1...60).contains(keyframeInterval), detail.isFinite, (0...2).contains(detail),
      frameLimit.map({ $0 > 0 }) ?? true else {
      throw RelayVSRError.invalid("Use an interval of 1…60, a detail value of 0…2, and a positive frame limit.")
    }
    guard sourceStart.map({ $0.isFinite && $0 >= 0 }) ?? true,
      sourceDuration.map({ $0.isFinite && $0 > 0 }) ?? true else {
      throw RelayVSRError.invalid("The source interval must have a nonnegative start and positive duration.")
    }
  }
}

public struct RelayMovieResult: Sendable {
  public let frames: Int
  public let sourceIn: Double
  public let duration: Double
}

public enum RelayMovie {
  public struct Admission: Codable, Sendable {
    public let scale: Int, outputWidth: Int, outputHeight: Int
    public let estimatedBytes: UInt64
  }
  private static func validateModels(_ root: URL, scale: NativeRelayVSRScale) throws -> URL {
    let folder = try scale.resolveDirectory(root)
    let wan = try SafeTensorFile(url: folder.appendingPathComponent(scale.checkpointNames[0]))
    let flash = try SafeTensorFile(url: folder.appendingPathComponent(scale.checkpointNames[1]))
    try scale.validateMetadata(wan: wan.metadata, flash: flash.metadata)
    guard wan.tensors["lq_proj_in.conv1.weight"]?.shape == [768, UInt64(3 * scale.wanUnshuffle * scale.wanUnshuffle), 1, 1] else {
      throw RelayVSRError.invalid("The RelayVSR learned LQ projection does not match the selected scale.")
    }
    return folder
  }
  public static func preflight(input: URL, modelDirectory: URL, options: RelayMovieOptions = .init()) async throws -> Admission {
    try options.validate()
    _ = try validateModels(modelDirectory, scale: options.scale)
    let asset = AVURLAsset(url: input, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
    guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw RelayVSRError.invalid("Input has no video track.") }
    let size = try await track.load(.naturalSize)
    guard size.width.isFinite, size.height.isFinite, (1...4096).contains(size.width), (1...4096).contains(size.height) else {
      throw RelayVSRError.invalid("Unsupported RelayVSR source dimensions.")
    }
    let g = try RelayGeometry(width: Int(size.width), height: Int(size.height), scale: options.scale)
    try NativeUpscaleMovieWriter.validateRaster(width: g.outputWidth, height: g.outputHeight)
    guard let device = MTLCreateSystemDefaultDevice() else { throw RelayVSRError.invalid("RelayVSR requires a Metal GPU.") }
    try RelayMemoryPlan.validate(geometry: g, options: options, budget: device.recommendedMaxWorkingSetSize)
    return .init(scale: options.scale.rawValue, outputWidth: g.outputWidth, outputHeight: g.outputHeight,
      estimatedBytes: RelayMemoryPlan.estimatedBytes(geometry: g, options: options))
  }
  @discardableResult
  public static func upscale(input: URL, output: URL, modelDirectory: URL,
    options: RelayMovieOptions = .init(), onWait: @Sendable () -> Void = {},
    progress: @Sendable (Int) -> Void = { _ in }) async throws
    -> RelayMovieResult
  {
    try options.validate()
    let modelDirectory = try validateModels(modelDirectory, scale: options.scale)
    guard input.isFileURL, output.isFileURL, modelDirectory.isFileURL,
      output.pathExtension.lowercased() == "mov",
      !FileManager.default.fileExists(atPath: output.path) else {
      throw RelayVSRError.invalid("Provide local paths and a new .mov output file.")
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw RelayVSRError.invalid("RelayVSR requires a Metal GPU.")
    }
    let asset = AVURLAsset(url: input, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw RelayVSRError.invalid("Input has no video track.")
    }
    let (transform, trackRange, timeScale) = try await track.load(
      .preferredTransform, .timeRange, .naturalTimeScale)
    let content = try await videoContentRange(track, fallback: trackRange)
    let selected = options.sourceStart != nil || options.sourceDuration != nil
    // Editorial bounds also govern audio, whose clock can be finer than the video track's.
    let editorialScale: CMTimeScale = 600_000_000
    let start = CMTime(seconds: options.sourceStart ?? 0, preferredTimescale: editorialScale)
    let selectionEnd = options.sourceDuration.map {
      CMTime(seconds: (options.sourceStart ?? 0) + $0, preferredTimescale: editorialScale)
    }
    let timeRange = selected ? CMTimeRange(start: max(content.start, start),
      end: min(content.end, selectionEnd ?? content.end)) : content
    guard timeRange.isValid, timeRange.duration > .zero else {
      throw RelayVSRError.invalid("The selected interval contains no video.")
    }
    let audio = try await asset.loadTracks(withMediaType: .audio)
    let reader = try AVAssetReader(asset: asset)
    reader.timeRange = timeRange
    let video = AVAssetReaderTrackOutput(track: track,
      outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    video.alwaysCopiesSampleData = false
    guard reader.canAdd(video) else { throw RelayVSRError.invalid("Cannot decode the source video track.") }
    reader.add(video)
    guard reader.startReading() else { throw reader.error ?? RelayVSRError.invalid("Video reader failed.") }
    defer { reader.cancelReading() }
    guard let first = try readFrame(video) else {
      throw reader.error ?? RelayVSRError.invalid("Input has no decodable video frames.")
    }
    let geometry = try RelayGeometry(width: CVPixelBufferGetWidth(first.pixels),
      height: CVPixelBufferGetHeight(first.pixels), scale: options.scale)
    try NativeUpscaleMovieWriter.validateRaster(width: geometry.outputWidth, height: geometry.outputHeight)
    try RelayMemoryPlan.validate(geometry: geometry, options: options,
      budget: device.recommendedMaxWorkingSetSize)
    let folder = output.deletingLastPathComponent()
    let work = folder.appendingPathComponent(".relayvsr-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: work, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: work) }
    let videoFile = work.appendingPathComponent("video.mov")
    let muxedFile = work.appendingPathComponent("output.mov")
    let writer = try RelayMovieWriter(url: videoFile, geometry: geometry, transform: transform,
      timeScale: timeScale)
    defer { writer.cancelIfWriting() }

    var pending: (image: MLXArray, timestamp: CMTime)?
    var count = 0
    func write(_ image: MLXArray, at timestamp: CMTime, until end: CMTime) throws {
      try writer.append(image, at: timestamp, duration: end - timestamp)
      count += 1
      progress(count)
    }
    let inferenceLease = try NativeInferenceLease.acquire(onWait: onWait)
    defer { inferenceLease.release() }
    try renderFrames(first: first, read: { try readFrame(video) },
      geometry: geometry, modelDirectory: modelDirectory, options: options) { rgb, timestamp in
      if let pending { try write(pending.image, at: pending.timestamp, until: timestamp) }
      pending = (rgb, timestamp)
    }
    let previewEnd = options.frameLimit != nil && reader.status == .reading
      ? try readFrame(video)?.timestamp : nil
    guard reader.status != .failed else {
      throw reader.error ?? RelayVSRError.invalid("Decoding the source video failed.")
    }
    guard let pending else {
      throw RelayVSRError.invalid("RelayVSR produced no video frames.")
    }
    let finalTimestamp = first.timestamp + pending.timestamp
    let sourceEnd = previewEnd ?? finalVideoEnd(asset, track: track,
      last: finalTimestamp, editEnd: timeRange.end)
    let end = sourceEnd - first.timestamp
    guard end.isNumeric, end > pending.timestamp else {
      throw RelayVSRError.invalid("Cannot determine the source video's final frame duration.")
    }
    try write(pending.image, at: pending.timestamp, until: end)
    try await writer.finish(at: end)
    try Task.checkCancellation()
    let videoStart: CMTime
    if audio.isEmpty && (!selected || first.timestamp == start) {
      try FileManager.default.moveItem(at: videoFile, to: output)
      videoStart = .zero
    } else {
      videoStart = try await mux(videoFile, audio: audio, sourceStart: first.timestamp,
        audioStart: selected ? start : nil, previewEnd: previewEnd ?? selectionEnd, output: muxedFile)
      try Task.checkCancellation()
      try FileManager.default.moveItem(at: muxedFile, to: output)
    }
    return RelayMovieResult(frames: count, sourceIn: videoStart.seconds, duration: end.seconds)
  }

  private static func readFrame(_ video: AVAssetReaderTrackOutput) throws -> RelayMovieFrame? {
    guard let sample = video.copyNextSampleBuffer() else { return nil }
    guard let pixels = sample.imageBuffer, sample.presentationTimeStamp.isNumeric else {
      throw RelayVSRError.invalid("The source contains a frame without pixels or a timestamp.")
    }
    return RelayMovieFrame(pixels: pixels, timestamp: sample.presentationTimeStamp)
  }

  static func videoContentRange(_ track: AVAssetTrack, fallback: CMTimeRange) async throws -> CMTimeRange {
    try await NativeUpscaleMovieIO.videoContentRange(track, fallback: fallback)
  }
  private static func finalVideoEnd(_ asset: AVAsset, track: AVAssetTrack, last: CMTime, editEnd: CMTime) -> CMTime {
    NativeUpscaleMovieIO.finalVideoEnd(asset, track: track, last: last, editEnd: editEnd)
  }

  private static func renderFrames(first: RelayMovieFrame, read: () throws -> RelayMovieFrame?,
    geometry: RelayGeometry, modelDirectory: URL, options: RelayMovieOptions,
    append: (MLXArray, CMTime) throws -> Void) throws
  {
    let flash = try RelayFlash(weights: modelDirectory.appendingPathComponent(
      geometry.scale.checkpointNames[1]), geometry: geometry)
    let wan = try RelayWan(weights: modelDirectory.appendingPathComponent(
      geometry.scale.checkpointNames[0]), geometry: geometry,
      steps: options.steps, seed: options.seed)
    var windows = RelayWindowBuffer<RelayMovieFrame>(interval: options.keyframeInterval)
    var start: [RelayKV]?

    func process(_ window: [RelayMovieFrame]) throws {
      let firstWindow = start == nil
      if firstWindow { start = try flash.prepareEndpoint(wan.keyframe(window[0].tensor(geometry))) }
      guard let startEndpoint = start else { return }
      let endEndpoint = window.count == 1 ? startEndpoint
        : try flash.prepareEndpoint(wan.keyframe(window[window.count - 1].tensor(geometry)))
      flash.beginWindow()
      for (index, frame) in window.enumerated() {
        try Task.checkCancellation()
        let rgb = try flash.frame(frame.tensor(geometry), start: startEndpoint,
          end: endEndpoint, detail: options.detail)
        if index != 0 || firstWindow {
          try append(rgb, frame.timestamp - first.timestamp)
        }
      }
      start = endEndpoint
    }

    var last = first
    var count = 1
    _ = windows.append(first)
    while options.frameLimit.map({ count < $0 }) ?? true {
      try Task.checkCancellation()
      guard let frame = try read() else { break }
      guard frame.timestamp > last.timestamp else {
        throw RelayVSRError.invalid("Source video presentation timestamps must be strictly increasing.")
      }
      last = frame
      count += 1
      if let window = windows.append(frame) { try process(window) }
    }
    if let window = windows.finish() { try process(window) }
  }

  @discardableResult
  static func mux(_ videoFile: URL, audio: [AVAssetTrack], sourceStart: CMTime,
    audioStart: CMTime? = nil, previewEnd: CMTime?, output: URL) async throws -> CMTime {
    try await NativeUpscaleMovieIO.mux(videoFile, audio: audio, sourceStart: sourceStart,
      audioStart: audioStart, previewEnd: previewEnd, output: output)
  }

}

private struct RelayMovieFrame {
  let pixels: CVPixelBuffer
  let timestamp: CMTime

  func tensor(_ geometry: RelayGeometry) throws -> MLXArray {
    let width = geometry.width
    let height = geometry.height
    guard CVPixelBufferGetWidth(pixels) == width, CVPixelBufferGetHeight(pixels) == height,
      CVPixelBufferLockBaseAddress(pixels, .readOnly) == kCVReturnSuccess else {
      throw RelayVSRError.invalid("Source frame dimensions changed or its pixel buffer is unavailable.")
    }
    defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixels) else {
      throw RelayVSRError.invalid("Source frame pixel buffer is empty.")
    }
    let stride = CVPixelBufferGetBytesPerRow(pixels)
    var bytes = Data(count: width * height * 4)
    bytes.withUnsafeMutableBytes { output in
      for row in 0..<height {
        output.baseAddress!.advanced(by: row * width * 4)
          .copyMemory(from: base.advanced(by: row * stride), byteCount: width * 4)
      }
    }
    let bgra = MLXArray(bytes, [height, width, 4], dtype: .uint8)
    return geometry.prepare(bgra.take(MLXArray([Int32(2), 1, 0]), axis: 2))
  }
}

final class RelayMovieWriter {
  let writer: AVAssetWriter
  let input: AVAssetWriterInput
  let adaptor: AVAssetWriterInputPixelBufferAdaptor
  let geometry: RelayGeometry
  private var format: CMVideoFormatDescription?

  init(url: URL, geometry: RelayGeometry, transform: CGAffineTransform,
    timeScale: CMTimeScale) throws
  {
    self.geometry = geometry
    writer = try AVAssetWriter(outputURL: url, fileType: .mov)
    let outputSettings: [String: Any] = [
      AVVideoCodecKey: AVVideoCodecType.hevc,
      AVVideoWidthKey: geometry.outputWidth,
      AVVideoHeightKey: geometry.outputHeight,
      AVVideoCompressionPropertiesKey: [AVVideoAllowFrameReorderingKey: false],
    ]
    guard writer.canApply(outputSettings: outputSettings, forMediaType: .video) else {
      throw RelayVSRError.invalid("Cannot encode the exact requested RelayVSR output raster.")
    }
    input = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
    input.transform = CGAffineTransform(a: transform.a, b: transform.b, c: transform.c, d: transform.d,
      tx: transform.tx * CGFloat(geometry.scale.rawValue), ty: transform.ty * CGFloat(geometry.scale.rawValue))
    if timeScale > 0 {
      input.mediaTimeScale = timeScale
      writer.movieTimeScale = timeScale
    }
    adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
      kCVPixelBufferWidthKey as String: geometry.outputWidth,
      kCVPixelBufferHeightKey as String: geometry.outputHeight,
    ])
    guard writer.canAdd(input) else { throw RelayVSRError.invalid("Cannot encode the requested video dimensions.") }
    writer.add(input)
    guard writer.startWriting() else {
      throw writer.error ?? RelayVSRError.invalid("Cannot start the video encoder.")
    }
    writer.startSession(atSourceTime: .zero)
  }

  func append(_ rgb: MLXArray, at timestamp: CMTime, duration: CMTime) throws {
    let width = geometry.outputWidth
    let height = geometry.outputHeight
    let image = rgb[0..<height, 0..<width, 0..<3]
    let bgr = clip((image + 1) * 127.5 + 0.5, min: 0, max: 255)
      .asType(.uint8).take(MLXArray([Int32(2), 1, 0]), axis: 2)
    let alpha = full([height, width, 1], values: MLXArray(UInt8(255)))
    let bytes = concatenated([bgr, alpha], axis: 2).asArray(UInt8.self)
    while !input.isReadyForMoreMediaData {
      try Task.checkCancellation()
      guard writer.status == .writing else {
        throw writer.error ?? RelayVSRError.invalid("Video encoding stopped.")
      }
      Thread.sleep(forTimeInterval: 0.002)
    }
    var buffer: CVPixelBuffer?
    guard let pool = adaptor.pixelBufferPool,
      CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer) == kCVReturnSuccess, let buffer,
      CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess else {
      throw RelayVSRError.invalid("Could not allocate an encoder pixel buffer.")
    }
    if let base = CVPixelBufferGetBaseAddress(buffer) {
      let stride = CVPixelBufferGetBytesPerRow(buffer)
      bytes.withUnsafeBytes { source in
        for row in 0..<height {
          base.advanced(by: row * stride).copyMemory(
            from: source.baseAddress!.advanced(by: row * width * 4), byteCount: width * 4)
        }
      }
    } else {
      CVPixelBufferUnlockBaseAddress(buffer, [])
      throw RelayVSRError.invalid("Encoder pixel buffer is empty.")
    }
    CVPixelBufferUnlockBaseAddress(buffer, [])
    if format == nil {
      guard CMVideoFormatDescriptionCreateForImageBuffer(allocator: nil, imageBuffer: buffer,
        formatDescriptionOut: &format) == noErr else {
        throw RelayVSRError.invalid("Could not describe the encoder pixel buffer.")
      }
    }
    var sample: CMSampleBuffer?
    var timing = CMSampleTimingInfo(duration: duration, presentationTimeStamp: timestamp,
      decodeTimeStamp: .invalid)
    guard let format,
      CMSampleBufferCreateReadyWithImageBuffer(allocator: nil, imageBuffer: buffer,
        formatDescription: format, sampleTiming: &timing, sampleBufferOut: &sample) == noErr,
      let sample else {
      throw RelayVSRError.invalid("Could not create an encoder sample with source timing.")
    }
    guard input.append(sample) else {
      throw writer.error ?? RelayVSRError.invalid("The encoder rejected a frame.")
    }
  }

  func finish(at end: CMTime) async throws {
    try Task.checkCancellation()
    writer.endSession(atSourceTime: end)
    input.markAsFinished()
    await writer.finishWriting()
    try Task.checkCancellation()
    guard writer.status == .completed else {
      throw writer.error ?? RelayVSRError.invalid("Could not finalize the upscaled video.")
    }
  }

  func cancelIfWriting() {
    if writer.status == .writing { writer.cancelWriting() }
  }
}
