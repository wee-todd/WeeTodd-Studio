@preconcurrency import AVFoundation
import CoreVideo
import Foundation
import VideoToolbox

/// Accepts owning RGB8 frames and explicit sample timing. No nominal-FPS clock.
public final class NativeUpscaleMovieWriter {
  /// Explicit export quality; sample PTS/durations remain the only playback clock.
  /// Qualified against native RGB, including each frame, rather than lossy movie equality.
  public static let hevcQuality: Float = 0.9
  static var hevcCompressionProperties: [String: Any] {
    // Use the VT compressed-video quality contract. AV's still-image symbol
    // aliases the same key string, but does not document HEVC quality semantics.
    // Leaving quality unset couples export decisions to encoder defaults,
    // including decisions affected by the expected frame rate.
    [AVVideoAllowFrameReorderingKey: false,
      kVTCompressionPropertyKey_Quality as String: hevcQuality]
  }
  public static func validateRaster(width: Int, height: Int) throws {
    guard (2...16384).contains(width), (2...16384).contains(height), width.isMultiple(of: 2), height.isMultiple(of: 2) else {
      throw NativeUpscaleMovieError.invalid("HEVC movie export requires even output dimensions between 2 and 16384 pixels. Choose another scale or prepare a compatible source raster.")
    }
  }
  private let writer: AVAssetWriter, input: AVAssetWriterInput
  private let adaptor: AVAssetWriterInputPixelBufferAdaptor
  private let width: Int, height: Int
  private var format: CMVideoFormatDescription?
  private var last: CMTime?
  public init(url: URL, width: Int, height: Int, timeScale: CMTimeScale,
    transform: CGAffineTransform = .identity, scale: Double = 1,
    colorProperties: [String: Any] = [:]) throws {
    try Self.validateRaster(width: width, height: height)
    guard timeScale > 0,
      !FileManager.default.fileExists(atPath: url.path) else { throw NativeUpscaleMovieError.invalid("Invalid encoder dimensions, timescale or existing output.") }
    self.width = width; self.height = height
    writer = try AVAssetWriter(outputURL: url, fileType: .mov)
    var settings: [String: Any] = [AVVideoCodecKey: AVVideoCodecType.hevc,
      AVVideoWidthKey: width, AVVideoHeightKey: height,
      AVVideoCompressionPropertiesKey: Self.hevcCompressionProperties]
    if !colorProperties.isEmpty {
      // AVFoundation requires a complete SDR tuple even when the source omits
      // its YCbCr matrix. Preserve supplied tags and declare the RGB encoder's
      // missing conversion fields explicitly as Rec.709.
      var complete: [String: Any] = [AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
        AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
        AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2]
      complete.merge(colorProperties) { _, supplied in supplied }
      settings[AVVideoColorPropertiesKey] = complete
    }
    guard writer.canApply(outputSettings: settings, forMediaType: .video) else { throw NativeUpscaleMovieError.invalid("The encoder cannot preserve this exact output raster.") }
    input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    input.transform = CGAffineTransform(a: transform.a, b: transform.b, c: transform.c, d: transform.d,
      tx: transform.tx * scale, ty: transform.ty * scale)
    input.mediaTimeScale = timeScale; writer.movieTimeScale = timeScale
    adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
      kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height])
    guard writer.canAdd(input) else { throw NativeUpscaleMovieError.invalid("The encoder cannot accept this movie track.") }
    writer.add(input)
    guard writer.startWriting() else { throw writer.error ?? NativeUpscaleMovieError.invalid("Cannot start movie encoding.") }
    writer.startSession(atSourceTime: .zero)
  }
  public func appendRGB(_ bytes: [UInt8], at timestamp: CMTime, duration: CMTime) throws {
    guard bytes.count == width * height * 3, timestamp.isNumeric, timestamp >= .zero,
      duration.isNumeric, duration > .zero, last.map({ timestamp > $0 }) ?? true else {
      throw NativeUpscaleMovieError.invalid("Frame dimensions or presentation timing changed.")
    }
    while !input.isReadyForMoreMediaData {
      try Task.checkCancellation()
      guard writer.status == .writing else { throw writer.error ?? NativeUpscaleMovieError.invalid("Video encoding stopped.") }
      Thread.sleep(forTimeInterval: 0.002)
    }
    try Task.checkCancellation()
    var allocated: CVPixelBuffer?
    guard let pool = adaptor.pixelBufferPool, CVPixelBufferPoolCreatePixelBuffer(nil, pool, &allocated) == kCVReturnSuccess,
      let buffer = allocated, CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess else { throw NativeUpscaleMovieError.invalid("Cannot allocate encoder pixels.") }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { CVPixelBufferUnlockBaseAddress(buffer, []); throw NativeUpscaleMovieError.invalid("Encoder pixels are unavailable.") }
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    bytes.withUnsafeBytes { source in
      let input = source.bindMemory(to: UInt8.self)
      for y in 0..<height {
        let output = base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self)
        for x in 0..<width {
          let i = (y * width + x) * 3, j = x * 4
          output[j] = input[i + 2]; output[j + 1] = input[i + 1]; output[j + 2] = input[i]; output[j + 3] = 255
        }
      }
    }
    CVPixelBufferUnlockBaseAddress(buffer, [])
    if format == nil { guard CMVideoFormatDescriptionCreateForImageBuffer(allocator: nil, imageBuffer: buffer,
      formatDescriptionOut: &format) == noErr else { throw NativeUpscaleMovieError.invalid("Cannot describe encoder pixels.") } }
    var sample: CMSampleBuffer?, timing = CMSampleTimingInfo(duration: duration, presentationTimeStamp: timestamp, decodeTimeStamp: .invalid)
    guard let format, CMSampleBufferCreateReadyWithImageBuffer(allocator: nil, imageBuffer: buffer, formatDescription: format,
      sampleTiming: &timing, sampleBufferOut: &sample) == noErr, let sample, input.append(sample) else {
      throw writer.error ?? NativeUpscaleMovieError.invalid("The encoder rejected a source-timed frame.")
    }
    last = timestamp
  }
  public func finish(at end: CMTime) async throws {
    guard last.map({ end > $0 }) == true else { throw NativeUpscaleMovieError.invalid("The movie has no valid final frame duration.") }
    writer.endSession(atSourceTime: end); input.markAsFinished(); await writer.finishWriting()
    guard writer.status == .completed else { throw writer.error ?? NativeUpscaleMovieError.invalid("Movie encoding failed.") }
  }
  public func cancelIfWriting() { if writer.status == .writing { writer.cancelWriting() } }
  deinit { cancelIfWriting() }
}
