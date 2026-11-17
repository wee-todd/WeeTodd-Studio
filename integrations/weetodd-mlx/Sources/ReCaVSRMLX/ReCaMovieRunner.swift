@preconcurrency import AVFoundation
import CoreVideo
import Foundation
import InferenceContracts
import InferenceMedia
import Metal
import MLX

public struct ReCaProgress: Sendable {
  public let phase: String, completedFrames: Int
  public let seconds: Double
}

private struct ReCaMovieFrame {
  let pixels: CVPixelBuffer, timestamp: CMTime
  func rgb(width: Int, height: Int) throws -> MLXArray {
    guard CVPixelBufferGetWidth(pixels) == width, CVPixelBufferGetHeight(pixels) == height,
      CVPixelBufferLockBaseAddress(pixels, .readOnly) == kCVReturnSuccess else { throw ReCaError.invalid("Source frame raster changed.") }
    defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixels) else { throw ReCaError.invalid("Source pixels are unavailable.") }
    let stride = CVPixelBufferGetBytesPerRow(pixels)
    var bytes = Data(count: width * height * 4)
    bytes.withUnsafeMutableBytes { destination in
      for y in 0..<height { destination.baseAddress!.advanced(by: y * width * 4)
        .copyMemory(from: base.advanced(by: y * stride), byteCount: width * 4) }
    }
    return MLXArray(bytes, [height, width, 4], dtype: .uint8).take(MLXArray([Int32(2), 1, 0]), axis: 2)
  }
}

private final class ReCaMovieSource {
  let asset: AVURLAsset, track: AVAssetTrack, reader: AVAssetReader, output: AVAssetReaderTrackOutput
  let transform: CGAffineTransform, timeScale: CMTimeScale, range: CMTimeRange
  let audio: [AVAssetTrack], color: [String: Any]
  init(recipe: NativeReCaVSRRecipe) async throws {
    asset = AVURLAsset(url: URL(fileURLWithPath: recipe.source.path), options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
    guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw ReCaError.invalid("Input has no video track.") }
    self.track = track
    let (transform, timeScale, fallback, descriptions) = try await track.load(.preferredTransform, .naturalTimeScale, .timeRange, .formatDescriptions)
    self.transform = transform; self.timeScale = max(1, timeScale)
    var colors: [String: Any] = [:]
    if let desc = descriptions.first, let extensions = CMFormatDescriptionGetExtensions(desc) {
      let metadata = extensions as NSDictionary
      let transfer = (metadata[kCMFormatDescriptionExtension_TransferFunction] as? String ?? "").lowercased()
      guard !transfer.contains("2084"), !transfer.contains("hlg"), metadata["ContainsAlphaChannel"] as? Bool != true else {
        throw ReCaError.invalid("ReCaVSR currently supports SDR movies without alpha. Convert this source explicitly first.")
      }
      for (source, target) in [(kCMFormatDescriptionExtension_ColorPrimaries, AVVideoColorPrimariesKey),
        (kCMFormatDescriptionExtension_TransferFunction, AVVideoTransferFunctionKey),
        (kCMFormatDescriptionExtension_YCbCrMatrix, AVVideoYCbCrMatrixKey)] {
        if let value = metadata[source] as? String { colors[target] = value }
      }
    }
    color = colors
    let content = try await NativeUpscaleMovieIO.videoContentRange(track, fallback: fallback)
    if let interval = recipe.interval {
      let start = CMTime(seconds: interval.startSeconds, preferredTimescale: 600_000_000)
      let end = CMTime(seconds: interval.startSeconds + interval.durationSeconds, preferredTimescale: 600_000_000)
      range = CMTimeRange(start: max(content.start, start), end: min(content.end, end))
    } else { range = content }
    guard range.isValid, range.duration > .zero else { throw ReCaError.invalid("Selected source interval has no video.") }
    audio = try await asset.loadTracks(withMediaType: .audio)
    reader = try AVAssetReader(asset: asset); reader.timeRange = range
    output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else { throw ReCaError.invalid("Cannot decode the source movie.") }
    reader.add(output)
    guard reader.startReading() else { throw reader.error ?? ReCaError.invalid("Cannot read source video.") }
  }
  func next() throws -> ReCaMovieFrame? {
    while let sample = output.copyNextSampleBuffer() {
      if sample.numSamples == 0 { continue }
      guard let pixels = sample.imageBuffer, sample.presentationTimeStamp.isNumeric else { throw ReCaError.invalid("Source sample has no pixels or timestamp.") }
      return .init(pixels: pixels, timestamp: sample.presentationTimeStamp)
    }
    if reader.status == .failed { throw reader.error ?? ReCaError.invalid("Source decoding failed.") }
    return nil
  }
  deinit { reader.cancelReading() }
}

public enum ReCaMovieRunner {
  public static func preflight(_ recipe: NativeReCaVSRRecipe) async throws -> ReCaMemoryPlan {
    try recipe.verifyInputs(); try ReCaReleasedAssets.validate(recipe.assets, precision: recipe.settings.precision)
    let source = try await ReCaMovieSource(recipe: recipe)
    guard let first = try source.next() else { throw ReCaError.invalid("Source movie is empty.") }
    let geometry = try ReCaGeometry(sourceWidth: CVPixelBufferGetWidth(first.pixels), sourceHeight: CVPixelBufferGetHeight(first.pixels), scale: recipe.settings.scale)
    try NativeUpscaleMovieWriter.validateRaster(width: geometry.width, height: geometry.height)
    let budget = MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0
    guard budget > 0 else { throw ReCaError.invalid("ReCaVSR requires an Apple Metal GPU.") }
    return try ReCaMemoryPlan.admit(geometry: geometry, profile: recipe.settings.precision,
      residency: recipe.settings.residency, budgetBytes: min(budget, recipe.maximumActiveBytes))
  }
  public static func run(recipe: NativeReCaVSRRecipe,
    attentionScheduling: ReCaAttentionScheduling = .automatic, captureRGBDirectory: URL? = nil,
    progress: @Sendable (ReCaProgress) -> Void = { _ in }) async throws -> NativeReCaVSRReceipt {
    let began = Date()
    _ = try await preflight(recipe)
    let source = try await ReCaMovieSource(recipe: recipe)
    guard let first = try source.next() else { throw ReCaError.invalid("Source movie is empty.") }
    let geometry = try ReCaGeometry(sourceWidth: CVPixelBufferGetWidth(first.pixels), sourceHeight: CVPixelBufferGetHeight(first.pixels), scale: recipe.settings.scale)
    let output = URL(fileURLWithPath: recipe.outputPath), folder = output.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    let scratch = folder.appendingPathComponent(".recavsr-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: scratch) }
    let movie = scratch.appendingPathComponent("video.mov"), muxed = scratch.appendingPathComponent("muxed.mov")
    let writer = try NativeUpscaleMovieWriter(url: movie, width: geometry.width, height: geometry.height,
      timeScale: source.timeScale, transform: source.transform, scale: recipe.settings.scale, colorProperties: source.color)
    defer { writer.cancelIfWriting() }
    let capture: NativeRGBFrameCapture?
    if let captureRGBDirectory {
      let colors = try JSONSerialization.data(withJSONObject:source.color,options:.sortedKeys)
      capture = try NativeRGBFrameCapture(directory:captureRGBDirectory,width:geometry.width,height:geometry.height,
        metadata:["attentionMode":attentionScheduling.rawValue,"sourceSHA256":recipe.source.sha256,
          "workerSHA256":recipe.worker.sha256,"metalSHA256":recipe.metal.sha256,
          "timeScale":String(source.timeScale),"hevcQuality":String(NativeUpscaleMovieWriter.hevcQuality),
          "colorPropertiesJSON":String(decoding:colors,as:UTF8.self)])
    } else { capture = nil }
    let lease = try NativeInferenceLease.acquire(at: recipe.leasePath.map { URL(fileURLWithPath: $0) } ?? NativeInferenceLease.defaultURL,
      onWait: { progress(.init(phase: "Waiting for native inference slot", completedFrames: 0, seconds: Date().timeIntervalSince(began))) })
    defer { lease.release() }
    let previousLimit = Memory.memoryLimit, previousCache = Memory.cacheLimit
    Memory.memoryLimit = Int(min(recipe.maximumActiveBytes, UInt64(previousLimit)))
    Memory.cacheLimit = min(2 * 1024 * 1024 * 1024, Memory.memoryLimit / 8)
    Memory.peakMemory = 0
    defer { Memory.memoryLimit = previousLimit; Memory.cacheLimit = previousCache }
    let transformerFile = try ReCaCheckpoint(asset: recipe.assets[.transformer], component: .transformer, precision: recipe.settings.precision)
    let decoderFile = try ReCaCheckpoint(asset: recipe.assets[.flashDecoder], component: .flashDecoder)
    let promptFile = try ReCaCheckpoint(asset: recipe.assets[.prompt], component: .prompt)
    let prompt = try promptFile.tensor("positive")
    let normalization = try JSONDecoder().decode(ReCaVAENormalization.self,
      from: Data(contentsOf: URL(fileURLWithPath: recipe.assets[.vaeConfig].path)))
    let transformer = ReCaTransformerSession(geometry: geometry, seed: recipe.settings.seed, scheduling:attentionScheduling)
    let decoder = try ReCaFlashDecoder(geometry: geometry, normalization: normalization, scheduling:attentionScheduling)
    defer { transformer.close(); decoder.close() }
    var buffered: ReCaMovieFrame? = first, count = 0, emitted = 0, shotStart = 0
    var lastSource = first.timestamp, pending: (bytes: [UInt8], timestamp: CMTime)?
    func write(_ bytes: [UInt8], at timestamp: CMTime, until end: CMTime) throws {
      try capture?.appendRGB(bytes,at:timestamp,duration:end-timestamp)
      try writer.appendRGB(bytes, at: timestamp, duration: end - timestamp)
    }
    while let startFrame = buffered {
      try Task.checkCancellation()
      if recipe.resetFrames.contains(count) { transformer.reset(); decoder.reset(); shotStart = count }
      let needed = count == shotStart ? 21 : 8
      let nextCut = recipe.resetFrames.first(where: { $0 > count }) ?? Int.max
      let limit = min(needed, min((recipe.frameLimit ?? Int.max) - count, nextCut - count))
      if limit <= 0 { break }
      var frames = [startFrame]; buffered = nil
      while frames.count < limit {
        guard let frame = try source.next() else { break }
        guard frame.timestamp > frames.last!.timestamp else { throw ReCaError.invalid("Source PTS must increase strictly.") }
        frames.append(frame)
      }
      let rgb = try stacked(frames.map { try $0.rgb(width: geometry.sourceWidth, height: geometry.sourceHeight) })
      let lq = try ReCaImagePreparation.prepare(rgb, geometry: geometry, frames: needed)
      progress(.init(phase: "Loading transformer", completedFrames: emitted, seconds: Date().timeIntervalSince(began)))
      if transformer.weights == nil { try transformer.loadWeights(from: transformerFile, prompt: prompt) }
      progress(.init(phase: "Transformer", completedFrames: emitted, seconds: Date().timeIntervalSince(began)))
      let latent = try transformer.step(lq: lq)
      if recipe.settings.residency == .staged { transformer.unloadWeights() }
      try decoder.loadWeights(from: decoderFile)
      var decodedOffset = 0
      try decoder.decode(latents: latent, lq: lq) { decoded in
        let take = min(decoded.dim(0), frames.count - decodedOffset)
        if take <= 0 { return }
        let visible = decoded[0..<take, 0..<geometry.height, 0..<geometry.width, 0..<3]
        let pixels: MLXArray
        switch recipe.settings.colorFix {
        case .none: pixels = ReCaColorCorrection.quantizeNormalized(visible)
        case .adain: pixels = try ReCaColorCorrection.quantizeNormalized(ReCaColorCorrection.adain(decodedNormalized: visible,
          enlargedLQNormalized: lq[decodedOffset..<decodedOffset + take, 0..<geometry.height, 0..<geometry.width]))
        case .wavelet: pixels = try ReCaColorCorrection.wavelet(decodedNormalized: visible,
          originalLQUInt8: rgb[decodedOffset..<decodedOffset + take])
        }
        eval(pixels)
        for i in 0..<take {
          let timestamp = frames[decodedOffset + i].timestamp - first.timestamp
          if let pending { try write(pending.bytes, at: pending.timestamp, until: timestamp) }
          pending = (pixels[i].asArray(UInt8.self), timestamp); emitted += 1
        }
        decodedOffset += take
      }
      if recipe.settings.residency == .staged { decoder.unloadWeights() }
      count += frames.count; lastSource = frames.last!.timestamp
      progress(.init(phase: "Encoded frames", completedFrames: emitted, seconds: Date().timeIntervalSince(began)))
      buffered = try source.next()
      if let frame = buffered, frame.timestamp <= lastSource { throw ReCaError.invalid("Source PTS must increase strictly.") }
      if count >= (recipe.frameLimit ?? Int.max) { break }
    }
    guard emitted == count, let pending, recipe.resetFrames.allSatisfy({ $0 < count }) else { throw ReCaError.invalid("ReCaVSR frame accounting or reset boundary is invalid.") }
    let finalEnd = buffered?.timestamp ?? NativeUpscaleMovieIO.finalVideoEnd(source.asset, track: source.track, last: lastSource, editEnd: source.range.end)
    let relativeEnd = finalEnd - first.timestamp
    try write(pending.bytes, at: pending.timestamp, until: relativeEnd)
    try capture?.finish()
    try await writer.finish(at: relativeEnd)
    let audioStart = recipe.interval.map { CMTime(seconds: $0.startSeconds, preferredTimescale: 600_000_000) }
    let sourceIn = try await NativeUpscaleMovieIO.mux(movie, audio: source.audio, sourceStart: first.timestamp,
      audioStart: audioStart, previewEnd: buffered?.timestamp ?? (recipe.interval == nil ? nil : source.range.end), output: muxed)
    try Task.checkCancellation()
    try await verifyVideo(muxed, recipe: recipe, geometry: geometry, count: count, sourceIn: sourceIn)
    transformer.close(); decoder.close(); Memory.clearCache()
    try FileManager.default.moveItem(at: muxed, to: output)
    do {
      let receipt = try NativeReCaVSRReceipt(recipe: recipe, video: ReCaFileIdentity.capture(output), frames: count,
        width: geometry.width, height: geometry.height, sourceIn: sourceIn.seconds, duration: relativeEnd.seconds,
        seconds: Date().timeIntervalSince(began), peakMLXBytes: UInt64(Memory.peakMemory), activeAfterStagesBytes: UInt64(Memory.activeMemory))
      let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
      try encoder.encode(receipt).write(to: output.appendingPathExtension("json"), options: .withoutOverwriting)
      return receipt
    } catch { try? FileManager.default.removeItem(at: output); throw error }
  }
  private static func verifyVideo(_ video: URL, recipe: NativeReCaVSRRecipe, geometry: ReCaGeometry, count: Int, sourceIn: CMTime) async throws {
    let original = try await ReCaMovieSource(recipe: recipe)
    let asset = AVURLAsset(url: video, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
    guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw ReCaError.invalid("Encoded movie has no video track.") }
    guard try await track.load(.naturalSize) == CGSize(width: geometry.width, height: geometry.height) else {
      throw ReCaError.invalid("The encoder changed the requested output raster.")
    }
    let range = try await NativeUpscaleMovieIO.videoContentRange(track, fallback: track.load(.timeRange))
    let reader = try AVAssetReader(asset: asset); reader.timeRange = range
    let stream = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(stream); guard reader.startReading() else { throw ReCaError.invalid("Cannot verify encoded movie.") }
    defer { reader.cancelReading() }
    var first: CMTime?, seen = 0
    while let output = stream.copyNextSampleBuffer() {
      if output.numSamples == 0 { continue }
      guard seen < count, let input = try original.next() else { throw ReCaError.invalid("Encoded movie frame count changed.") }
      if first == nil { first = input.timestamp }
      guard output.presentationTimeStamp == input.timestamp - first! + sourceIn else { throw ReCaError.invalid("Encoded movie changed a source presentation timestamp.") }
      seen += 1
    }
    guard reader.status == .completed, seen == count else { throw ReCaError.invalid("Encoded movie did not verify completely.") }
  }
}
