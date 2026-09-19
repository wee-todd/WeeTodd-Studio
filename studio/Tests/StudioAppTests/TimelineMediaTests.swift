import AVFoundation
import AppKit
import StudioCore
import XCTest
@testable import WeeToddStudio

final class TimelineMediaTests: XCTestCase {
  func directory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url
  }

  /// Three one-second color blocks let decoded pixels prove both source trims and timeline cuts.
  func movie(in directory: URL) async throws -> URL {
    let url = directory.appendingPathComponent("colors.mov")
    let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
      AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 64, AVVideoHeightKey: 64])
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
      kCVPixelBufferWidthKey as String: 64, kCVPixelBufferHeightKey as String: 64])
    writer.add(input)
    XCTAssertTrue(writer.startWriting())
    writer.startSession(atSourceTime: .zero)
    for frame in 0..<30 {
      while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
      var buffer: CVPixelBuffer?
      CVPixelBufferCreate(kCFAllocatorDefault, 64, 64, kCVPixelFormatType_32ARGB, nil, &buffer)
      let pixelBuffer = try XCTUnwrap(buffer)
      CVPixelBufferLockBaseAddress(pixelBuffer, [])
      let bytes = CVPixelBufferGetBaseAddress(pixelBuffer)!.assumingMemoryBound(to: UInt8.self)
      let stride = CVPixelBufferGetBytesPerRow(pixelBuffer)
      for y in 0..<64 { for x in 0..<64 {
        let offset = y * stride + x * 4
        bytes[offset] = 255
        for channel in 0..<3 { bytes[offset + 1 + channel] = channel == frame / 10 ? 255 : 0 }
      } }
      CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
      XCTAssertTrue(adaptor.append(pixelBuffer, withPresentationTime: CMTime(value: Int64(frame), timescale: 10)))
    }
    writer.endSession(atSourceTime: CMTime(seconds: 3, preferredTimescale: 600))
    input.markAsFinished()
    await writer.finishWriting()
    XCTAssertEqual(writer.status, .completed)
    return url
  }

  func testCompositionDecodesTrimmedShotsAcrossMissingClipWithoutMovingLaterShots() async throws {
    let url = try await movie(in: directory())
    var project = StudioProject(); project.settings.width = 64; project.settings.height = 64
    var a = Clip(name: "Green", engine: .movie)
    a.sourcePath = url.path; a.sourceIn = 1; a.duration = 1
    var missing = Clip(name: "Not generated"); missing.duration = 0.5
    var b = Clip(name: "Blue", engine: .movie)
    b.sourcePath = url.path; b.sourceIn = 2; b.duration = 1
    project.clips = [a, missing, b]
    let media = try await TimelinePlaybackBuilder.build(TimelinePlaybackPlan(project: project))
    XCTAssertTrue(media.hasMedia)
    XCTAssertTrue(media.unavailableClips.isEmpty)
    XCTAssertEqual(media.composition.duration.seconds, 2.5, accuracy: 0.001)
    let image = AVAssetImageGenerator(asset: media.composition)
    image.videoComposition = media.videoComposition
    image.requestedTimeToleranceBefore = .zero; image.requestedTimeToleranceAfter = .zero
    for (seconds, channel) in [(0.2, 1), (1.7, 2)] {
      let result = try await image.image(at: CMTime(seconds: seconds, preferredTimescale: 600))
      let color = try XCTUnwrap(NSBitmapImageRep(cgImage: result.image).colorAt(x: 32, y: 32)?.usingColorSpace(.deviceRGB))
      let components = [color.redComponent, color.greenComponent, color.blueComponent]
      XCTAssertGreaterThan(components[channel], 0.8)
      XCTAssertLessThan(components[(channel + 1) % 3], 0.2)
    }
  }

  func testFractionalFrameCutsHaveNoCompositorGaps() async throws {
    let url = try await movie(in: directory())
    var project = StudioProject(); project.settings.width = 64; project.settings.height = 64
    project.settings.fps = 24
    project.clips = (0..<40).map { index in
      var clip = Clip(engine: .movie)
      clip.sourcePath = url.path
      clip.duration = Double([23, 41, 17, 13, 29][index % 5]) / 24
      return clip
    }
    let media = try await TimelinePlaybackBuilder.build(TimelinePlaybackPlan(project: project))
    let composition = try XCTUnwrap(media.videoComposition)
    var end = CMTime.zero
    for instruction in composition.instructions {
      XCTAssertEqual(CMTimeCompare(instruction.timeRange.start, end), 0,
        "Even one timeline tick of empty space can black out AVPlayer's entire video composition")
      XCTAssertGreaterThan(CMTimeCompare(instruction.timeRange.duration, .zero), 0)
      end = CMTimeRangeGetEnd(instruction.timeRange)
    }
    XCTAssertEqual(CMTimeCompare(end, media.composition.duration), 0)
    let valid = try await composition.isValid(for: media.composition,
      timeRange: CMTimeRange(start: .zero, duration: media.composition.duration), validationDelegate: nil)
    XCTAssertTrue(valid)
  }

  @MainActor func testNativePlayerCrossesClipBoundaryAndStopsAtTimelineEnd() async throws {
    let folder = try directory()
    let url = try await movie(in: folder)
    let store = StudioStore(dataDirectory: folder, restoreSession: false)
    var a = Clip(name: "First", engine: .movie); a.sourcePath = url.path; a.duration = 0.3
    var b = Clip(name: "Second", engine: .movie); b.sourcePath = url.path; b.sourceIn = 2; b.duration = 0.3
    store.project.clips = [a, b]
    store.select(a.id)
    await store.timelineBuildTask?.value
    let item = try XCTUnwrap(store.player.currentItem)
    for _ in 0..<100 where item.status == .unknown { try await Task.sleep(nanoseconds: 20_000_000) }
    XCTAssertEqual(item.status, .readyToPlay, item.error?.localizedDescription ?? "")
    store.togglePlayback()
    for _ in 0..<100 where store.isPlaying { try await Task.sleep(nanoseconds: 20_000_000) }
    XCTAssertEqual(store.playhead, 0.6, accuracy: 0.001)
    XCTAssertEqual(store.previewClip?.id, b.id)
    XCTAssertEqual(store.selectedClipID, a.id)
    XCTAssertFalse(store.isPlaying)
    store.select(b.id)
    XCTAssertTrue(store.player.currentItem === item, "Selecting a clip must reuse the timeline composition")
    store.seek(0.1); store.seek(0.5); store.seek(0.2)
    for _ in 0..<100 where store.timelineSeekPending { try await Task.sleep(nanoseconds: 20_000_000) }
    XCTAssertEqual(store.player.currentTime().seconds, 0.2, accuracy: 0.02)
    XCTAssertEqual(store.playhead, 0.2)
  }
}

extension TimelineMediaTests {
  func audio(in folder: URL) throws -> URL {
    let url = folder.appendingPathComponent("silence.caf")
    let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1))
    let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 144000))
    buffer.frameLength = 144000
    buffer.floatChannelData![0].initialize(repeating: 0, count: 144000)
    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    try file.write(from: buffer)
    return url
  }

  @MainActor func testAudioOnlyCompositionPlaysUnrenderedTimelineWithRegionFades() async throws {
    let folder = try directory()
    let sound = try audio(in: folder)
    let store = StudioStore(dataDirectory: folder, restoreSession: false)
    var clip = Clip(); clip.duration = 0.4
    var region = AudioRegion(assetID: UUID(), path: sound.path)
    region.start = 0; region.sourceIn = 1; region.duration = 0.4; region.volume = 0.6; region.fade = 0.1
    store.project.clips = [clip]; store.project.audio = [region]
    store.select(clip.id)
    await store.timelineBuildTask?.value
    let item = try XCTUnwrap(store.player.currentItem)
    let mix = try XCTUnwrap(item.audioMix?.inputParameters.last)
    var start: Float = 0; var end: Float = 0; var range = CMTimeRange.zero
    XCTAssertTrue(mix.getVolumeRamp(for: TimelinePlaybackBuilder.time(0.05), startVolume: &start, endVolume: &end, timeRange: &range))
    XCTAssertEqual(start, 0); XCTAssertEqual(end, 0.6, accuracy: 0.001)
    for _ in 0..<100 where item.status == .unknown { try await Task.sleep(nanoseconds: 20_000_000) }
    XCTAssertEqual(item.status, .readyToPlay, item.error?.localizedDescription ?? "")
    store.togglePlayback()
    for _ in 0..<100 where store.isPlaying { try await Task.sleep(nanoseconds: 20_000_000) }
    XCTAssertEqual(store.playhead, 0.4, accuracy: 0.001)
    XCTAssertFalse(store.isPlaying)
  }

  @MainActor func testEmptyLeadAndTailKeepTheirPositionsDuringRealVideoPlayback() async throws {
    let folder = try directory()
    let url = try await movie(in: folder)
    let store = StudioStore(dataDirectory: folder, restoreSession: false)
    var blank = Clip(); blank.duration = 0.2
    var video = Clip(engine: .movie); video.sourcePath = url.path; video.sourceIn = 1; video.duration = 0.2
    var tail = Clip(); tail.duration = 0.2
    store.project.clips = [blank, video, tail]
    store.select(blank.id)
    await store.timelineBuildTask?.value
    let item = try XCTUnwrap(store.player.currentItem)
    for _ in 0..<100 where item.status == .unknown { try await Task.sleep(nanoseconds: 20_000_000) }
    XCTAssertEqual(item.status, .readyToPlay, item.error?.localizedDescription ?? "")
    store.togglePlayback()
    for _ in 0..<100 where store.isPlaying { try await Task.sleep(nanoseconds: 20_000_000) }
    XCTAssertEqual(store.playhead, 0.6, accuracy: 0.001)
    XCTAssertFalse(store.isPlaying)
    XCTAssertEqual(store.previewClip?.id, tail.id)
  }
}
