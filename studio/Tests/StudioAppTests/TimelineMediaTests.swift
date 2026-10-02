import AVFoundation
import AppKit
import StudioCore
import XCTest
@testable import WeeToddStudio

final class TimelineMediaTests: XCTestCase {
  @MainActor func testSwiftH3ExtensionAcceptsNewFramesWithoutPythonAndReopens() async throws {
    let folder=try directory()
    let generated=ProcessInfo.processInfo.environment["WEETODD_H3_EXTENSION_RESULT_MOVIE"]
    let result:URL
    if let generated { result=URL(fileURLWithPath:generated) }
    else { result=try await audiovisualMovie(in:folder) }
    let invocation:Bridge.Invocation = { command,_,_,_ in
      guard command == "h3-native-render" || command == "h3-native-describe" else {
        throw StudioError.invalid("Unexpected non-native command: \(command)")
      }
      return command == "h3-native-render" ? ["video":result.path,"nativeRuntime":"swift-mlx"] : [:]
    }
    let store=StudioStore(dataDirectory:folder,restoreSession:false,invocation:invocation)
    store.runtime.nativeH3Enabled=true;store.runtime.pythonPath="/unavailable/python"
    var source=Clip(engine:.h3);source.sourcePath=result.path;source.duration=2.5
    var target=Clip(engine:.h3);target.duration=generated == nil ? 2.5 : 4
    if let original=ProcessInfo.processInfo.environment["WEETODD_H3_EXTENSION_EDITOR_PROJECT"] {
      store.project=try ProjectStorage.read(URL(fileURLWithPath:original))
      source=try XCTUnwrap(store.project.clips.first)
      target=try XCTUnwrap(store.project.clips.last)
      XCTAssertEqual(target.extensionClipID,source.id)
    } else {
      target.extensionDirection="after";target.extensionSource=source.sourcePath;target.extensionClipID=source.id
      store.project.clips=[source,target]
    }
    store.selectedClipID=target.id
    store.preparedRecipe=ProcessInfo.processInfo.environment["WEETODD_H3_EXTENSION_PREPARED_RECIPE"] ?? "/tmp/prepared/native-extension.json"
    store.preparedFingerprint=store.signature(for:target)
    await store.renderPrepared()
    XCTAssertNil(store.error)
    XCTAssertEqual(store.project.clips[0],source)
    let accepted=try XCTUnwrap(store.selectedClip),take=try XCTUnwrap(accepted.versions.last)
    XCTAssertEqual(accepted.sourceIn,0);XCTAssertEqual(accepted.duration,target.duration)
    XCTAssertEqual(take.usableSourceIn,0)
    let saved=folder.appendingPathComponent("extension.weetodd")
    try ProjectStorage.write(store.project,to:saved)
    let reopened=StudioStore(dataDirectory:folder,restoreSession:false);reopened.load(saved)
    XCTAssertEqual(reopened.project.clips[1],accepted)
    if let evidence=ProcessInfo.processInfo.environment["WEETODD_H3_EXTENSION_ACCEPTANCE_EVIDENCE"] {
      let evidenceURL=URL(fileURLWithPath:evidence)
      let retained=evidenceURL.deletingLastPathComponent().appendingPathComponent("accepted-corrected.weetodd")
      try ProjectStorage.write(store.project,to:retained)
      let body:[String:Any]=["video":result.path,"sourceIn":accepted.sourceIn,"duration":accepted.duration,
        "acceptedProject":retained.path,"pythonPath":store.runtime.pythonPath,"renderReceiptReplayed":true,
        "acceptanceAndReopening":"passed"]
      try JSONSerialization.data(withJSONObject:body,options:[.prettyPrinted,.sortedKeys]).write(to:evidenceURL)
    }
  }
  func audiovisualMovie(in folder:URL) async throws -> URL {
    let video=try await movie(in:folder),sound=try audio(in:folder)
    let ffmpeg=URL(fileURLWithPath:ProcessInfo.processInfo.environment["WEETODD_TEST_FFMPEG"] ?? "/opt/homebrew/bin/ffmpeg")
    guard FileManager.default.isExecutableFile(atPath:ffmpeg.path) else { throw XCTSkip("Audiovisual fixture needs FFmpeg") }
    let result=folder.appendingPathComponent("audiovisual.mp4")
    let mux=Process();mux.executableURL=ffmpeg
    mux.arguments=["-hide_banner","-loglevel","error","-i",video.path,"-i",sound.path,
      "-c:v","copy","-c:a","aac","-shortest",result.path]
    try mux.run();mux.waitUntilExit();XCTAssertEqual(mux.terminationStatus,0)
    return result
  }
  @MainActor func testNativeTakesPlayTheirTrimmedSoundWithoutPythonMixer() async throws {
    let folder=try directory(),movie=try await audiovisualMovie(in:folder)
    var mixCalls=0
    let store=StudioStore(dataDirectory:folder,restoreSession:false,invocation: { command,_,_,_ in
      if command == "audio-mix" { mixCalls += 1;throw StudioError.invalid("Python must not be required for source sound") }
      return [:]
    })
    store.runtime.pythonPath="/unavailable/python"
    store.runtime.nativeH3Enabled=true;store.runtime.nativeLTX25Enabled=true
    var a=Clip(engine:.h3);a.sourcePath=movie.path;a.duration=0.4;a.volume=0.25
    var b=Clip(engine:.ltx25);b.sourcePath=movie.path;b.sourceIn=1;b.duration=0.4;b.volume=0.5
    store.project.clips=[a,b];store.select(a.id)
    await store.timelineBuildTask?.value
    XCTAssertEqual(mixCalls,0);XCTAssertNil(store.timelinePlaybackWarning)
    let item=try XCTUnwrap(store.player.currentItem)
    let tracks=try await item.asset.loadTracks(withMediaType:.audio)
    XCTAssertEqual(tracks.count,1)
    let mix=try XCTUnwrap(item.audioMix?.inputParameters.first)
    for (time,volume) in [(0.1,Float(0.25)),(0.5,Float(0.5))] {
      var start:Float=0,end:Float=0,range=CMTimeRange.zero
      XCTAssertTrue(mix.getVolumeRamp(for:TimelinePlaybackBuilder.time(time),startVolume:&start,endVolume:&end,timeRange:&range))
      XCTAssertEqual(start,volume,accuracy:0.001);XCTAssertEqual(end,volume,accuracy:0.001)
    }
    for _ in 0..<100 where item.status == .unknown { try await Task.sleep(nanoseconds:20_000_000) }
    XCTAssertEqual(item.status,.readyToPlay,item.error?.localizedDescription ?? "")
    store.togglePlayback()
    for _ in 0..<100 where store.isPlaying { try await Task.sleep(nanoseconds:20_000_000) }
    XCTAssertEqual(store.playhead,0.8,accuracy:0.001);XCTAssertFalse(store.isPlaying)
  }
  @MainActor func testSourcePanAndRegionsStillUseCanonicalMixer() async throws {
    let folder=try directory(),movie=try await audiovisualMovie(in:folder),mixed=try audio(in:folder)
    var mixCalls=0
    let store=StudioStore(dataDirectory:folder,restoreSession:false,invocation: { command,_,_,_ in
      XCTAssertEqual(command,"audio-mix");mixCalls += 1;return ["path":mixed.path]
    })
    var clip=Clip(engine:.h3);clip.sourcePath=movie.path;clip.duration=0.4;clip.sourcePan=0.5
    store.project.clips=[clip]
    store.prepareTimelinePlayback(force:true);await store.timelineBuildTask?.value
    XCTAssertEqual(mixCalls,1);XCTAssertNotNil(store.timelineAudioLease)
    store.project.clips[0].sourcePan=0
    var region=AudioRegion(assetID:UUID(),path:mixed.path);region.duration=0.4
    store.project.audio=[region]
    store.prepareTimelinePlayback(force:true);await store.timelineBuildTask?.value
    XCTAssertEqual(mixCalls,2);XCTAssertNotNil(store.timelineAudioLease)
    XCTAssertNil(store.timelinePlaybackWarning)
  }
  @MainActor func testReplacingPlayableTimelineWithUnrenderedShotReleasesOldPlayer() async throws {
    let folder=try directory(),movie=try await audiovisualMovie(in:folder)
    let store=StudioStore(dataDirectory:folder,restoreSession:false,invocation: { _,_,_,_ in
      XCTFail("No soundtrack is required for source-only cuts");return [:]
    })
    var clip=Clip(engine:.h3);clip.sourcePath=movie.path;clip.duration=0.4
    store.project.clips=[clip];store.prepareTimelinePlayback(force:true);await store.timelineBuildTask?.value
    XCTAssertNotNil(store.player.currentItem)
    clip.sourcePath="";store.project.clips=[clip]
    store.prepareTimelinePlayback(force:true);await store.timelineBuildTask?.value
    XCTAssertNil(store.player.currentItem)
    store.togglePlayback()
    for _ in 0..<100 where store.isPlaying { try await Task.sleep(nanoseconds:20_000_000) }
    XCTAssertEqual(store.playhead,0.4,accuracy:0.001);XCTAssertFalse(store.isPlaying)
  }
  func testSourceAudioPreviewRejectsOverlapsAndUnsupportedGain() {
    var project=StudioProject(),a=Clip(),b=Clip()
    a.duration=1;b.duration=1;project.clips=[a,b]
    XCTAssertTrue(TimelinePlaybackBuilder.canUseSourceAudio(project))
    project.clips[1].transition="dissolve";project.clips[1].transitionDuration=0.2
    XCTAssertFalse(TimelinePlaybackBuilder.canUseSourceAudio(project))
    project.clips[1].transition="cut";project.clips[0].volume=3
    XCTAssertFalse(TimelinePlaybackBuilder.canUseSourceAudio(project))
  }
  func directory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url
  }

  @MainActor func playbackStore(in folder: URL) throws -> StudioStore {
    // Media composition tests inject a prepared soundtrack; Python mixer tests verify its PCM.
    let mixed = try audio(in: folder)
    let invocation: Bridge.Invocation = { command, _, payload, _ in
      if command == "audio-mix" {
        XCTAssertEqual(payload["purpose"] as? String, "preview")
        return ["path": mixed.path]
      }
      return [:]
    }
    return StudioStore(dataDirectory: folder, restoreSession: false, invocation: invocation)
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
    let store = try playbackStore(in: folder)
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
    let store = try playbackStore(in: folder)
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
    XCTAssertEqual(start, 1); XCTAssertEqual(end, 1, accuracy: 0.001) // Fades are baked once into canonical PCM.
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
    let store = try playbackStore(in: folder)
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

extension TimelineMediaTests {
  @MainActor func testClearingTimelineStopsOldPreparedItem() async throws {
    let folder = try directory(), source = try await audiovisualMovie(in: folder)
    let store = try playbackStore(in: folder)
    var clip = Clip(); clip.duration = 2; clip.sourcePath = source.path; store.project.clips = [clip]
    store.prepareTimelinePlayback(); await store.timelineBuildTask?.value
    XCTAssertNotNil(store.player.currentItem)
    store.togglePlayback(); XCTAssertTrue(store.isPlaying)
    store.project.clips = []
    store.prepareTimelinePlayback(force: true)
    XCTAssertNil(store.player.currentItem); XCTAssertFalse(store.isPlaying)
  }
}
