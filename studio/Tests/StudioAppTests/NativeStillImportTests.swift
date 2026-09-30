import AVFoundation
import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeStillImportTests: XCTestCase {
  private func wav(at url: URL) throws {
    let samples = 16_000
    let byteCount = samples * 2
    var data = Data()
    func word(_ value: UInt16) {
      data.append(UInt8(truncatingIfNeeded: value))
      data.append(UInt8(truncatingIfNeeded: value >> 8))
    }
    func dword(_ value: UInt32) {
      word(UInt16(truncatingIfNeeded: value))
      word(UInt16(truncatingIfNeeded: value >> 16))
    }
    data.append(contentsOf: "RIFF".utf8); dword(UInt32(36 + byteCount))
    data.append(contentsOf: "WAVEfmt ".utf8); dword(16)
    word(1); word(1); dword(16_000); dword(32_000); word(2); word(16)
    data.append(contentsOf: "data".utf8); dword(UInt32(byteCount))
    data.append(Data(count: byteCount))
    try data.write(to: url)
  }

  @MainActor func testAudioReferenceImportDoesNotInvokePythonBridge() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let audio = directory.appendingPathComponent("voice.wav")
    try wav(at: audio)
    let store = StudioStore(dataDirectory: directory, restoreSession: false,
      invocation: { _, _, _, _ in
        XCTFail("Audio-reference import must not start the Python bridge.")
        throw StudioError.invalid("Unexpected bridge call")
      })
    store.runtime.pythonPath = "/missing/python"
    let clip = Clip(engine: .h3)
    store.project.clips = [clip]
    store.selectedClipID = clip.id
    await store.importURLs([audio], scope: .clip)
    XCTAssertNil(store.error)
    let asset = try XCTUnwrap(store.project.assets.first)
    XCTAssertEqual(asset.kind, .audio)
    XCTAssertEqual(asset.duration, 1, accuracy: 0.01)
    XCTAssertEqual(asset.path, audio.path)
  }

  @MainActor func testVideoReferenceImportDoesNotInvokePythonBridge() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let movie = directory.appendingPathComponent("reference.mov")
    let writer = try AVAssetWriter(outputURL: movie, fileType: .mov)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: 32, AVVideoHeightKey: 32])
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32ARGB),
        kCVPixelBufferWidthKey as String: 32,
        kCVPixelBufferHeightKey as String: 32])
    writer.add(input)
    XCTAssertTrue(writer.startWriting())
    writer.startSession(atSourceTime: .zero)
    for frame in 0..<2 {
      var buffer: CVPixelBuffer?
      XCTAssertEqual(CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault,
        try XCTUnwrap(adaptor.pixelBufferPool), &buffer), kCVReturnSuccess)
      XCTAssertTrue(adaptor.append(try XCTUnwrap(buffer),
        withPresentationTime: CMTime(value: Int64(frame), timescale: 24)))
    }
    input.markAsFinished()
    await writer.finishWriting()
    XCTAssertEqual(writer.status, .completed)
    let store = StudioStore(dataDirectory: directory, restoreSession: false,
      invocation: { _, _, _, _ in
        XCTFail("Movie-reference import must not start the Python bridge.")
        throw StudioError.invalid("Unexpected bridge call")
      })
    store.runtime.pythonPath = "/missing/python"
    let clip = Clip(engine: .h3)
    store.project.clips = [clip]
    store.selectedClipID = clip.id
    await store.importURLs([movie], scope: .clip)
    XCTAssertNil(store.error)
    let asset = try XCTUnwrap(store.project.assets.first)
    XCTAssertEqual(asset.kind, .video)
    XCTAssertEqual(asset.width, 32)
    XCTAssertEqual(asset.height, 32)
    XCTAssertEqual(asset.fps, 24, accuracy: 0.1)
    XCTAssertGreaterThan(asset.duration, 0)
  }

  @MainActor func testImageImportDoesNotInvokePythonBridge() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let image = directory.appendingPathComponent("reference.png")
    let png = try XCTUnwrap(Data(base64Encoded:
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+lWZkAAAAASUVORK5CYII="))
    try png.write(to: image)
    let store = StudioStore(dataDirectory: directory, restoreSession: false,
      invocation: { _, _, _, _ in
        XCTFail("Still-image import must not start the Python bridge.")
        throw StudioError.invalid("Unexpected bridge call")
      })
    store.runtime.pythonPath = "/missing/python"
    let clip = Clip(engine: .h3)
    store.project.clips = [clip]
    store.selectedClipID = clip.id
    await store.importURLs([image], scope: .clip)
    XCTAssertNil(store.error)
    let asset = try XCTUnwrap(store.project.assets.first)
    XCTAssertEqual(asset.kind, .image)
    XCTAssertEqual(asset.width, 1)
    XCTAssertEqual(asset.height, 1)
    XCTAssertEqual(asset.path, image.path)
  }
}
