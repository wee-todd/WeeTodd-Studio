import AVFoundation
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import LTX25Audio
@testable import WeeToddLTXPipelineProbe

final class PipelineProbeTests: XCTestCase {
  private var validRequest: [String: Any] {
    ["gemma_root": "/models/gemma", "transformer_root": "/models/transformer",
      "connector_checkpoint": "/models/connector.safetensors", "video_checkpoint": "/models/video.safetensors",
      "audio_checkpoint": "/models/audio.safetensors", "prompt": "A red fox.",
      "width": 64, "height": 64, "frames": 9, "fps": 24.0, "seed": 42,
      "output_directory": "/outputs/new-run"]
  }

  private func withDirectory(_ body: (URL) throws -> Void) throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    try body(directory)
  }

  private func saveRequest(_ values: [String: Any], at url: URL) throws {
    try JSONSerialization.data(withJSONObject: values, options: [.sortedKeys]).write(to: url)
  }

  func testLoadRequestAcceptsOnlyItsCompleteExplicitContract() throws {
    try withDirectory { directory in
      let url = directory.appendingPathComponent("request.json")
      try saveRequest(validRequest, at: url)
      let decoded = try PipelineProbe.loadRequest(at: url)
      XCTAssertEqual(decoded.prompt, "A red fox.")
      XCTAssertEqual(decoded.seed, 42)
      XCTAssertEqual(decoded.width, 64)
      XCTAssertEqual(decoded.height, 64)
      XCTAssertEqual(decoded.frames, 9)
      XCTAssertEqual(decoded.fps, 24)
      XCTAssertEqual(decoded.transformerRoot, "/models/transformer")
      XCTAssertEqual(decoded.outputDirectory, "/outputs/new-run")
      // Missing required fields and unsupported inference controls must fail
      // before any checkpoint access or execution can begin.
      for name in validRequest.keys {
        var missing = validRequest; missing.removeValue(forKey: name)
        try saveRequest(missing, at: url)
        XCTAssertThrowsError(try PipelineProbe.loadRequest(at: url), "Missing \(name) was accepted")
      }
      for name in ["lora", "image", "steps", "eta", "backend"] {
        var unknown = validRequest; unknown[name] = "unsupported"
        try saveRequest(unknown, at: url)
        XCTAssertThrowsError(try PipelineProbe.loadRequest(at: url), "Unknown \(name) was accepted")
      }
    }
  }

  func testLoadRequestRejectsRelativeAndInvalidPathsForEveryPathField() throws {
    try withDirectory { directory in
      let url = directory.appendingPathComponent("request.json")
      for field in ["gemma_root", "transformer_root", "connector_checkpoint", "video_checkpoint",
        "audio_checkpoint", "output_directory"] {
        for value in ["relative/path", "~/models", "/models/\u{0}invalid", "/" + String(repeating: "x", count: 4096)] {
          var request = validRequest; request[field] = value
          try saveRequest(request, at: url)
          XCTAssertThrowsError(try PipelineProbe.loadRequest(at: url), "Invalid path accepted in \(field)")
        }
      }
    }
  }

  func testLoadRequestRejectsEmptyWhitespaceAndOversizedPrompts() throws {
    try withDirectory { directory in
      let url = directory.appendingPathComponent("request.json")
      for prompt in ["", " \n\t\u{2003}", String(repeating: "x", count: 128 * 1024 + 1)] {
        var request = validRequest; request["prompt"] = prompt
        try saveRequest(request, at: url)
        XCTAssertThrowsError(try PipelineProbe.loadRequest(at: url))
      }
    }
  }

  func testLoadRequestChecksRegularFileAndInclusiveByteLimit() throws {
    try withDirectory { directory in
      XCTAssertThrowsError(try PipelineProbe.loadRequest(at: directory))
      let url = directory.appendingPathComponent("request.json")
      var data = try JSONSerialization.data(withJSONObject: validRequest, options: [.sortedKeys])
      // JSON permits trailing whitespace, so the failure beyond the bound is
      // the request byte limit rather than a malformed JSON document.
      data.append(Data(repeating: UInt8(ascii: " "), count: 256 * 1024 - data.count))
      try data.write(to: url)
      XCTAssertEqual(try PipelineProbe.loadRequest(at: url).prompt, "A red fox.")
      data.append(UInt8(ascii: " "))
      try data.write(to: url)
      XCTAssertThrowsError(try PipelineProbe.loadRequest(at: url))
    }
  }

  func testPNGIsReadableAtRequestedDimensionsAndCannotBeOverwritten() throws {
    try withDirectory { directory in
      let url = directory.appendingPathComponent("frame.png")
      let rgb: [Float] = [1,-1,-1, -1,1,-1, -1,-1,1, 0,0,0, 1,1,1, -1,-1,-1]
      try PipelineProbe.writePNG(rgb, width: 3, height: 2, to: url)
      let original = try Data(contentsOf: url)
      let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
      XCTAssertEqual(CGImageSourceGetType(source) as String?, UTType.png.identifier)
      XCTAssertEqual(CGImageSourceGetCount(source), 1)
      let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
      XCTAssertEqual(image.width, 3)
      XCTAssertEqual(image.height, 2)
      XCTAssertEqual(image.bitsPerComponent, 8)
      XCTAssertThrowsError(try PipelineProbe.writePNG([Float](repeating: 0, count: 18), width: 3, height: 2, to: url))
      XCTAssertEqual(try Data(contentsOf: url), original)
    }
  }

  func testPNGRejectsInvalidPayloadBeforeCreatingOutput() throws {
    try withDirectory { directory in
      let url = directory.appendingPathComponent("frame.png")
      XCTAssertThrowsError(try PipelineProbe.writePNG([0,0], width: 1, height: 1, to: url))
      XCTAssertThrowsError(try PipelineProbe.writePNG([0,.nan,0], width: 1, height: 1, to: url))
      XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }
  }

  func testWAVIndependentParserReadsFloatStereoAndExactUnpaddedSampleCount() throws {
    try withDirectory { directory in
      let url = directory.appendingPathComponent("audio.wav")
      let left: [Float] = [0.125,-0.25,0.5,0.75,-1]
      let right: [Float] = [-0.875,0.25,0,-0.5,1]
      let waveform = AudioWaveform(samples: left + right, sampleRate: 48000, channels: 2)
      try PipelineProbe.writeWAV(waveform, to: url)
      // AVFoundation parses the RIFF/fmt/fact/data contract independently of
      // the writer. Distinct channels catch accidental planar WAV payloads.
      let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
      XCTAssertEqual(file.fileFormat.sampleRate, 48000)
      XCTAssertEqual(file.fileFormat.channelCount, 2)
      XCTAssertEqual(file.fileFormat.commonFormat, .pcmFormatFloat32)
      XCTAssertEqual(file.length, 5)
      let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 16))
      try file.read(into: buffer)
      XCTAssertEqual(buffer.frameLength, 5)
      let channels = try XCTUnwrap(buffer.floatChannelData)
      for index in left.indices {
        XCTAssertEqual(channels[0][index], left[index], accuracy: 1e-7)
        XCTAssertEqual(channels[1][index], right[index], accuracy: 1e-7)
      }
      let original = try Data(contentsOf: url)
      XCTAssertThrowsError(try PipelineProbe.writeWAV(waveform, to: url))
      XCTAssertEqual(try Data(contentsOf: url), original)
    }
  }
}
