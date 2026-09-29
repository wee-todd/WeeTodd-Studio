import CryptoKit
import Foundation
import StudioCore

/// Studio's side of the versioned worker request. The worker rehashes the
/// recipe before model load, so a changed prepared job cannot be rendered.
enum NativeVideoJobRequest {
  static func ltx(payload: [String: Any], runtime: RuntimeSettings, output: URL) throws -> [String: Any] {
    try make(payload: payload, runtime: runtime, output: output, engine: "ltx25", label: "LTX")
  }
  static func h3(payload: [String: Any], runtime: RuntimeSettings, output: URL) throws -> [String: Any] {
    try make(payload: payload, runtime: runtime, output: output, engine: "h3", label: "H3")
  }
  private static func make(payload: [String: Any], runtime: RuntimeSettings, output: URL,
    engine: String, label: String) throws -> [String: Any] {
    guard Set(payload.keys) == ["recipePath"], let path = payload["recipePath"] as? String,
      path.hasPrefix("/"), output.isFileURL else {
      throw StudioError.invalid("A prepared \(label) recipe and local output directory are required.")
    }
    let attributes = try FileManager.default.attributesOfItem(atPath: path)
    guard attributes[.type] as? FileAttributeType == .typeRegular,
      let size = attributes[.size] as? NSNumber, size.intValue <= 1024 * 1024 else {
      throw StudioError.invalid("The native \(label) recipe must be a regular file of at most 1 MiB.")
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    guard data.count <= 1024 * 1024 else {
      throw StudioError.invalid("The native \(label) recipe exceeds 1 MiB.")
    }
    var envelope: [String: Any] = ["version": 1, "jobID": UUID().uuidString, "engine": engine,
      "recipePath": path, "recipeSHA256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
      "outputDirectory": output.path]
    if !runtime.ffmpegPath.isEmpty { envelope["ffmpegPath"] = runtime.ffmpegPath }
    return envelope
  }
}
