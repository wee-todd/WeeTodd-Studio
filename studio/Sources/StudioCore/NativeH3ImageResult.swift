import CoreFoundation
import Foundation
import ImageIO

public enum NativeH3ImageResult {
  public static func validateIdentity(result: [String: Any], fingerprint: String?, requestID: UUID,
    seed: UInt32, width: Int, height: Int) throws {
    guard result["nativeRuntime"] as? String == "swift-mlx",
      (result["jobID"] as? String).flatMap(UUID.init(uuidString:)) != nil,
      result["requestID"] as? String == requestID.uuidString,
      let actualFingerprint = result["fingerprint"] as? String, actualFingerprint.hasPrefix("sha256:"),
      actualFingerprint.utf8.count == 71, fingerprint == nil || actualFingerprint == fingerprint,
      let normalized = result["normalizedRequest"] as? [String: Any],
      let config = normalized["configuration"] as? [String: Any],
      config["width"] as? Int == width, config["height"] as? Int == height,
      let actualSeed = config["seed"] as? NSNumber, CFGetTypeID(actualSeed) != CFBooleanGetTypeID(),
      actualSeed.doubleValue == Double(seed) else {
      throw StudioError.invalid("The H3 image completion does not match its captured request.")
    }
  }
  public static func image(result: [String: Any], output: URL, fingerprint: String,
    requestID: UUID, seed: UInt32, width: Int, height: Int) throws -> URL {
    try validateIdentity(result: result, fingerprint: fingerprint, requestID: requestID,
      seed: seed, width: width, height: height)
    let expected = output.appendingPathComponent("image.png").standardizedFileURL.resolvingSymlinksInPath()
    guard let asset = result["asset"] as? [String: Any], let path = asset["path"] as? String,
      asset["width"] as? Int == width, asset["height"] as? Int == height,
      URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath() == expected,
      expected.deletingLastPathComponent() == output.standardizedFileURL.resolvingSymlinksInPath(),
      let size = try FileManager.default.attributesOfItem(atPath: expected.path)[.size] as? NSNumber,
      (1...16 * 1024 * 1024).contains(size.intValue),
      let source = CGImageSourceCreateWithURL(expected as CFURL, nil), CGImageSourceGetCount(source) == 1,
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil), image.width == width, image.height == height else {
      throw StudioError.invalid("The H3 worker did not publish a complete PNG at the requested canvas inside its output.")
    }
    return expected
  }
}
