import CryptoKit
import CoreFoundation
import Foundation
import ImageIO

public enum NativeMOSSImageResult {
  public static func validateRegionalControl(_ result: [String:Any], required: Bool) throws {
    guard required else { return }
    let request = result["normalizedRequest"] as? [String:Any]
    guard request?["sourceImage"] == nil, (request?["configuration"] as? [String:Any])?["regionalLayout"] as? Bool == true,
      result["conditioningMode"] as? String == "regional-attention-v2",
      let contexts = result["regionalContexts"] as? [[String:Any]], !contexts.isEmpty, contexts.count <= 62 else {
      throw StudioError.invalid("This MOSS worker does not admit regional subject-box control. Use the current bundled native worker.")
    }
  }
  public static func validateCleanup(_ result: [String: Any]) throws {
    let request = result["normalizedRequest"] as? [String: Any]
    let source = request?["sourceImage"] as? [String: Any]
    if (request?["configuration"] as? [String:Any])?["regionalLayout"] as? Bool == true {
      try validateRegionalControl(result, required: true)
    }
    let expected = source == nil ? ["conditioning", "transformer", "decoding"] : ["encoding", "conditioning", "transformer", "decoding"]
    guard result["releasedStages"] as? [String] == expected,
      result["generationMode"] as? String != "vae-roundtrip-control" else {
      throw StudioError.invalid("MOSS did not finish its generation and staged cleanup contract.")
    }
    if let source {
      let bound = (result["sourceReconstruction"] as? [String: Any])?["source"] as? [String: Any]
      guard result["generationMode"] as? String == "source-reconstruction",
        let evaluations = result["modelEvaluations"] as? Int, evaluations > 0,
        bound?["path"] as? String == source["path"] as? String,
        bound?["sha256"] as? String == source["sha256"] as? String,
        let strength = source["noiseLevel"] as? NSNumber, let actual = bound?["noiseLevel"] as? NSNumber,
        strength.doubleValue.isFinite, strength.doubleValue > 0, strength.doubleValue < 1,
        strength.floatValue == actual.floatValue else {
        throw StudioError.invalid("MOSS returned incomplete source reconstruction provenance.")
      }
    }
  }
  private static func integer(_ value: Any?, equals expected: Int) -> Bool {
    guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite else { return false }
    return number.doubleValue == Double(expected)
  }
  public static func validateIdentity(_ result: [String: Any], requestID: UUID, fingerprint: String?,
    seed: UInt32, width: Int, height: Int) throws {
    guard result["schema"] as? String == "weetodd-moss-image-result-v1", result["eligibility"] as? String == "allowed",
      result["requestID"] as? String == requestID.uuidString, integer(result["width"], equals: width),
      integer(result["height"], equals: height), integer(result["seed"], equals: Int(seed)),
      let actual = result["fingerprint"] as? String, !actual.isEmpty, fingerprint == nil || fingerprint == actual else {
      throw StudioError.invalid("MOSS result does not match the captured generation request.")
    }
  }
  public static func image(_ result: [String: Any], output: URL, requestID: UUID, fingerprint: String,
    seed: UInt32, width: Int, height: Int) throws -> URL {
    try validateIdentity(result, requestID: requestID, fingerprint: fingerprint, seed: seed, width: width, height: height)
    guard let request = result["normalizedRequest"] as? [String: Any], request["requestID"] as? String == requestID.uuidString,
      let configuration = request["configuration"] as? [String: Any], integer(configuration["seed"], equals: Int(seed)),
      integer(configuration["width"], equals: width), integer(configuration["height"], equals: height),
      let effectivePrompt = result["effectivePrompt"] as? String, !effectivePrompt.isEmpty else {
      throw StudioError.invalid("MOSS returned incomplete generation provenance.")
    }
    let expected = output.appendingPathComponent("image.png").standardizedFileURL
    guard let path = result["imagePath"] as? String, URL(fileURLWithPath: path).standardizedFileURL == expected,
      let sha = result["imageSHA256"] as? String, sha.count == 64,
      let source = CGImageSourceCreateWithURL(expected as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any],
      properties[kCGImagePropertyPixelWidth as String] as? Int == width,
      properties[kCGImagePropertyPixelHeight as String] as? Int == height else { throw StudioError.invalid("MOSS returned an invalid PNG or output path.") }
    let file = try FileHandle(forReadingFrom: expected); defer { try? file.close() }
    var digest = SHA256()
    while let bytes = try file.read(upToCount: 1024 * 1024), !bytes.isEmpty { digest.update(data: bytes) }
    guard digest.finalize().map({ String(format: "%02x", $0) }).joined() == sha else { throw StudioError.invalid("MOSS image changed after rendering.") }
    try validateCleanup(result)
    return expected
  }
}
