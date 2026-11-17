import Foundation

/// The released checkpoint pairs have distinct learned LQ encoders and decoder strides.
public enum NativeRelayVSRScale: Int, Codable, CaseIterable, Sendable {
  case x2 = 2, x4 = 4
  public var label: String { "\(rawValue)×" }
  public var checkpointNames: [String] {
    self == .x2 ? ["relayvsr_x2_lightx2v_int8_convrot.safetensors", "relayvsr_flash_vae_x2_fp32.safetensors"]
      : ["relayvsr_x4_int8_convrot.safetensors", "relayvsr_flash_vae_x4_fp32.safetensors"]
  }
  public var wanFormat: String { self == .x2 ? "relayvsr-wan-x2-v1" : "refvsr-wan-v1" }
  public var flashFormat: String { self == .x2 ? "relayvsr-flash-x2-v1" : "refvsr-flash-v1" }
  public var paddingMultiple: Int { self == .x2 ? 16 : 8 }
  public var wanGridDivisor: Int { self == .x2 ? 16 : 8 }
  public var flashGridDivisor: Int { self == .x2 ? 8 : 4 }
  public var wanUnshuffle: Int { self == .x2 ? 8 : 16 }
  public var flashThirdStride: Int { self == .x2 ? 2 : 1 }

  public func resolveDirectory(_ root: URL) throws -> URL {
    var candidates = [root, root.appendingPathComponent("x\(rawValue)")]
    if ["x2", "x4"].contains(root.lastPathComponent) {
      candidates.append(root.deletingLastPathComponent().appendingPathComponent("x\(rawValue)"))
    }
    guard let folder = candidates.first(where: { candidate in
      checkpointNames.allSatisfy { FileManager.default.isReadableFile(atPath: candidate.appendingPathComponent($0).path) }
    }) else { throw ContractError.invalid("Choose a folder containing the matching RelayVSR \(label) checkpoint pair.") }
    return folder
  }
  public func validateMetadata(wan: [String: String], flash: [String: String]) throws {
    func configuration(_ metadata: [String: String]) throws -> [String: Any] {
      guard let bytes = metadata["config"]?.data(using: .utf8), bytes.count <= 65536,
        let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
        throw ContractError.invalid("RelayVSR checkpoint configuration is missing or invalid.")
      }
      return object
    }
    let w = try configuration(wan), f = try configuration(flash)
    guard wan["format"] == wanFormat, flash["format"] == flashFormat,
      (w["upscale_factor"] as? Int ?? 4) == rawValue,
      f["upscale_factor"] as? Int == rawValue else {
      throw ContractError.invalid("The RelayVSR checkpoint pair does not match the selected \(label) model.")
    }
  }
}
