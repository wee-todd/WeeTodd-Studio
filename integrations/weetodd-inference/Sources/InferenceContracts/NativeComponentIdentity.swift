import CryptoKit
import Foundation

public enum NativeModelFamily: String, Codable, Sendable { case h3, flashVSR, ltx25 }
public enum NativeComponentRole: String, Codable, CaseIterable, Sendable {
  case transformer, projection, fixedContext, decoder, videoVAE, audioVAE
  case refiner, sharpCurve, sharpInput, latentUpscaler, linearBranch, standardAdapter, turboAdapter
}
public enum NativeTimestepBasis: String, Codable, Sendable {
  case sharp1025x8, native1001x64, original1025x2688
}

/// Content and interpretation are both part of an asset's identity. Paths are
/// deliberately absent: moving a verified file does not change its model math.
public struct NativeComponentIdentity: Codable, Equatable, Sendable {
  public let version: Int
  public let family: NativeModelFamily
  public let role: NativeComponentRole
  public let sourceSHA256: String
  public let layout: String
  public let importerVersion: String?
  public let timestepBasis: NativeTimestepBasis?
  public let qkvLayout: String?
  public let swigluOrder: String?
  public let scaleMode: String?

  public init(family: NativeModelFamily, role: NativeComponentRole, sourceSHA256: String,
    layout: String, importerVersion: String? = nil, timestepBasis: NativeTimestepBasis? = nil,
    qkvLayout: String? = nil, swigluOrder: String? = nil, scaleMode: String? = nil) {
    version = 1; self.family = family; self.role = role; self.sourceSHA256 = sourceSHA256
    self.layout = layout; self.importerVersion = importerVersion; self.timestepBasis = timestepBasis
    self.qkvLayout = qkvLayout; self.swigluOrder = swigluOrder; self.scaleMode = scaleMode
  }
  public func validate() throws {
    guard version == 1, sourceSHA256.utf8.count == 64,
      sourceSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      [layout].allSatisfy({ !$0.isEmpty && $0.utf8.count <= 128 }),
      [importerVersion, qkvLayout, swigluOrder, scaleMode].compactMap({ $0 })
        .allSatisfy({ !$0.isEmpty && $0.utf8.count <= 128 }) else {
      throw ContractError.invalid("Invalid native component content or layout identity.")
    }
  }
  public func fingerprint() throws -> String {
    try validate()
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    return SHA256.hash(data: try encoder.encode(self)).map { String(format: "%02x", $0) }.joined()
  }
  private enum CodingKeys: String, CodingKey, CaseIterable {
    case version, family, role, sourceSHA256, layout, importerVersion, timestepBasis, qkvLayout, swigluOrder, scaleMode
  }
  public init(from decoder: Decoder) throws {
    try NativeContractObject.check(decoder, keys: CodingKeys.allCases.map(\.rawValue))
    let c = try decoder.container(keyedBy: CodingKeys.self)
    version = try c.decode(Int.self, forKey: .version); family = try c.decode(NativeModelFamily.self, forKey: .family)
    role = try c.decode(NativeComponentRole.self, forKey: .role); sourceSHA256 = try c.decode(String.self, forKey: .sourceSHA256)
    layout = try c.decode(String.self, forKey: .layout); importerVersion = try c.decodeIfPresent(String.self, forKey: .importerVersion)
    timestepBasis = try c.decodeIfPresent(NativeTimestepBasis.self, forKey: .timestepBasis)
    qkvLayout = try c.decodeIfPresent(String.self, forKey: .qkvLayout); swigluOrder = try c.decodeIfPresent(String.self, forKey: .swigluOrder)
    scaleMode = try c.decodeIfPresent(String.self, forKey: .scaleMode)
    try validate()
  }
}

enum NativeContractObject {
  private struct Key: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
  }
  static func check(_ decoder: Decoder, keys: [String]) throws {
    let container = try decoder.container(keyedBy: Key.self)
    guard Set(container.allKeys.map(\.stringValue)).isSubset(of: Set(keys)) else {
      throw ContractError.invalid("Unknown native capability field.")
    }
  }
}
