import CryptoKit
import Foundation

public enum ReCaPrecision: String, Codable, Sendable { case bf16, int8ConvRot }
public enum ReCaColorFix: String, Codable, Sendable { case none, adain, wavelet }
public enum ReCaResidency: String, Codable, Sendable { case staged, resident }

public struct ReCaVSRSettings: Codable, Equatable, Sendable {
  public var scale: Double
  public var seed: UInt64
  public var precision: ReCaPrecision
  public var colorFix: ReCaColorFix
  public var residency: ReCaResidency
  public init(scale: Double = 4, seed: UInt64 = 42, precision: ReCaPrecision = .bf16,
    colorFix: ReCaColorFix = .none, residency: ReCaResidency = .staged) throws {
    self.scale = scale; self.seed = seed; self.precision = precision
    self.colorFix = colorFix; self.residency = residency
    try validate()
  }
  public func validate() throws {
    guard [1.5, 2, 4].contains(scale) else {
      throw ContractError.invalid("ReCaVSR supports scale 1.5, 2 or 4.")
    }
  }
  private enum CodingKeys: String, CodingKey, CaseIterable { case scale, seed, precision, colorFix, residency }
  public init(from decoder: Decoder) throws {
    try NativeContractObject.check(decoder, keys: CodingKeys.allCases.map(\.rawValue))
    let c = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(scale: c.decode(Double.self, forKey: .scale), seed: c.decode(UInt64.self, forKey: .seed),
      precision: c.decode(ReCaPrecision.self, forKey: .precision), colorFix: c.decode(ReCaColorFix.self, forKey: .colorFix),
      residency: c.decode(ReCaResidency.self, forKey: .residency))
  }
}

public struct ReCaSourceInterval: Codable, Equatable, Sendable {
  public let startSeconds: Double
  public let durationSeconds: Double
  public init(startSeconds: Double, durationSeconds: Double) throws {
    self.startSeconds = startSeconds; self.durationSeconds = durationSeconds
    try validate()
  }
  public func validate() throws {
    guard startSeconds.isFinite, startSeconds >= 0, durationSeconds.isFinite,
      durationSeconds > 0, (startSeconds + durationSeconds).isFinite else {
      throw ContractError.invalid("Select a finite nonnegative source start and positive duration.")
    }
  }
}

/// Paths locate immutable bytes; interpretation and fingerprints use their content identity.
public struct ReCaFileIdentity: Codable, Equatable, Sendable {
  public let path: String
  public let sha256: String
  public let bytes: UInt64
  public init(path: String, sha256: String, bytes: UInt64) {
    self.path = path; self.sha256 = sha256; self.bytes = bytes
  }
  public func validate() throws {
    _ = try NativeMediaSource(path: path, sha256: sha256)
    guard bytes > 0 else { throw ContractError.invalid("ReCaVSR inputs must be nonempty regular files.") }
  }
  public static func capture(_ url: URL) throws -> Self {
    guard url.isFileURL else { throw ContractError.invalid("ReCaVSR requires local files.") }
    let canonical = url.standardizedFileURL.resolvingSymlinksInPath()
    let values = try canonical.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
    guard values.isRegularFile == true, let size = values.fileSize, size > 0 else {
      throw ContractError.invalid("Cannot read the selected ReCaVSR input.")
    }
    let file = try FileHandle(forReadingFrom: canonical)
    defer { try? file.close() }
    var hash = SHA256()
    while let data = try file.read(upToCount: 4 * 1024 * 1024), !data.isEmpty {
      try Task.checkCancellation(); hash.update(data: data)
    }
    let result = Self(path: canonical.path,
      sha256: hash.finalize().map { String(format: "%02x", $0) }.joined(), bytes: UInt64(size))
    try result.verify()
    return result
  }
  public func verify() throws {
    try validate()
    let values = try URL(fileURLWithPath: path).resourceValues(forKeys: [.fileSizeKey])
    guard values.fileSize.map(UInt64.init) == bytes else {
      throw ContractError.invalid("A ReCaVSR input changed. Prepare it again.")
    }
    try NativeMediaSource(path: path, sha256: sha256).verify()
  }
}

public enum ReCaAssetRole: String, Codable, CaseIterable, Sendable {
  case transformer, prompt, flashDecoder, modelConfig, vaeConfig, flashDecoderConfig
}
public struct ReCaAsset: Codable, Equatable, Sendable {
  public let role: ReCaAssetRole
  public let file: ReCaFileIdentity
  public init(role: ReCaAssetRole, file: ReCaFileIdentity) { self.role = role; self.file = file }
}
public struct ReCaAssets: Codable, Equatable, Sendable {
  public let files: [ReCaAsset]
  public init(files: [ReCaAsset]) throws { self.files = files; try validate() }
  public func validate() throws {
    guard files.count == ReCaAssetRole.allCases.count,
      Set(files.map(\.role)) == Set(ReCaAssetRole.allCases) else {
      throw ContractError.invalid("ReCaVSR requires one transformer, prompt, FlashDecoder and three matching configurations.")
    }
    for file in files { try file.file.validate() }
  }
  public subscript(_ role: ReCaAssetRole) -> ReCaFileIdentity { files.first { $0.role == role }!.file }
  public func verify() throws { try validate(); for file in files { try file.file.verify() } }
}

public struct NativeReCaVSRRecipe: Codable, Equatable, Sendable {
  public let format: String
  public let source: ReCaFileIdentity
  public let interval: ReCaSourceInterval?
  public let worker: ReCaFileIdentity
  public let metal: ReCaFileIdentity
  public let assets: ReCaAssets
  public let settings: ReCaVSRSettings
  public var outputPath: String
  public let maximumActiveBytes: UInt64
  public var frameLimit: Int?
  public var resetFrames: [Int]
  public var leasePath: String?
  public var ownerPID: Int32?

  public init(source: ReCaFileIdentity, interval: ReCaSourceInterval?, worker: ReCaFileIdentity,
    metal: ReCaFileIdentity, assets: ReCaAssets, settings: ReCaVSRSettings,
    outputPath: String, maximumActiveBytes: UInt64 = 32 * 1024 * 1024 * 1024,
    frameLimit: Int? = nil, resetFrames: [Int] = [], leasePath: String? = nil, ownerPID: Int32? = nil) throws {
    format = "weetodd-recavsr-v1"; self.source = source; self.interval = interval
    self.worker = worker; self.metal = metal; self.assets = assets; self.settings = settings
    self.outputPath = outputPath; self.maximumActiveBytes = maximumActiveBytes
    self.frameLimit = frameLimit; self.resetFrames = resetFrames
    self.leasePath = leasePath; self.ownerPID = ownerPID
    try validate()
  }
  public func validate() throws {
    try source.validate(); try worker.validate(); try metal.validate()
    try assets.validate(); try settings.validate(); try interval?.validate()
    guard format == "weetodd-recavsr-v1", outputPath.hasPrefix("/"), !outputPath.utf8.contains(0),
      outputPath.utf8.count <= 4096, URL(fileURLWithPath: outputPath).pathExtension.lowercased() == "mov",
      maximumActiveBytes >= 1024 * 1024 * 1024,
      frameLimit.map({ $0 > 0 }) ?? true, resetFrames.allSatisfy({ $0 > 0 }),
      resetFrames == Array(Set(resetFrames)).sorted(),
      ownerPID.map({ $0 > 1 }) ?? true, leasePath.map({ $0.hasPrefix("/") && !$0.utf8.contains(0) }) ?? true else {
      throw ContractError.invalid("Invalid ReCaVSR recipe, memory budget, output or reset frames.")
    }
    let output = URL(fileURLWithPath: outputPath).standardizedFileURL.resolvingSymlinksInPath().path
    let inputs = [source, worker, metal] + assets.files.map(\.file)
    guard !inputs.contains(where: { URL(fileURLWithPath: $0.path).standardizedFileURL.resolvingSymlinksInPath().path == output }) else {
      throw ContractError.invalid("Choose a new output file separate from ReCaVSR inputs.")
    }
  }
  public func verifyInputs() throws {
    try validate(); try source.verify(); try worker.verify(); try metal.verify(); try assets.verify()
    guard !FileManager.default.fileExists(atPath: outputPath) else {
      throw ContractError.invalid("The ReCaVSR output already exists. Choose a new filename.")
    }
  }
  public func fingerprint() throws -> String {
    try validate()
    var copy = self
    copy.leasePath = nil; copy.ownerPID = nil
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    var object = try JSONSerialization.jsonObject(with: encoder.encode(copy)) as! [String: Any]
    object.removeValue(forKey: "outputPath")
    for key in ["source", "worker", "metal"] {
      var file = object[key] as! [String: Any]; file.removeValue(forKey: "path"); object[key] = file
    }
    object["assets"] = assets.files.sorted { $0.role.rawValue < $1.role.rawValue }.map {
      ["role": $0.role.rawValue, "sha256": $0.file.sha256, "bytes": $0.file.bytes] as [String: Any]
    }
    return SHA256.hash(data: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))
      .map { String(format: "%02x", $0) }.joined()
  }
  public static func prepare(_ data: Data) throws -> Self {
    guard data.count <= 1024 * 1024,
      let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(object.keys).isSubset(of: ["format", "source", "interval", "worker", "metal", "assets", "settings", "outputPath", "maximumActiveBytes", "frameLimit", "resetFrames", "leasePath", "ownerPID"]) else {
      throw ContractError.invalid("Invalid or unknown ReCaVSR recipe fields.")
    }
    let value = try JSONDecoder().decode(Self.self, from: data); try value.validate(); return value
  }
}
