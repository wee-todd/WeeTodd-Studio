import Darwin
import Foundation
import InferenceContracts

public struct FlashVSRFile: Codable, Equatable, Sendable {
  public let path: String, sha256: String
  public let bytes: Int64
  public init(path: String, sha256: String, bytes: Int64) {
    self.path = path; self.sha256 = sha256; self.bytes = bytes
  }
  public static func capture(_ url: URL) throws -> Self {
    let canonical = url.resolvingSymlinksInPath()
    let identity = try NativeMovieFrozenMedia.Identity(canonical)
    return Self(path: canonical.path, sha256: try NativeMovieFrozenMedia.digest(canonical), bytes: identity.bytes)
  }
  public func verify() throws {
    guard self == (try Self.capture(URL(fileURLWithPath: path))) else {
      throw StudioError.invalid("A frozen FlashVSR+ input changed: " + URL(fileURLWithPath: path).lastPathComponent)
    }
  }
}
public struct FlashVSRComponents: Codable, Equatable, Sendable {
  public let root: String
  public let dit: FlashVSRFile, projection: FlashVSRFile, context: FlashVSRFile, decoder: FlashVSRFile
  public var files: [FlashVSRFile] { [dit, projection, context, decoder] }
}
public enum FlashVSRModelManifest {
  public static func verifyLookup(_ components: FlashVSRComponents, settings: FlashVSRSettings) throws {
    let urls = try lookupURLs(root: URL(fileURLWithPath: components.root), settings: settings)
    guard zip(urls, components.files).allSatisfy({ $0.resolvingSymlinksInPath().path == $1.path }) else {
      throw StudioError.invalid("FlashVSR+ component lookup changed. Check the model folder again before running.")
    }
    for file in components.files { try file.verify() }
  }
  public static func resolve(root: URL, settings: FlashVSRSettings) throws -> FlashVSRComponents {
    let files = try lookupURLs(root: root, settings: settings).map(FlashVSRFile.capture)
    return FlashVSRComponents(root: root.path, dit: files[0], projection: files[1], context: files[2], decoder: files[3])
  }
  private static func lookupURLs(root: URL, settings: FlashVSRSettings) throws -> [URL] {
    try settings.validate()
    return try NativeFlashVSRAssets.resolve(root: root,
      variant: settings.variant == .turbo ? .turbo : .base,
      decoder: settings.decoder == .tiny ? .tiny : .wan,
      policy: settings.backend == .swiftMLX ? .native : .legacyVpipe).files
  }
  public static func verifyPublished(_ files: FlashVSRComponents, settings: FlashVSRSettings) throws {
    let hashes = [
      settings.variant == .turbo ? "7862b92178176fa3aebc34c6c071daca6b56ab1dddb8e7ff9a01c47f5226ec8b" : "58d64c1f1b3a559bafb69bb014f3ab2745e800c89435afdc154966eb96bc7459",
      settings.variant == .turbo ? "e77b378ae12342eefbb58f924256f362a151b93ce73dd32808fcdaca9c051496" : "a1fe997f19060171db8dd4d58c30debc78ddfcc357b63481798792a38a356d8d",
      contextSHA256,
      settings.decoder == .tiny ? "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161" : "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
    ]
    guard zip(files.files, hashes).allSatisfy({ $0.sha256 == $1 }) else {
      throw StudioError.invalid("FlashVSR+ component identity does not match the selected official variant and decoder.")
    }
  }
  // Official fixed context; source revision and acquisition are documented in settings help.
  public static let contextSHA256 = "4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef"
}
