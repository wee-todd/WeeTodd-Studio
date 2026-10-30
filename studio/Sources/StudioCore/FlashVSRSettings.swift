import Foundation

/// Backend-independent controls. Resolved jobs retain an explicit engine identity.
public struct FlashVSRSettings: Codable, Equatable, Sendable {
  public enum Backend: String, Codable, CaseIterable, Sendable { case vpipe, swiftMLX }
  public enum Variant: String, Codable, CaseIterable, Sendable { case turbo, base }
  public enum Decoder: String, Codable, CaseIterable, Sendable { case tiny, vae
    public var label: String { self == .tiny ? "Tiny · faster" : "Full Wan VAE" }
  }
  public enum Fit: String, Codable, CaseIterable, Sendable { case fit, fill }
  public enum ColorFix: String, Codable, CaseIterable, Sendable { case wavelet, adain, off }
  public enum Continuity: String, Codable, CaseIterable, Sendable { case stream, clips }
  public var version = 1
  public var backend: Backend = .swiftMLX
  public var variant: Variant = .turbo
  public var decoder: Decoder = .tiny
  public var width = 1280, height = 720
  public var fit: Fit = .fit
  public var seed: UInt32 = 0
  public var temporalCompression = 4
  public var temporalSigma = 1.0
  public var colorFix: ColorFix = .wavelet
  public var waveletLevels = 5
  public var continuity: Continuity = .stream
  public var preserveAudio = true
  public init() {}
  private enum CodingKeys: String, CodingKey, CaseIterable { case version, backend, variant, decoder, width, height, fit, seed, temporalCompression, temporalSigma, colorFix, waveletLevels, continuity, preserveAudio }
  public init(from decoder: Swift.Decoder) throws {
    // Older saved requests omitted this field when vpipe was the only engine.
    backend = .vpipe
    try FlashVSRStrictObject.check(decoder, allowed: Set(CodingKeys.allCases.map(\.rawValue)))
    let values = try decoder.container(keyedBy: CodingKeys.self)
    if values.contains(.version) { self.version = try values.decode(Int.self, forKey: .version) }
    if values.contains(.backend) { self.backend = try values.decode(Backend.self, forKey: .backend) }
    if values.contains(.variant) { self.variant = try values.decode(Variant.self, forKey: .variant) }
    if values.contains(.decoder) { self.decoder = try values.decode(Decoder.self, forKey: .decoder) }
    if values.contains(.width) { self.width = try values.decode(Int.self, forKey: .width) }
    if values.contains(.height) { self.height = try values.decode(Int.self, forKey: .height) }
    if values.contains(.fit) { self.fit = try values.decode(Fit.self, forKey: .fit) }
    if values.contains(.seed) { self.seed = try values.decode(UInt32.self, forKey: .seed) }
    if values.contains(.temporalCompression) { self.temporalCompression = try values.decode(Int.self, forKey: .temporalCompression) }
    if values.contains(.temporalSigma) { self.temporalSigma = try values.decode(Double.self, forKey: .temporalSigma) }
    if values.contains(.colorFix) { self.colorFix = try values.decode(ColorFix.self, forKey: .colorFix) }
    if values.contains(.waveletLevels) { self.waveletLevels = try values.decode(Int.self, forKey: .waveletLevels) }
    if values.contains(.continuity) { self.continuity = try values.decode(Continuity.self, forKey: .continuity) }
    if values.contains(.preserveAudio) { self.preserveAudio = try values.decode(Bool.self, forKey: .preserveAudio) }
  }

  public func validate() throws {
    guard version == 1, [width, height].allSatisfy({ (64...8192).contains($0) && $0 % 2 == 0 }),
      [1, 2, 4].contains(temporalCompression), temporalSigma.isFinite, temporalSigma > 0,
      (1...8).contains(waveletLevels) else {
      throw StudioError.invalid("FlashVSR+ needs even dimensions from 64 to 8192, compression 1, 2 or 4, and valid color/motion settings.")
    }
  }
  public func validateExecution() throws {
    try validate()
    if backend == .swiftMLX {
      guard continuity == .stream else { throw StudioError.invalid("Native FlashVSR+ uses continuous stream state. Select Stream before rendering.") }
    } else {
      guard variant == .turbo else { throw StudioError.invalid("The optional vpipe backend executes the qualified Turbo profile.") }
    }
  }
  public func canvas() throws -> FlashVSRCanvas {
    try validate()
    return FlashVSRCanvas(width: ((width + 127) / 128) * 128, height: ((height + 127) / 128) * 128,
      visibleWidth: width, visibleHeight: height)
  }
}
public struct FlashVSRCanvas: Codable, Equatable, Sendable {
  public let width: Int, height: Int, visibleWidth: Int, visibleHeight: Int
}

enum FlashVSRStrictObject {
  struct Key: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
  }
  static func check(_ decoder: Swift.Decoder, allowed: Set<String>) throws {
    let values = try decoder.container(keyedBy: Key.self)
    let unknown = values.allKeys.map(\.stringValue).filter { !allowed.contains($0) }
    guard unknown.isEmpty else { throw StudioError.invalid("Unsupported FlashVSR+ setting: " + unknown.sorted().joined(separator: ", ")) }
  }
}
