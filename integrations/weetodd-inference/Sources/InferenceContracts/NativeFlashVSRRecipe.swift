import Foundation

/// Frozen native worker input. RGB transport is lossless and has exactly the
/// source clock; model-grid expansion is an internal index map only.
public struct NativeFlashVSRRecipe: Codable, Equatable, Sendable {
  public enum Variant: String, Codable, Sendable { case turbo, base }
  public enum Decoder: String, Codable, Sendable { case tiny, wan }
  public enum ColorFix: String, Codable, Sendable { case wavelet, adain, off }
  public let version: Int
  public let engine: String
  public let sourceID: String
  public let inputRGBPath: String
  public let inputSHA256: String
  public let outputRGBPath: String
  public let scratchDirectory: String
  public let modelRoot: String
  public let width: Int
  public let height: Int
  public let frames: Int
  public let fpsNumerator: Int
  public let fpsDenominator: Int
  public let temporalCompression: Int
  public let variant: Variant
  public let decoder: Decoder
  public let seed: UInt32
  public let colorFix: ColorFix
  public let waveletLevels: Int
  public let temporalSigma: Double
  public let maximumActiveBytes: UInt64
  public let sampling: NativeSamplingContract
  public let leasePath: String?
  public let ownerPID: Int32?

  public var workingWidth: Int { (width + 127) / 128 * 128 }
  public var workingHeight: Int { (height + 127) / 128 * 128 }
  public var repeatFactor: Int { 4 / temporalCompression }
  public var effectiveFrames: Int { ((frames - 1 + temporalCompression - 1) / temporalCompression) * temporalCompression + 1 }
  public var expandedFrames: Int { 1 + (effectiveFrames - 1) * repeatFactor }
  /// Four latents produce thirteen initial frames, then sixteen per group.
  public var groupCount: Int { (expandedFrames + 18) / 16 }
  public var inputBytes: UInt64 { UInt64(workingWidth * workingHeight * frames * 3) }
  public var outputBytes: UInt64 { UInt64(width * height * frames * 3) }
  public var capability: NativeCapabilityDescriptor {
    NativeCapabilityDescriptor.flashProfiles.first { $0.id == "flashvsr-\(variant.rawValue)-\(decoder.rawValue)" }!
  }

  public init(sourceID: String, inputRGBPath: String, inputSHA256: String, outputRGBPath: String,
    scratchDirectory: String, modelRoot: String, width: Int, height: Int, frames: Int,
    fpsNumerator: Int, fpsDenominator: Int, temporalCompression: Int, variant: Variant,
    decoder: Decoder, seed: UInt32, colorFix: ColorFix = .wavelet, waveletLevels: Int = 5,
    temporalSigma: Double = 1, maximumActiveBytes: UInt64 = 64 * 1024 * 1024 * 1024,
    leasePath: String? = nil, ownerPID: Int32? = nil) throws {
    version = 1; engine = "flashvsr"; self.sourceID = sourceID; self.inputRGBPath = inputRGBPath; self.inputSHA256 = inputSHA256
    self.outputRGBPath = outputRGBPath; self.scratchDirectory = scratchDirectory; self.modelRoot = modelRoot
    self.width = width; self.height = height; self.frames = frames
    self.fpsNumerator = fpsNumerator; self.fpsDenominator = fpsDenominator
    self.temporalCompression = temporalCompression; self.variant = variant; self.decoder = decoder; self.seed = seed
    self.colorFix = colorFix; self.waveletLevels = waveletLevels; self.temporalSigma = temporalSigma
    self.maximumActiveBytes = maximumActiveBytes; sampling = NativeCapabilityDescriptor.flashTurboTiny.sampling
    self.leasePath = leasePath; self.ownerPID = ownerPID
    try validate()
  }

  public func sourceIndex(expandedIndex: Int) -> Int {
    let index = max(0, min(expandedFrames - 1, expandedIndex))
    return min(frames - 1, index == 0 ? 0 : 1 + (index - 1) / repeatFactor)
  }
  public func projectionExpandedIndices(group: Int) -> [Int] {
    let start = group == 0 ? 0 : 1 + group * 16
    let count = group == 0 ? 17 : 16
    return (start..<(start + count)).map { min(expandedFrames - 1, $0) }
  }
  public func validate() throws {
    func digest(_ value: String) -> Bool {
      value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
    func path(_ value: String) -> Bool {
      value.hasPrefix("/") && value.utf8.count <= 4096 && !value.unicodeScalars.contains { $0.value < 32 }
    }
    guard version == 1, engine == "flashvsr", digest(sourceID), digest(inputSHA256),
      [inputRGBPath, outputRGBPath, scratchDirectory, modelRoot].allSatisfy(path),
      leasePath == nil || path(leasePath!), ownerPID == nil || ownerPID! > 1,
      [width, height].allSatisfy({ (64...8192).contains($0) && $0 % 2 == 0 }),
      (1...100000).contains(frames), [1, 2, 4].contains(temporalCompression),
      (1...1000000).contains(fpsNumerator), (1...1000000).contains(fpsDenominator),
      (1...120).contains(Double(fpsNumerator) / Double(fpsDenominator)),
      (1...8).contains(waveletLevels), temporalSigma.isFinite, temporalSigma > 0,
      (UInt64(1024 * 1024 * 1024)...UInt64(256 * 1024 * 1024 * 1024)).contains(maximumActiveBytes),
      sampling == NativeCapabilityDescriptor.flashTurboTiny.sampling else {
      throw ContractError.invalid("Invalid native FlashVSR source, model grid, schedule or memory admission.")
    }
    let input = URL(fileURLWithPath: inputRGBPath).standardizedFileURL.path
    let output = URL(fileURLWithPath: outputRGBPath).standardizedFileURL.path
    let scratch = URL(fileURLWithPath: scratchDirectory).standardizedFileURL.path
    let models = URL(fileURLWithPath: modelRoot).standardizedFileURL.path
    guard input != output, !input.hasPrefix(scratch + "/"), !output.hasPrefix(scratch + "/"),
      scratch != input, scratch != output, models != scratch,
      !models.hasPrefix(scratch + "/"), !scratch.hasPrefix(models + "/") else {
      throw ContractError.invalid("Native FlashVSR source/output files must be separate from its disposable scratch directory.")
    }
    try sampling.validate()
  }
  public static func prepare(data: Data) throws -> Self {
    let keys: Set<String> = ["version", "engine", "sourceID", "inputRGBPath", "inputSHA256", "outputRGBPath",
      "scratchDirectory", "modelRoot", "width", "height", "frames", "fpsNumerator", "fpsDenominator",
      "temporalCompression", "variant", "decoder", "seed", "colorFix", "waveletLevels", "temporalSigma",
      "maximumActiveBytes", "sampling", "leasePath", "ownerPID"]
    guard data.count <= 1024 * 1024, let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(object.keys).isSubset(of: keys) else { throw ContractError.invalid("Invalid or unknown native FlashVSR recipe fields.") }
    let recipe = try JSONDecoder().decode(Self.self, from: data)
    try recipe.validate()
    return recipe
  }
}
