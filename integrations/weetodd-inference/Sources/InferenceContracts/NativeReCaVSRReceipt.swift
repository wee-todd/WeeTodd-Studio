import Foundation

public struct NativeReCaVSRReceipt: Codable, Sendable {
  public let format: String
  public let recipe: NativeReCaVSRRecipe
  public let fingerprint: String
  public let video: ReCaFileIdentity
  public let frames: Int, width: Int, height: Int
  public let sourceIn: Double, duration: Double, seconds: Double
  public let sourceTimingPreserved: Bool
  public let audioPolicy: String
  public let fidelity: String
  public let peakMLXBytes: UInt64
  public let activeAfterStagesBytes: UInt64
  public init(recipe: NativeReCaVSRRecipe, video: ReCaFileIdentity, frames: Int, width: Int, height: Int,
    sourceIn: Double, duration: Double, seconds: Double, peakMLXBytes: UInt64 = 0,
    activeAfterStagesBytes: UInt64 = 0) throws {
    self.format = "weetodd-recavsr-result-v1"; self.recipe = recipe; fingerprint = try recipe.fingerprint()
    self.video = video; self.frames = frames; self.width = width; self.height = height
    self.sourceIn = sourceIn; self.duration = duration; self.seconds = seconds
    sourceTimingPreserved = true; audioPolicy = "source-passthrough"
    fidelity = "experimental-native; cross-platform-image-parity-unverified"
    self.peakMLXBytes = peakMLXBytes; self.activeAfterStagesBytes = activeAfterStagesBytes
  }
  public func validate() throws {
    try recipe.validate(); try video.validate()
    guard format == "weetodd-recavsr-result-v1", fingerprint == (try recipe.fingerprint()),
      video.path == recipe.outputPath, frames > 0, (1...16384).contains(width), (1...16384).contains(height),
      sourceIn.isFinite, sourceIn >= 0, duration.isFinite, duration > 0, seconds.isFinite, seconds >= 0,
      sourceTimingPreserved, audioPolicy == "source-passthrough" else {
      throw ContractError.invalid("The ReCaVSR result does not match its prepared movie job.")
    }
  }
  public func validate(expected: NativeReCaVSRRecipe, output: String) throws {
    try validate()
    guard fingerprint == (try expected.fingerprint()),
      URL(fileURLWithPath: video.path).standardizedFileURL.resolvingSymlinksInPath().path ==
        URL(fileURLWithPath: output).standardizedFileURL.resolvingSymlinksInPath().path else {
      throw ContractError.invalid("The ReCaVSR receipt belongs to another submitted job or output.")
    }
  }
}
