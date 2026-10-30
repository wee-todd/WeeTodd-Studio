import Foundation
import InferenceContracts

/// User controls for the fixed H3 restoration profile. Model preparation and
/// inference resolve into the same immutable worker job from every frontend.
public struct H3RegenerateSettings: Codable, Equatable, Sendable {
  public enum SourceMode: String, Codable, CaseIterable, Sendable { case movie, nativeLatentsWhenAvailable }
  public var version = 1
  public var targetWidth = 2560, targetHeight = 1440
  public var decoder: H3RegenerateJob.Decoder = .releasedH3
  public var encoderProfile: H3RegenerateJob.EncoderProfile = .publisherPosteriorFP32
  public var schedule: H3RegenerateJob.Schedule = .publisherThreeStep
  public var sourceMode: SourceMode = .nativeLatentsWhenAvailable
  public var fit: FlashVSRSettings.Fit = .fit
  public var seed: UInt64 = 0
  public var posteriorSeed: UInt64? = 0
  public var latentUpscaleMode: H3RegenerateJob.UpscaleMode = .directPublisher
  public var baseProfilePath = "", modelLibraryRoot = ""
  public var publisherEncoderPath = "", fastDecoderPath = ""
  public var prompt = "Preserve the source subjects, identities, composition, lighting, camera and motion while refining fine detail and removing compression artifacts."
  public init() {}
  private enum CodingKeys: String, CodingKey, CaseIterable {
    case version, targetWidth, targetHeight, decoder, encoderProfile, schedule, sourceMode, fit, seed,
      posteriorSeed, latentUpscaleMode, baseProfilePath, modelLibraryRoot, publisherEncoderPath, fastDecoderPath, prompt
  }
  public init(from decoder: Decoder) throws {
    try FlashVSRStrictObject.check(decoder, allowed: Set(CodingKeys.allCases.map(\.rawValue)))
    let c = try decoder.container(keyedBy: CodingKeys.self)
    version = try c.decode(Int.self, forKey: .version)
    targetWidth = try c.decode(Int.self, forKey: .targetWidth); targetHeight = try c.decode(Int.self, forKey: .targetHeight)
    self.decoder = try c.decode(H3RegenerateJob.Decoder.self, forKey: .decoder)
    encoderProfile = try c.decode(H3RegenerateJob.EncoderProfile.self, forKey: .encoderProfile)
    schedule = try c.decode(H3RegenerateJob.Schedule.self, forKey: .schedule)
    sourceMode = try c.decode(SourceMode.self, forKey: .sourceMode); fit = try c.decode(FlashVSRSettings.Fit.self, forKey: .fit)
    seed = try c.decode(UInt64.self, forKey: .seed); posteriorSeed = try c.decodeIfPresent(UInt64.self, forKey: .posteriorSeed)
    latentUpscaleMode = try c.decode(H3RegenerateJob.UpscaleMode.self, forKey: .latentUpscaleMode)
    baseProfilePath = try c.decode(String.self, forKey: .baseProfilePath); modelLibraryRoot = try c.decode(String.self, forKey: .modelLibraryRoot)
    publisherEncoderPath = try c.decode(String.self, forKey: .publisherEncoderPath); fastDecoderPath = try c.decode(String.self, forKey: .fastDecoderPath)
    prompt = try c.decode(String.self, forKey: .prompt)
    try validate()
  }
  public func validate() throws {
    guard version == 1, [targetWidth, targetHeight].allSatisfy({ (32...4096).contains($0) && $0 % 2 == 0 }),
      !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, prompt.utf8.count <= 100000,
      encoderProfile != .nativeNormalizedLatents,
      (encoderProfile == .publisherPosteriorFP32) == (posteriorSeed != nil),
      [baseProfilePath, modelLibraryRoot, publisherEncoderPath, fastDecoderPath].allSatisfy({
        $0.isEmpty || $0.hasPrefix("/") && $0.utf8.count <= 4096 && !$0.unicodeScalars.contains(where: { $0.value < 32 })
      }) else {
      throw StudioError.invalid("Choose valid H3 restoration dimensions, source precision, prompt and local model paths.")
    }
  }
  public func validateSourceClock(numerator: Int, denominator: Int) throws {
    guard numerator > 0, denominator > 0, numerator <= 1000000, denominator <= 1000000,
      numerator == denominator * 24 else {
      throw StudioError.invalid("H3 Regenerate preserves 24 FPS sources. Choose FlashVSR+ for this source rate, or explicitly prepare a separate 24 FPS copy before selecting H3. No frame-rate conversion was applied.")
    }
  }
}
