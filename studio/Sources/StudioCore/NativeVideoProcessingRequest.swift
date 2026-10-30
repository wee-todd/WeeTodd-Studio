import Foundation

/// Frontend-neutral standalone processing. Model paths remain separate from
/// the reusable settings, exactly as in exported native generation jobs.
public struct NativeVideoProcessingRequest: Codable, Sendable {
  public var format = "weetodd-video-processing-request-v2"
  public var sourcePath: String
  public var sourceSHA256: String?
  public var recipe: VideoProcessingRecipe
  public var runtime: FlashVSRRuntime?
  public var nativeRuntime: NativePixelCQRuntime?
  public var h3Runtime: H3RegenerateRuntime?
  public init(sourcePath: String, recipe: VideoProcessingRecipe, runtime: FlashVSRRuntime? = nil,
    nativeRuntime: NativePixelCQRuntime? = nil, h3Runtime: H3RegenerateRuntime? = nil) {
    self.sourcePath = sourcePath; self.recipe = recipe; self.runtime = runtime
    self.nativeRuntime = nativeRuntime; self.h3Runtime = h3Runtime
  }
  private enum CodingKeys: String, CodingKey, CaseIterable { case format, sourcePath, sourceSHA256, recipe, runtime, nativeRuntime, h3Runtime }
  public init(from decoder: Decoder) throws {
    try FlashVSRStrictObject.check(decoder, allowed: Set(CodingKeys.allCases.map(\.rawValue)))
    let c = try decoder.container(keyedBy: CodingKeys.self)
    format = try c.decode(String.self, forKey: .format); sourcePath = try c.decode(String.self, forKey: .sourcePath)
    sourceSHA256 = try c.decodeIfPresent(String.self, forKey: .sourceSHA256)
    recipe = try c.decode(VideoProcessingRecipe.self, forKey: .recipe)
    runtime = try c.decodeIfPresent(FlashVSRRuntime.self, forKey: .runtime)
    nativeRuntime = try c.decodeIfPresent(NativePixelCQRuntime.self, forKey: .nativeRuntime)
    h3Runtime = try c.decodeIfPresent(H3RegenerateRuntime.self, forKey: .h3Runtime)
    try validate()
  }
  public func validate() throws {
    try recipe.validate()
    guard format == "weetodd-video-processing-request-v2", sourcePath.hasPrefix("/"), sourcePath.utf8.count <= 4096,
      !sourcePath.unicodeScalars.contains(where: { $0.value < 32 }),
      sourceSHA256 == nil || sourceSHA256!.utf8.count == 64 && sourceSHA256!.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      recipe.hasFlashVSR == (runtime != nil), recipe.hasNative == (nativeRuntime != nil), recipe.hasH3Regenerate == (h3Runtime != nil) else {
      throw StudioError.invalid("Choose a local source and exactly the runtimes used by this processing recipe.")
    }
  }
}
