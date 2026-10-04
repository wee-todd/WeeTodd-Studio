import Foundation
import MLX
import LTX25Engine

/// Explicit Dev-model policy. It never changes the qualified distilled recipe.
public struct MLXGuidedSampling: Codable, Sendable {
  public enum Mode: String, Codable, Sendable { case guided, guidedHQ = "guided_hq" }
  public let mode: Mode
  public let steps: Int
  public let negativePrompt: String
  public let videoCFG, audioCFG, stg, videoRescale, audioRescale, modality: Float
  public let stgBlocks: [Int]
  public let sigmas: [Double]?
  public let distilledAdapterPath: String
  enum CodingKeys: String, CodingKey, CaseIterable {
    case mode, steps, sigmas
    case negativePrompt = "negative_prompt", videoCFG = "video_cfg_scale", audioCFG = "audio_cfg_scale"
    case stg = "stg_scale", videoRescale = "video_rescale_scale", audioRescale = "audio_rescale_scale"
    case modality = "modality_scale", stgBlocks = "stg_blocks", distilledAdapterPath = "distilled_adapter_path"
  }
  private struct Key: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
  }
  public init(mode: Mode, steps: Int, negativePrompt: String = "", videoCFG: Float = 3,
    audioCFG: Float = 7, stg: Float, videoRescale: Float, audioRescale: Float,
    modality: Float = 3, stgBlocks: [Int], sigmas: [Double]? = nil,
    distilledAdapterPath: String) throws {
    guard (1...64).contains(steps), negativePrompt.utf8.count <= 16384,
      [videoCFG, audioCFG, stg, modality].allSatisfy({ $0.isFinite && (0...100).contains($0) }),
      [videoRescale, audioRescale].allSatisfy({ $0.isFinite && (0...1).contains($0) }),
      stgBlocks.allSatisfy({ (0..<48).contains($0) }), Set(stgBlocks).count == stgBlocks.count,
      stg == 0 || !stgBlocks.isEmpty,
      distilledAdapterPath.hasPrefix("/"), distilledAdapterPath.utf8.count <= 4096,
      !distilledAdapterPath.utf8.contains(0) else {
      throw LTXError.invalid("Dev guidance needs finite scales, valid STG blocks and an explicit distilled refinement adapter.")
    }
    if let sigmas {
      guard sigmas.count == steps + 1 else { throw LTXError.invalid("Guided sigma points must match the update count.") }
      _ = try SamplingSchedule(sigmas: sigmas, eta: 0)
      guard mode != .guidedHQ || sigmas[sigmas.count-2] > 0.0011 else {
        throw LTXError.invalid("HQ guided sampling needs its last positive sigma above the terminal 0.0011 denoise.")
      }
    }
    self.mode = mode; self.steps = steps; self.negativePrompt = negativePrompt
    self.videoCFG = videoCFG; self.audioCFG = audioCFG; self.stg = stg
    self.videoRescale = videoRescale; self.audioRescale = audioRescale; self.modality = modality
    self.stgBlocks = stgBlocks; self.sigmas = sigmas; self.distilledAdapterPath = distilledAdapterPath
  }
  public init(from decoder: Decoder) throws {
    let all = try decoder.container(keyedBy: Key.self)
    guard Set(all.allKeys.map(\.stringValue)).isSubset(of: Set(CodingKeys.allCases.map(\.rawValue))) else {
      throw LTXError.invalid("Unsupported Dev guidance field.")
    }
    let c = try decoder.container(keyedBy: CodingKeys.self)
    let mode = try c.decode(Mode.self, forKey: .mode), hq = mode == .guidedHQ
    try self.init(mode: mode, steps: c.decode(Int.self, forKey: .steps),
      negativePrompt: c.decodeIfPresent(String.self, forKey: .negativePrompt) ?? "",
      videoCFG: c.decodeIfPresent(Float.self, forKey: .videoCFG) ?? 3,
      audioCFG: c.decodeIfPresent(Float.self, forKey: .audioCFG) ?? 7,
      stg: c.decodeIfPresent(Float.self, forKey: .stg) ?? (hq ? 0 : 1),
      videoRescale: c.decodeIfPresent(Float.self, forKey: .videoRescale) ?? (hq ? 0.45 : 0.7),
      audioRescale: c.decodeIfPresent(Float.self, forKey: .audioRescale) ?? (hq ? 1 : 0.7),
      modality: c.decodeIfPresent(Float.self, forKey: .modality) ?? 3,
      stgBlocks: c.decodeIfPresent([Int].self, forKey: .stgBlocks) ?? (hq ? [] : [28]),
      sigmas: c.decodeIfPresent([Double].self, forKey: .sigmas),
      distilledAdapterPath: c.decode(String.self, forKey: .distilledAdapterPath))
  }
  public func schedule(videoTokens: Int) throws -> SamplingSchedule {
    try SamplingSchedule(sigmas: sigmas ?? Self.adaptiveSigmas(steps: steps, videoTokens: videoTokens), eta: 0)
  }
  public static func adaptiveSigmas(steps: Int, videoTokens: Int) throws -> [Double] {
    guard (1...64).contains(steps), (1...131072).contains(videoTokens) else {
      throw LTXError.invalid("Guided scheduling needs admitted steps and video tokens.")
    }
    let slope = (2.05-0.95)/Double(4096-1024), shift = Double(videoTokens)*slope + 0.95-slope*1024
    let exponential = exp(shift)
    let lastRatio=Double(steps-1)
    let lastComplement=lastRatio/(exponential+lastRatio)
    let values = (0..<steps).map { index -> Double in
      if index == 0 { return 1 }
      let ratio=Double(index)/Double(steps-index)
      return 1-0.9*(ratio/(exponential+ratio))/lastComplement
    }
    return values + [0]
  }
  /// Serial passes retain small output latents, never multiple weighted blocks.
  public static func reserveBytes(videoTokens: Int, audioTokens: Int) -> Int {
    (videoTokens+audioTokens)*128*4*12 + 1024*(4096+2048)*4*2
  }
}

enum MLXGuidanceMath {
  static func combine(_ conditional: MLXArray, negative: MLXArray?, perturbed: MLXArray?,
    isolated: MLXArray?, cfg: Float, stg: Float, modality: Float, rescale: Float) -> MLXArray {
    var result = conditional
    if cfg != 1 { result = result + (conditional-negative!)*(cfg-1) }
    if stg != 0 { result = result + (conditional-perturbed!)*stg }
    if modality != 1 { result = result + (conditional-isolated!)*(modality-1) }
    if rescale != 0 {
      let factor = sqrt(conditional.variance())/(sqrt(result.variance())+1e-8)
      result = result*(rescale*factor + (1-rescale))
    }
    return result
  }
  static func res2Coefficients(_ h: Double) -> (a21: Double, b1: Double, b2: Double) {
    func phi1(_ z: Double) -> Double { abs(z) < 1e-10 ? 1 : expm1(z)/z }
    func phi2(_ z: Double) -> Double { abs(z) < 1e-10 ? 0.5 : (expm1(z)-z)/(z*z) }
    let a = 0.5*phi1(-h*0.5), b = 2*phi2(-h)
    return (a, phi1(-h)-b, b)
  }
  static func normalizedNoise(_ value: MLXArray) -> MLXArray {
    let normalized = (value-value.mean())/(sqrt(value.variance())+1e-8)
    return (normalized-normalized.mean(axis: 0, keepDims: true)) /
      (sqrt(normalized.variance(axis: 0, keepDims: true))+1e-8)
  }
  static func res2Noise(sample: MLXArray, prediction: MLXArray, sigma: Double,
    nextSigma: Double, noise: MLXArray) -> MLXArray {
    guard nextSigma != 0 else { return prediction.asType(.bfloat16).asType(.float32) }
    let up = min(nextSigma*0.5, nextSigma*0.9999)
    let residual = max(0, nextSigma*nextSigma-up*up).squareRoot()
    let alpha = 1-nextSigma+residual, down = alpha > 0 ? residual/alpha : nextSigma
    // Derive scalar coefficients at schedule precision, then cast each complete
    // coefficient at the array operation. Premature Float subtraction can move
    // an SDE value across the subsequent BF16 rounding boundary.
    let epsilon = (sample-prediction)/Float(sigma-nextSigma)
    return (Float(alpha)*(sample-Float(sigma)*epsilon+Float(down)*epsilon)+Float(up)*noise)
      .asType(.bfloat16).asType(.float32)
  }
}
