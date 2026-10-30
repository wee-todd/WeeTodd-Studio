import Foundation

/// Describes exact released math; execution remains owned by each engine.
public struct NativeSamplingContract: Codable, Equatable, Sendable {
  public enum Prediction: String, Codable, Sendable { case dataWardVelocity, flowVelocity }
  public enum TimeBasis: String, Codable, Sendable { case oneMinusSigma, sigmaTimes1000 }
  public enum Update: String, Codable, Sendable { case addDataWardVelocity, subtractVelocity }
  public enum AudioClock: String, Codable, Sendable { case fixedZero, separateSigma, sourcePassthrough }
  public let version: Int
  public let sigmas: [Double]
  public let audioSigmas: [Double]
  public let prediction: Prediction
  public let timeBasis: TimeBasis
  public let updateConvention: Update
  public let audioClock: AudioClock
  public let guidance: Double
  public var evaluations: Int { sigmas.count - 1 }

  public init(sigmas: [Double], audioSigmas: [Double], prediction: Prediction,
    timeBasis: TimeBasis, updateConvention: Update, audioClock: AudioClock, guidance: Double = 1) {
    version = 1; self.sigmas = sigmas; self.audioSigmas = audioSigmas; self.prediction = prediction
    self.timeBasis = timeBasis; self.updateConvention = updateConvention; self.audioClock = audioClock; self.guidance = guidance
  }
  public func validate() throws {
    func decreasing(_ points: [Double]) -> Bool {
      (2...102).contains(points.count) && points.last == 0
        && points.allSatisfy({ $0.isFinite && (0...1).contains($0) })
        && zip(points, points.dropFirst()).allSatisfy({ $0 > $1 })
    }
    guard version == 1, decreasing(sigmas), guidance == 1 else {
      throw ContractError.invalid("Unsupported native sigma grid or guidance policy.")
    }
    switch audioClock {
    case .fixedZero:
      guard audioSigmas == Array(repeating: 0, count: sigmas.count) else { throw ContractError.invalid("Encoded silence must remain at audio sigma zero.") }
    case .separateSigma:
      guard audioSigmas.count == sigmas.count, decreasing(audioSigmas) else { throw ContractError.invalid("Invalid separate audio clock.") }
    case .sourcePassthrough:
      guard audioSigmas.isEmpty else { throw ContractError.invalid("Source audio passthrough has no denoising clock.") }
    }
    guard (prediction == .dataWardVelocity && timeBasis == .oneMinusSigma && updateConvention == .addDataWardVelocity)
      || (prediction == .flowVelocity && timeBasis == .sigmaTimes1000 && updateConvention == .subtractVelocity) else {
      throw ContractError.invalid("Prediction, timestep and update conventions are incompatible.")
    }
  }
  private enum CodingKeys: String, CodingKey, CaseIterable { case version, sigmas, audioSigmas, prediction, timeBasis, updateConvention, audioClock, guidance }
  public init(from decoder: Decoder) throws {
    try NativeContractObject.check(decoder, keys: CodingKeys.allCases.map(\.rawValue))
    let c = try decoder.container(keyedBy: CodingKeys.self)
    version = try c.decode(Int.self, forKey: .version); sigmas = try c.decode([Double].self, forKey: .sigmas)
    audioSigmas = try c.decode([Double].self, forKey: .audioSigmas); prediction = try c.decode(Prediction.self, forKey: .prediction)
    timeBasis = try c.decode(TimeBasis.self, forKey: .timeBasis); updateConvention = try c.decode(Update.self, forKey: .updateConvention)
    audioClock = try c.decode(AudioClock.self, forKey: .audioClock); guidance = try c.decode(Double.self, forKey: .guidance)
    try validate()
  }
}
