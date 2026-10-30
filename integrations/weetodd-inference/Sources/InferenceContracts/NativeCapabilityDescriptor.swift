import Foundation

public struct NativeCapabilityDescriptor: Codable, Equatable, Sendable {
  public enum Qualification: String, Codable, Sendable { case importedUnqualified, experimental, qualified }
  public let version: Int
  public let id: String
  public let family: NativeModelFamily
  public let requiredRoles: [NativeComponentRole]
  public let sampling: NativeSamplingContract
  public let qualification: Qualification

  private init(id: String, family: NativeModelFamily, roles: [NativeComponentRole],
    sampling: NativeSamplingContract, qualification: Qualification) {
    version = 1; self.id = id; self.family = family; requiredRoles = roles
    self.sampling = sampling; self.qualification = qualification
  }
  public static let h3Regenerate = Self(id: "h3-regenerate-publisher-3", family: .h3,
    roles: [.transformer, .refiner, .sharpCurve, .sharpInput, .latentUpscaler, .videoVAE, .audioVAE],
    sampling: .init(sigmas: [0.909375, 6.0 / 7, 2.0 / 3, 0], audioSigmas: [0, 0, 0, 0],
      prediction: .dataWardVelocity, timeBasis: .oneMinusSigma, updateConvention: .addDataWardVelocity, audioClock: .fixedZero),
    qualification: .importedUnqualified)
  public static let flashProfiles: [Self] = ["turbo", "base"].flatMap { variant in
    ["tiny", "wan"].map { decoder in Self(id: "flashvsr-\(variant)-\(decoder)", family: .flashVSR,
      roles: [.transformer, .projection, .fixedContext, .decoder],
      sampling: .init(sigmas: [1, 0], audioSigmas: [], prediction: .flowVelocity,
        timeBasis: .sigmaTimes1000, updateConvention: .subtractVelocity, audioClock: .sourcePassthrough),
      qualification: .importedUnqualified) }
  }
  public static var flashTurboTiny: Self { flashProfiles[0] }
  public static func vdn(evaluations: Int) throws -> Self {
    guard [8, 50].contains(evaluations) else { throw ContractError.invalid("VDN requires eight or fifty evaluations.") }
    func grid(_ shift: Double) -> [Double] {
      (0...evaluations).map { i in let x = Double(evaluations - i) / Double(evaluations); return shift * x / (1 + (shift - 1) * x) }
    }
    return Self(id: "h3-vdn-\(evaluations)", family: .h3,
      roles: [.transformer, .linearBranch, .standardAdapter] + (evaluations == 8 ? [.turboAdapter] : []),
      sampling: .init(sigmas: grid(12), audioSigmas: grid(3), prediction: .dataWardVelocity,
        timeBasis: .oneMinusSigma, updateConvention: .addDataWardVelocity, audioClock: .separateSigma), qualification: .experimental)
  }
  private func validateDescriptor() throws {
    try sampling.validate()
    guard version == 1, !id.isEmpty, id.utf8.count <= 128, !requiredRoles.isEmpty,
      Set(requiredRoles).count == requiredRoles.count else { throw ContractError.invalid("Invalid native capability descriptor.") }
    let admitted = [Self.h3Regenerate] + Self.flashProfiles + [try Self.vdn(evaluations: 8), try Self.vdn(evaluations: 50)]
    guard admitted.contains(self) else { throw ContractError.invalid("Unrecognized native capability or modified released profile.") }
  }
  public func validate(components: [NativeComponentIdentity], sampling requested: NativeSamplingContract) throws {
    try validateDescriptor(); try requested.validate()
    guard requested == sampling, components.count == requiredRoles.count,
      Set(components.map(\.role)) == Set(requiredRoles), components.allSatisfy({ $0.family == family }) else {
      throw ContractError.invalid("Selected components or sampling schedule do not match this native profile.")
    }
    for component in components {
      try component.validate()
      if id == Self.h3Regenerate.id {
        if [.sharpInput, .sharpCurve].contains(component.role),
          component.timestepBasis != nil && component.timestepBasis != .sharp1025x8 {
          throw ContractError.invalid("Sharp requires its own 1025 by 8 timestep coordinates.")
        }
        if component.role == .refiner {
          guard component.qkvLayout == nil || component.qkvLayout == "split-diffusers",
            component.swigluOrder == nil || component.swigluOrder == "gate-up",
            component.scaleMode == nil || component.scaleMode == "folded-unit" else {
            throw ContractError.invalid("H3 regeneration adapter ordering or scaling differs from the selected profile.")
          }
        }
      }
    }
  }
  private enum CodingKeys: String, CodingKey, CaseIterable { case version, id, family, requiredRoles, sampling, qualification }
  public init(from decoder: Decoder) throws {
    try NativeContractObject.check(decoder, keys: CodingKeys.allCases.map(\.rawValue))
    let c = try decoder.container(keyedBy: CodingKeys.self)
    version = try c.decode(Int.self, forKey: .version); id = try c.decode(String.self, forKey: .id)
    family = try c.decode(NativeModelFamily.self, forKey: .family); requiredRoles = try c.decode([NativeComponentRole].self, forKey: .requiredRoles)
    sampling = try c.decode(NativeSamplingContract.self, forKey: .sampling); qualification = try c.decode(Qualification.self, forKey: .qualification)
    try validateDescriptor()
  }
}
