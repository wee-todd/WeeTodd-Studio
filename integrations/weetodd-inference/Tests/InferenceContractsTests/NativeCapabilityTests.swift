import Foundation
import XCTest
@testable import InferenceContracts

final class NativeCapabilityTests: XCTestCase {
  func testPublishedRegenerateScheduleAndStrictRoundTrip() throws {
    let descriptor = NativeCapabilityDescriptor.h3Regenerate
    XCTAssertEqual(descriptor.sampling.sigmas, [0.909375, 6.0 / 7, 2.0 / 3, 0])
    XCTAssertEqual(descriptor.sampling.evaluations, 3)
    XCTAssertEqual(descriptor.sampling.audioClock, .fixedZero)
    XCTAssertEqual(descriptor.sampling.timeBasis, .oneMinusSigma)
    XCTAssertEqual(try JSONDecoder().decode(NativeCapabilityDescriptor.self,
      from: JSONEncoder().encode(descriptor)), descriptor)
    var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(descriptor)) as! [String: Any]
    object["version"] = 2
    XCTAssertThrowsError(try JSONDecoder().decode(NativeCapabilityDescriptor.self,
      from: JSONSerialization.data(withJSONObject: object)))
    object["version"] = 1; object["ignoredSetting"] = true
    XCTAssertThrowsError(try JSONDecoder().decode(NativeCapabilityDescriptor.self,
      from: JSONSerialization.data(withJSONObject: object)))
  }

  func testProfilesRejectWrongSchedulesComponentsAndTimestepCoordinates() throws {
    let profile = NativeCapabilityDescriptor.h3Regenerate
    var components = profile.requiredRoles.map { NativeComponentIdentity(family: .h3,
      role: $0, sourceSHA256: String(repeating: "a", count: 64), layout: "synthetic-v1") }
    XCTAssertNoThrow(try profile.validate(components: components, sampling: profile.sampling))
    XCTAssertThrowsError(try profile.validate(components: Array(components.dropLast()), sampling: profile.sampling))
    components[0] = NativeComponentIdentity(family: .flashVSR, role: components[0].role,
      sourceSHA256: String(repeating: "a", count: 64), layout: "synthetic-v1")
    XCTAssertThrowsError(try profile.validate(components: components, sampling: profile.sampling))
    components[0] = NativeComponentIdentity(family: .h3, role: components[0].role,
      sourceSHA256: String(repeating: "a", count: 64), layout: "synthetic-v1",
      timestepBasis: .native1001x64)
    XCTAssertNoThrow(try profile.validate(components: components, sampling: profile.sampling))
    let sharpIndex = components.firstIndex(where: { $0.role == .sharpInput })!
    components[sharpIndex] = NativeComponentIdentity(family: .h3, role: .sharpInput,
      sourceSHA256: String(repeating: "a", count: 64), layout: "synthetic-v1", timestepBasis: .native1001x64)
    XCTAssertThrowsError(try profile.validate(components: components, sampling: profile.sampling))
    XCTAssertThrowsError(try profile.validate(components: [], sampling: NativeCapabilityDescriptor.flashTurboTiny.sampling))
  }

  func testComponentIdentityBindsConversionAndAllFlashSelectionsAreExplicit() throws {
    let source = String(repeating: "b", count: 64)
    let original = NativeComponentIdentity(family: .flashVSR, role: .decoder,
      sourceSHA256: source, layout: "tiny-784-v1", importerVersion: "torch-descriptors-v1")
    let changed = NativeComponentIdentity(family: .flashVSR, role: .decoder,
      sourceSHA256: source, layout: "wan-16-v1", importerVersion: "torch-descriptors-v1")
    XCTAssertNotEqual(try original.fingerprint(), try changed.fingerprint())
    XCTAssertEqual(Set(NativeCapabilityDescriptor.flashProfiles.map(\.id)).count, 4)
    XCTAssertEqual(NativeCapabilityDescriptor.flashTurboTiny.sampling.sigmas, [1, 0])
    XCTAssertEqual(NativeCapabilityDescriptor.flashTurboTiny.sampling.updateConvention, .subtractVelocity)
    XCTAssertThrowsError(try NativeComponentIdentity(family: .h3, role: .decoder,
      sourceSHA256: "wrong", layout: "x").validate())
  }
}
