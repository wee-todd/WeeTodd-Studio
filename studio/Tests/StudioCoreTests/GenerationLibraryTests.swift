import Foundation
import XCTest
@testable import StudioCore

final class GenerationLibraryTests: XCTestCase {
  func testFastAdapterExplainsItsRequiredRecipeBeforeItIsApplied() throws {
    let entry = GenerationLoRAEntry(reference: .init(id: "fast", backend: "native", family: "moss_image"), name: "Fast",
      compatibleFamilies: ["moss_image"], availability: .installed,
      binding: .nativeFile(path: "/models/fast.safetensors", profile: "moss_sft_4step_v2"))
    let requirements = try XCTUnwrap(entry.recipeRequirements)
    for value in ["SFT", "strength 1", "CFG 1", "shift 5", "Euler", "release", "4–8"] { XCTAssertTrue(requirements.contains(value), value) }
  }
  func testPagedWeightFilesParticipateInComponentIdentity() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString), page = root.appendingPathComponent("pages/block-0.safetensors")
    try FileManager.default.createDirectory(at: page.deletingLastPathComponent(), withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try Data([0]).write(to: page)
    let files = GenerationLibraryFileIdentity.componentFiles(root)
    XCTAssertTrue(files.contains { $0.hasSuffix("/pages/block-0.safetensors") && FileManager.default.contentsEqual(atPath: $0, andPath: page.path) }, "Expected \(page.path), got \(files)")
  }
  func testInPlaceSameSizeEditWithRestoredModificationDateChangesIdentity() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("weights.safetensors")
    try Data([1, 2, 3, 4]).write(to: file)
    let modified = try XCTUnwrap(FileManager.default.attributesOfItem(atPath: file.path)[.modificationDate] as? Date)
    let before = GenerationLibraryFileIdentity.signature([file.path])
    let handle = try FileHandle(forWritingTo: file); try handle.write(contentsOf: Data([4, 3, 2, 1])); try handle.close()
    try FileManager.default.setAttributes([.modificationDate: modified], ofItemAtPath: file.path)
    XCTAssertNotEqual(GenerationLibraryFileIdentity.signature([file.path]), before)
  }
  private func moss(_ identity: String = "release-1") -> GenerationModelEntry {
    .init(reference: .init(id: "moss-sft", backend: "nativeMOSS", family: "moss_image",
      variant: "sft", precision: "bf16", sourceIdentity: identity), name: "MOSS Image SFT",
      workspaces: [.image], tasks: ["image"], inputRoleSets: [[]], supportsLoRAs: true,
      availability: .installed, binding: .nativeMOSS(transformer: "/library/SFT/transformer", shared: "/library/Shared", library: "/library"))
  }
  private var image: GenerationLibraryContext { .init(workspace: .image, task: "image") }

  func testModelAdmissionUsesExactInputRolesAndDoesNotClaimAnotherTask() throws {
    let entry = moss(), library = GenerationLibrarySnapshot(models: [entry])
    XCTAssertEqual(library.models(for: image).map(\.reference.id), ["moss-sft"])
    XCTAssertTrue(library.models(for: .init(workspace: .image, task: "image", inputRoles: ["canvas"])).isEmpty)
    XCTAssertThrowsError(try library.resolve(entry.reference, in: .init(workspace: .clip, task: "i2v")))
  }

  func testChangedCheckpointOrPrecisionCannotResolveAnOlderSavedReference() throws {
    let old = moss(), changed = moss("release-2")
    let library = GenerationLibrarySnapshot(models: [changed])
    XCTAssertThrowsError(try library.resolve(old.reference, in: image))
    var wrongPrecision = changed.reference; wrongPrecision.precision = "q8"
    XCTAssertThrowsError(try library.resolve(wrongPrecision, in: image))
    XCTAssertEqual(try library.resolve(changed.reference, in: image), changed.binding)
  }

  func testDisconnectedModelRemainsVisibleButCannotBeResolvedForExecution() throws {
    var entry = moss(); entry.availability = .driveDisconnected
    let library = GenerationLibrarySnapshot(models: [entry])
    XCTAssertEqual(library.models(for: image).map(\.name), ["MOSS Image SFT"])
    XCTAssertThrowsError(try library.resolve(entry.reference, in: image))
    XCTAssertEqual(library.modelEntries.first?.availability, .driveDisconnected)
  }

  func testLoRAsAreFilteredByFamilyVariantAndActualModelSupport() throws {
    let entry = moss()
    let fast = GenerationLoRAEntry(reference: .init(id: "moss-fast", backend: "native", family: "moss_image", sourceIdentity: "fast-v2"),
      name: "MOSS SFT Fast", compatibleFamilies: ["moss_image"], compatibleVariants: ["sft"],
      availability: .installed, binding: .nativeFile(path: "/library/LoRAs/fast.safetensors", profile: "moss_sft_4step_v2"))
    var qwen = entry; qwen.reference.id = "qwen"; qwen.reference.family = "qwen_image"; qwen.supportsLoRAs = false
    var base = entry; base.reference.id = "moss-base"; base.reference.variant = "base"
    let library = GenerationLibrarySnapshot(models: [entry, qwen, base], loras: [fast])
    XCTAssertEqual(library.loras(for: image, model: entry.reference).map(\.reference.id), ["moss-fast"])
    XCTAssertTrue(library.loras(for: image, model: qwen.reference).isEmpty)
    XCTAssertTrue(library.loras(for: image, model: base.reference).isEmpty)
    XCTAssertThrowsError(try library.resolve(fast.reference, for: base.reference, in: image))
  }

  func testIndexKeepsLogicalIdentityWhenAnExistingAliasIsRefreshed() throws {
    var index = GenerationLibraryIndex()
    let first = try index.registerModel(moss(), alias: "native:profile-1")
    var refreshed = moss(); refreshed.reference.id = "new-generated-id"; refreshed.name = "My MOSS"
    let second = try index.registerModel(refreshed, alias: "native:profile-1")
    XCTAssertEqual(first.id, second.id)
    XCTAssertEqual(index.snapshot.modelEntries.count, 1)
    XCTAssertEqual(index.snapshot.modelEntries.first?.name, "My MOSS")
  }

  func testIndexRoundTripRetainsOfflineBindingAndRejectsUnsupportedVersion() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let file = directory.appendingPathComponent("generation-library.json")
    var index = GenerationLibraryIndex(); var entry = moss(); entry.availability = .driveDisconnected
    _ = try index.registerModel(entry, alias: "native:profile-1")
    try index.write(to: file)
    let reopened = try GenerationLibraryIndex.read(from: file)
    XCTAssertEqual(reopened.snapshot.modelEntries.first?.binding, entry.binding)
    XCTAssertEqual(reopened.snapshot.modelEntries.first?.availability, .driveDisconnected)
    var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
    object["version"] = 99
    try JSONSerialization.data(withJSONObject: object).write(to: file)
    XCTAssertThrowsError(try GenerationLibraryIndex.read(from: file))
  }

  func testRejectedRegistrationCannotPoisonThePreviouslyValidIndex() throws {
    var index = GenerationLibraryIndex()
    _ = try index.registerModel(moss(), alias: "native:profile-1")
    let original = index
    var invalid = moss(); invalid.name = ""
    XCTAssertThrowsError(try index.registerModel(invalid, alias: "native:profile-1"))
    XCTAssertEqual(index, original)
  }

  func testAnUnrelatedSourceCannotTakeOverAnExistingLogicalID() throws {
    var index = GenerationLibraryIndex()
    _ = try index.registerModel(moss(), alias: "native:profile-1")
    let original = index
    var other = moss(); other.reference.family = "qwen_image"; other.reference.backend = "nativeMLX"
    XCTAssertThrowsError(try index.registerModel(other, alias: "another-source"))
    XCTAssertEqual(index, original)
  }
}
