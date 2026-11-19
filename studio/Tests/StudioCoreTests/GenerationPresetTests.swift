import Foundation
import XCTest
@testable import StudioCore

final class GenerationPresetTests: XCTestCase {
  func testNestedGuidancePreservesOmittedFieldsAndCapturesExplicitAutomatic() throws {
    var selection = GenerationSelection(); selection.ltx25Guidance = .init(mode: .guided, experimentalEnabled: true)
    selection.ltx25Guidance?.audioCFG = 7; selection.ltx25Guidance?.sigmas = [1, 0.5, 0]
    let partial: [String: GenerationPresetValue] = ["ltx25Guidance": .object(["mode": .string("guided"), "experimentalEnabled": .bool(true)])]
    let preserved = try GenerationPresetFields.applying(partial, to: selection, allowed: ["ltx25Guidance"])
    XCTAssertEqual(preserved.ltx25Guidance?.audioCFG, 7); XCTAssertEqual(preserved.ltx25Guidance?.sigmas, [1, 0.5, 0])
    selection.ltx25Guidance?.audioCFG = nil; selection.ltx25Guidance?.sigmas = nil
    let captured = try GenerationPresetFields.capture(selection, keys: ["ltx25Guidance"])
    guard case .object(let fields) = captured["ltx25Guidance"] else { return XCTFail("Missing guidance") }
    XCTAssertEqual(fields["audioCFG"], .automatic); XCTAssertEqual(fields["sigmas"], .automatic)
    for sigmas: GenerationPresetValue in [.automatic, .array([]), .array([.number(1), .number(0)])] {
      let snapshot = GenerationPresetSnapshot(context: .init(workspace: .clip, task: "t2v"), fields: ["ltx25Guidance": .object(["sigmas": sigmas])])
      let restored = try JSONDecoder().decode(GenerationPresetSnapshot.self, from: JSONEncoder().encode(snapshot))
      let result = try GenerationPresetFields.applying(restored.fields, to: preserved, allowed: ["ltx25Guidance"])
      XCTAssertEqual(result.ltx25Guidance?.audioCFG, 7)
      XCTAssertEqual(result.ltx25Guidance?.sigmas, sigmas == .automatic ? nil : sigmas == .array([]) ? [] : [1, 0])
    }
    XCTAssertThrowsError(try GenerationPresetFields.applying(["ltx25Guidance": .object(["unknownFutureControl": .number(1)])], to: selection, allowed: ["ltx25Guidance"]))
  }
  func testAddedDisabledAdapterReviewIncludesItsStateAndOwnedSettings() {
    var before = snapshot(); before.adapters = []
    var after = before; var adapter = GenerationPresetAdapter(reference: .init(id: "flex", backend: "native", family: "h3"), enabled: false)
    adapter.fields["h3Settings"] = .object(["profile": .string("pika_flex")]); after.adapters = [adapter]
    let changes = GenerationPresetDiff.changes(from: before, to: after)
    XCTAssertTrue(changes.contains { $0.path == "adapters.flex" && $0.current.contains("Off") })
    XCTAssertTrue(changes.contains { $0.path == "adapters.flex.h3Settings.profile" })
  }
  func testFinishingModelChangeIsOneSemanticModelRow() throws {
    let before = snapshot()
    var after = before; after.model?.id = "other"
    let a = GenerationPresetSnapshot(context: .init(workspace: .finishing, task: "finishing"), fields: ["stages": .object(["0": try .encoded(before)])])
    let b = GenerationPresetSnapshot(context: a.context, fields: ["stages": .object(["0": try .encoded(after)])])
    XCTAssertEqual(GenerationPresetDiff.changes(from: a, to: b).map(\.path), ["fields.stages.0.model"])
  }
  func testAdapterIdentityChangeIsVisibleEvenWithoutAContentHash() {
    var saved = snapshot(); saved.adapters = [.init(reference: .init(id: "same", backend: "native", family: "ltx23"))]
    var current = saved; current.adapters?[0].reference.family = "ltx25"
    XCTAssertTrue(GenerationPresetDiff.changes(from: saved, to: current).contains { $0.path == "adapters.same.identity" })
  }
  private func snapshot() -> GenerationPresetSnapshot {
    .init(context: .init(workspace: .image, task: "image"), model: .init(id: "moss", backend: "nativeMOSS", family: "moss_image", variant: "sft", precision: "bf16"), fields: ["steps": .integer(6), "guidance": .number(1)])
  }
  func testCreateNeverOverwritesAndNamesAreNormalized() throws {
    var library = GenerationPresetLibrary()
    let first = try library.create(name: "  My fast recipe  ", snapshot: snapshot())
    XCTAssertEqual(first.name, "My fast recipe")
    XCTAssertThrowsError(try library.create(name: "MY FAST RECIPE", snapshot: snapshot()))
    XCTAssertThrowsError(try library.create(name: " \n ", snapshot: snapshot()))
    let copy = try library.create(name: "Copy", snapshot: snapshot())
    XCTAssertNotEqual(first.id, copy.id); XCTAssertEqual(library.presets.count, 2)
  }
  func testReviewedRevisionIsRequiredAndSuppliedOverridesSurviveNewDefaults() throws {
    var supplied = StudioGenerationPreset(id: "supplied.moss.fast", name: "MOSS Fast", origin: .supplied, snapshot: snapshot())
    var library = GenerationPresetLibrary(defaults: [supplied])
    var changed = snapshot(); changed.fields["steps"] = .integer(8)
    let updated = try library.update(id: supplied.id, expectedRevision: supplied.revision, snapshot: changed)
    XCTAssertEqual(updated.revision, 2)
    XCTAssertThrowsError(try library.update(id: supplied.id, expectedRevision: 1, snapshot: snapshot()))
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("presets.json"); try library.write(to: file)
    supplied.snapshot.fields["steps"] = .integer(4)
    let restored = try GenerationPresetLibrary.read(from: file, defaults: [supplied])
    XCTAssertEqual(restored.presets.first?.snapshot, changed)
  }
  func testDiffDistinguishesMissingAutomaticAndEmptyAndIncludesNestedChanges() {
    let saved = snapshot(); var current = saved
    current.fields["sampler"] = .automatic; current.fields["stack"] = .array([])
    current.fields["advanced"] = .object(["enabled": .bool(true), "layers": .array([.integer(22)])])
    let changes = GenerationPresetDiff.changes(from: saved, to: current)
    XCTAssertEqual(Set(changes.map(\.path)), ["fields.sampler", "fields.stack", "fields.advanced.enabled", "fields.advanced.layers"])
    XCTAssertTrue(changes.allSatisfy { $0.saved != $0.current })
  }
  func testLoRAOrderStrengthAndToggleAreSemanticButRowIDsAreAbsent() {
    var saved = snapshot()
    saved.adapters = [.init(reference: .init(id: "a", backend: "native", family: "moss_image"), strength: 1, enabled: true),
      .init(reference: .init(id: "b", backend: "native", family: "moss_image"), strength: 0.5, enabled: false)]
    var current = saved; current.adapters!.reverse(); current.adapters![0].enabled = true; current.adapters![1].strength = 0.6
    let changes = GenerationPresetDiff.changes(from: saved, to: current)
    XCTAssertTrue(changes.contains { $0.path == "adapters.order" })
    XCTAssertTrue(changes.contains { $0.path == "adapters.a.strength" })
    XCTAssertTrue(changes.contains { $0.path == "adapters.b.enabled" })
  }
  func testInvalidAndOversizedSnapshotsAndUnknownVersionsAreRejectedTransactionally() throws {
    var library = GenerationPresetLibrary(); let good = try library.create(name: "Good", snapshot: snapshot())
    var bad = snapshot(); bad.fields["steps"] = .number(.infinity)
    XCTAssertThrowsError(try library.update(id: good.id, expectedRevision: 1, snapshot: bad))
    bad = snapshot(); bad.fields["huge"] = .string(String(repeating: "x", count: 65537))
    XCTAssertThrowsError(try library.create(name: "Huge", snapshot: bad))
    XCTAssertEqual(library.presets, [good])
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let file = root.appendingPathComponent("presets.json")
    try Data("{\"version\":999,\"records\":[],\"migrations\":[]}".utf8).write(to: file)
    XCTAssertThrowsError(try GenerationPresetLibrary.read(from: file))
  }
  func testFailedWritePreservesPreviousFile() throws {
    var library = GenerationPresetLibrary(); _ = try library.create(name: "Good", snapshot: snapshot())
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("presets.json"); try library.write(to: file)
    let original = try Data(contentsOf: file)
    XCTAssertThrowsError(try library.write(to: file.appendingPathComponent("impossible.json")))
    XCTAssertEqual(try Data(contentsOf: file), original)
  }
  func testClearingNestedOverridesShowsAutomaticInsteadOfMissing() {
    var saved = snapshot(); saved.fields["guidanceRecipe"] = .object(["cfg": .number(3)])
    var current = saved; current.fields["guidanceRecipe"] = .automatic
    let changes = GenerationPresetDiff.changes(from: saved, to: current)
    XCTAssertEqual(changes.count, 1)
    XCTAssertEqual(changes.first?.path, "fields.guidanceRecipe")
    XCTAssertEqual(changes.first?.current, "Automatic")
  }
  func testSmallExplicitNumericalChangesAreNeverHiddenByDisplayRounding() {
    var saved = snapshot(); saved.fields["shift"] = .number(1.000000001)
    var current = saved; current.fields["shift"] = .number(1.000000002)
    XCTAssertEqual(GenerationPresetDiff.changes(from: saved, to: current).map(\.path), ["fields.shift"])
  }
  func testOlderAdapterWithOmittedOptionalOwnedFieldsRetainsDefaultSemantics() throws {
    let adapter = GenerationPresetAdapter(reference: .init(id: "a", backend: "native", family: "h3"))
    var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(adapter)) as! [String: Any]
    object.removeValue(forKey: "fields")
    let restored = try JSONDecoder().decode(GenerationPresetAdapter.self, from: JSONSerialization.data(withJSONObject: object))
    XCTAssertEqual(restored.fields, [:])
  }
}
