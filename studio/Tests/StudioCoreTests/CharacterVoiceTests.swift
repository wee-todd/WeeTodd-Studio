import XCTest

@testable import StudioCore

final class CharacterVoiceTests: XCTestCase {
  func testVoiceAssignmentLeavesVisualApprovalIntactAndRejectsConflictingImport() throws {
    var p = StudioProject()
    var plan = ProjectPlanning()
    var character = PlanningSubject(name: "Alex", kind: .character)
    character.details = "Brown hair"
    plan.subjects = [character]
    try plan.approveSubject(character.id)
    let revision = plan.subjects[0].revision
    let sample = VoiceReference(path: "/retained/sample.wav", duration: 1, transcript: "Hello")
    plan.subjects[0].voiceBinding = CharacterVoiceBinding(
      profile: VoiceProfileSnapshot(name: "Alex", samples: [sample]))
    XCTAssertTrue(plan.isSubjectApproved(character.id))
    XCTAssertEqual(plan.subjects[0].revision, revision)
    p.planning = plan
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let lib = try ProductionLibrary(url: directory.appendingPathComponent("library.sqlite"))
    let package = try lib.publish(rootID: character.id, planning: plan, assets: [])
    var imported = StudioProject()
    try imported.importLibraryPackage(package)
    XCTAssertEqual(imported.planning?.subjects[0].voiceBinding, plan.subjects[0].voiceBinding)
    var conflict = package
    conflict.subjects[0].voiceBinding = CharacterVoiceBinding(
      profile: VoiceProfileSnapshot(name: "Other", samples: [sample]))
    XCTAssertThrowsError(try imported.importLibraryPackage(conflict))
  }
  func testPortableCharacterRetainsAssignedVoiceSamples() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let audio = directory.appendingPathComponent("sample.wav")
    try Data([1, 2, 3]).write(to: audio)
    let storage = CharacterSheetDocumentStore(root: directory.appendingPathComponent("documents"))
    var d = CharacterSheetDocument(title: "Alex")
    d.voiceBinding = CharacterVoiceBinding(
      profile: VoiceProfileSnapshot(
        name: "Alex", samples: [.init(path: audio.path, duration: 1, transcript: "Hi")]))
    let exported = directory.appendingPathComponent("export")
    try storage.export(d, to: exported)
    try FileManager.default.removeItem(at: audio)
    let restored = try storage.importDocument(from: exported)
    let retained = restored.voiceBinding!.profile.samples[0]
    XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: retained.path)), Data([1, 2, 3]))
    XCTAssertEqual(retained.transcript, "Hi")
  }
}
