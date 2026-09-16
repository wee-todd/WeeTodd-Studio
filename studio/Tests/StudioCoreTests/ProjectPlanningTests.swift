import XCTest
@testable import StudioCore

final class ProjectPlanningTests: XCTestCase {
  func testExternalGuidedRunImportsApprovedInventoryWithoutDuplicateObjects() throws {
    guard let runPath = ProcessInfo.processInfo.environment["WEETODD_GUIDED_RUN_PATH"] else {
      throw XCTSkip("Set WEETODD_GUIDED_RUN_PATH to a completed local qualification run.json.")
    }
    let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: Data(contentsOf: URL(fileURLWithPath: runPath)))
    XCTAssertEqual(run.status, "completed")
    let inventory = try run.subjectsForImport()
    let previewValue = try XCTUnwrap(run.outputs["h3_prompts"])
    let preview = try JSONDecoder().decode(H3PromptPreview.self, from: JSONEncoder().encode(previewValue))
    var plan = ProjectPlanning()
    try plan.importRun(run, sourceID: "qualification", sourceText: "Local qualification fixture")
    XCTAssertEqual(plan.subjects.count, inventory.count)
    XCTAssertEqual(plan.shots.count, preview.prompts.count)
    for prompt in preview.prompts {
      let shot = try XCTUnwrap(plan.shots.first { $0.sourceKey == "qualification:shot:" + prompt.clipID })
      XCTAssertEqual(shot.direction, prompt.prompt)
      for reference in prompt.subjectIDs {
        let subject = try XCTUnwrap(plan.subjects.first { $0.sourceKey.hasSuffix(":" + reference) })
        XCTAssertTrue(shot.subjectIDs.contains(subject.id))
      }
    }
  }

  func testGuidedImportUsesApprovedObjectIDsForCastSetsAndProps() throws {
    let data = Data(#"""
    {"status":"completed","totalSeconds":1,"outputs":{},"steps":{
      "subjects_coverage":{"status":"completed","name":"Subjects","approved":true,"outputs":{"subjects":[
        {"id":"character_hash","kind":"character","name":"Cat Woman","description":"Approved character design","aliases":[],"evidence":[],"suggestions":[]},
        {"id":"set_hash","kind":"set","name":"Vault","description":"Approved vault design","aliases":[],"evidence":[],"suggestions":[]},
        {"id":"prop_hash","kind":"prop","name":"Crystal","description":"Approved crystal design","aliases":[],"evidence":[],"suggestions":[]}
      ]}},
      "clips":{"status":"completed","name":"Shots","outputs":{"clips":{"fps":24,"totalFrames":120,"characters":[{"id":"character_hash","description":"Short cast summary"}],"clips":[{"id":"clip-1","startFrame":0,"frameCount":120,"action":"She takes the crystal","startState":"Standing","endState":"Holding crystal","location":"Inside the vault","characters":["character_hash"],"continuity":"cut"}]}}},
      "prompt_preview":{"status":"completed","name":"Prompt","approved":true,"outputs":{"h3_prompts":{"status":"draft","warnings":[],"prompts":[{"clipID":"clip-1","durationSeconds":5,"integrated_multimodal_description":"Reviewed scene","overall_soundscape":"Room tone","non_diegetic_music":"None","prompt":"Reviewed scene","referenceAssets":[],"subjectIDs":["character_hash","set_hash","prop_hash"]}]}}}
    }}
    """#.utf8)
    let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: data)
    for replacement in [
      String(decoding: data, as: UTF8.self).replacingOccurrences(of: "\"subjectIDs\":[\"character_hash\",\"set_hash\",\"prop_hash\"]", with: "\"subjectIDs\":[\"missing_object\"]"),
      String(decoding: data, as: UTF8.self).replacingOccurrences(of: "\"clipID\":\"clip-1\"", with: "\"clipID\":\"missing_shot\"")
    ] {
      let invalid = try JSONDecoder().decode(WorkflowRunSummary.self, from: Data(replacement.utf8))
      var untouched = ProjectPlanning()
      XCTAssertThrowsError(try untouched.importRun(invalid, sourceID: "invalid", sourceText: "A heist"))
      XCTAssertTrue(untouched.subjects.isEmpty)
      XCTAssertTrue(untouched.sourceScripts.isEmpty)
    }
    var plan = ProjectPlanning()
    try plan.importRun(run, sourceID: "guided", sourceText: "A heist")
    XCTAssertEqual(plan.subjects.count, 3)
    XCTAssertEqual(plan.subjects.filter { $0.kind == .character }.count, 1)
    XCTAssertEqual(plan.subjects.first { $0.kind == .character }?.details, "Approved character design")
    XCTAssertEqual(Set(plan.shots[0].subjectIDs), Set(plan.subjects.map(\.id)))
    XCTAssertEqual(plan.shots[0].direction, "Reviewed scene")
    let original = try XCTUnwrap(plan.subjects.first { $0.kind == .character })
    var replacement = PlanningSubject(name: "Reusable Cat Woman", kind: .character)
    replacement.details = "Library design"
    plan.subjects.append(replacement)
    var project = StudioProject(); project.planning = plan
    try project.reuseLibraryObject(replacing: original.id, targetID: replacement.id)
    XCTAssertTrue(project.planning!.shots[0].subjectIDs.contains(replacement.id))
    XCTAssertFalse(project.planning!.shots[0].subjectIDs.contains(original.id))
  }
  func testLegacyProjectAndPlanningRoundTrip() throws {
    let data = try JSONEncoder().encode(StudioProject())
    XCTAssertNil(try JSONDecoder().decode(StudioProject.self, from: data).planning)
    var p = StudioProject(); p.planning = ProjectPlanning()
    p.planning?.sourceText = "Original script with exact dialogue."
    XCTAssertEqual(try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(p)), p)
  }
  func testSubjectEditInvalidatesShotButNotOtherSubjects() throws {
    var plan = ProjectPlanning()
    var character = PlanningSubject(name: "Dog", kind: .character)
    character.details = "One natural brown eye and one sapphire optical eye."
    plan.subjects = [character]
    try plan.approveSubject(character.id)
    var shot = PlanningShot(name: "Greeting", frameCount: 120)
    shot.action = "Dog greets the woman"; shot.firstFrame = "Dog waits"; shot.lastFrame = "Dog wags"
    shot.subjectIDs = [character.id]; plan.shots = [shot]
    try plan.approveShot(shot.id)
    XCTAssertTrue(plan.isShotApproved(shot.id))
    plan.subjects[0].details = "Two optical eyes"
    XCTAssertFalse(plan.isSubjectApproved(character.id))
    XCTAssertFalse(plan.isShotApproved(shot.id))
    try plan.approveSubject(character.id)
    XCTAssertFalse(plan.isShotApproved(shot.id))
  }
  func testReferencesRequireImagesAndTheirOwnApproval() throws {
    var plan = ProjectPlanning()
    let subject = PlanningSubject(name: "Crystal", kind: .prop)
    plan.subjects = [subject]
    XCTAssertThrowsError(try plan.approveSubject(subject.id))
    plan.subjects[0].details = "Blue crystal"
    try plan.approveSubject(subject.id)
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
    try Data([1, 2, 3]).write(to: file)
    defer { try? FileManager.default.removeItem(at: file) }
    var image = MediaAsset(name: "Sheet", kind: .image, path: file.path)
    plan.subjects[0].referenceAssetIDs = [image.id]
    XCTAssertFalse(plan.areReferencesApproved(subject.id, assets: [image]))
    XCTAssertThrowsError(try plan.approveReferences(subject.id, assets: []))
    try plan.approveReferences(subject.id, assets: [image])
    XCTAssertTrue(plan.areReferencesApproved(subject.id, assets: [image]))
    try FileManager.default.removeItem(at: file)
    XCTAssertFalse(plan.areReferencesApproved(subject.id, assets: [image]))
    image.path = "/example/replaced.png"
    XCTAssertFalse(plan.areReferencesApproved(subject.id, assets: [image]))
    XCTAssertTrue(plan.isSubjectApproved(subject.id))
  }
  func testWorkflowImportPreservesEditsAndSource() throws {
    let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: Data(#"{"status":"awaiting_approval","totalSeconds":0,"outputs":{},"steps":{"story":{"name":"Story","status":"completed","outputs":{"story":{"characters":[{"id":"dog","description":"Small dog"}],"beats":["Dog jumps"]}}},"clips":{"name":"Clips","status":"completed","outputs":{"clips":{"fps":24,"totalFrames":120,"characters":[{"id":"dog","description":"Small dog"}],"clips":[{"id":"clip-1","startFrame":0,"frameCount":120,"action":"Dog jumps","startState":"Standing","endState":"Landing","location":"Roof","characters":["dog"],"continuity":"cut"}]}}}}}"#.utf8))
    var plan = ProjectPlanning()
    try plan.importRun(run, sourceID: "run-A", sourceText: "[Shot 1]\nExact camera and dialogue.")
    XCTAssertEqual(plan.shots.count, 1)
    XCTAssertEqual(plan.subjects.count, 2)
    XCTAssertEqual(plan.sourceText, "[Shot 1]\nExact camera and dialogue.")
    XCTAssertEqual(plan.sourceScripts["run-A"], plan.sourceText)
    XCTAssertTrue(plan.shots[0].direction.contains("Exact camera and dialogue."))
    let shotID = plan.shots[0].id
    plan.shots[0].action = "User correction"
    try plan.importRun(run, sourceID: "run-A", sourceText: "Changed input")
    XCTAssertEqual(plan.shots.count, 1)
    XCTAssertEqual(plan.shots[0].id, shotID)
    XCTAssertEqual(plan.shots[0].action, "User correction")
    XCTAssertEqual(plan.sourceText, "[Shot 1]\nExact camera and dialogue.")
  }
  private func inventoryRun(kind: String) throws -> WorkflowRunSummary {
    let data: [String: Any] = ["status": "completed", "totalSeconds": 0, "outputs": [:], "steps": [
      "subjects_coverage": ["status": "completed", "name": "Subjects", "approved": true, "outputs": [
        "subjects": [["id": "greenhouse", "kind": kind, "name": "Greenhouse",
          "description": "A glass greenhouse", "aliases": [], "evidence": [], "suggestions": []]]
      ]]
    ]]
    return try JSONDecoder().decode(WorkflowRunSummary.self, from: JSONSerialization.data(withJSONObject: data))
  }
  func testReclassifiedWorkflowSubjectReimportPreservesProjectObjectAndLegacyAliases() throws {
    for identity in ["current", "legacy", "merged"] {
      var plan = ProjectPlanning()
      try plan.importRun(inventoryRun(kind: "location"), sourceID: "run-A", sourceText: "Original story")
      if identity == "legacy" { plan.subjects[0].sourceKey = "run-A:subject:location:greenhouse" }
      if identity == "merged" {
        plan.subjects[0].sourceKey = "library:greenhouse"
        plan.subjects[0].sourceKeyAliases = ["run-A:subject:location:greenhouse"]
      }
      let id = plan.subjects[0].id
      plan.subjects[0].details = "User's reviewed greenhouse design"
      plan.subjects[0].referenceAssetIDs = [UUID()]
      try plan.approveSubject(id)
      var shot = PlanningShot(name: "Arrival", frameCount: 120)
      shot.subjectIDs = [id]; plan.shots = [shot]
      let saved = plan
      try plan.importRun(inventoryRun(kind: "environment"), sourceID: "run-A", sourceText: "Changed story")
      XCTAssertEqual(plan, saved, "Reclassification must not duplicate or overwrite the imported project object (\(identity))")
      try plan.importRun(inventoryRun(kind: "environment"), sourceID: "run-B", sourceText: "A different story")
      XCTAssertEqual(plan.subjects.count, 2, "A different workflow source must keep its own identity")
      XCTAssertNotEqual(plan.subjects[0].id, plan.subjects[1].id)
    }
  }
  func testReimportRejectsAmbiguousLegacySubjectsWithoutChangingProject() throws {
    var plan = ProjectPlanning()
    var a = PlanningSubject(name: "Greenhouse", kind: .location)
    a.sourceKey = "run-A:subject:location:greenhouse"
    var b = PlanningSubject(name: "Greenhouse copy", kind: .environment)
    b.sourceKeyAliases = ["run-A:subject:environment:greenhouse"]
    plan.subjects = [a, b]
    let saved = plan
    XCTAssertThrowsError(try plan.importRun(inventoryRun(kind: "set"), sourceID: "run-A", sourceText: "Story"))
    XCTAssertEqual(plan, saved)
  }
  func testContinueApprovalTracksPreviousEndingAndOrder() throws {
    var plan = ProjectPlanning()
    var a = PlanningShot(name: "A", frameCount: 24)
    a.action = "Walk"; a.firstFrame = "Standing"; a.lastFrame = "At door"
    var b = PlanningShot(name: "B", frameCount: 24)
    b.action = "Enter"; b.firstFrame = "At door"; b.lastFrame = "Inside"; b.continuity = "continue"
    plan.shots = [a,b]
    try plan.approveShot(a.id); try plan.approveShot(b.id)
    plan.shots[0].lastFrame = "At window"
    XCTAssertFalse(plan.isShotApproved(b.id))
    XCTAssertThrowsError(try plan.approveShot(b.id))
    plan.shots.reverse()
    XCTAssertThrowsError(try plan.approveShot(b.id))
  }
  func testExternalReferenceChangeInvalidatesDependentShot() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
    try Data([1,2,3]).write(to: file)
    defer { try? FileManager.default.removeItem(at: file) }
    var plan = ProjectPlanning()
    var subject = PlanningSubject(name: "Dog", kind: .character); subject.details = "One brown eye"
    let image = MediaAsset(name: "Sheet", kind: .image, path: file.path)
    subject.referenceAssetIDs = [image.id]; plan.subjects = [subject]
    try plan.approveSubject(subject.id); try plan.approveReferences(subject.id, assets: [image])
    var shot = PlanningShot(name: "A", frameCount: 24)
    shot.action = "Sit"; shot.firstFrame = "Here"; shot.lastFrame = "Seated"; shot.subjectIDs = [subject.id]
    plan.shots = [shot]; try plan.approveShot(shot.id, assets: [image])
    XCTAssertTrue(plan.isShotApproved(shot.id, assets: [image]))
    try FileManager.default.removeItem(at: file)
    XCTAssertFalse(plan.areReferencesApproved(subject.id, assets: [image]))
    XCTAssertFalse(plan.isShotApproved(shot.id, assets: [image]))
  }
  func testProjectReferenceLinksReuseGlobalFileWithoutDuplicates() throws {
    var project = StudioProject(); project.planning = ProjectPlanning()
    let subject = PlanningSubject(name: "Dog", kind: .character)
    project.planning?.subjects = [subject]
    let global = MediaAsset(name: "Sheet", kind: .image, path: "/references/dog.png", scope: .global)
    let first = try project.attachPlanningReference(global, subjectID: subject.id)
    let second = try project.attachPlanningReference(global, subjectID: subject.id)
    XCTAssertEqual(first, second)
    XCTAssertEqual(project.assets.count, 1)
    XCTAssertEqual(project.assets[0].scope, .project)
    XCTAssertEqual(project.assets[0].path, global.path)
    XCTAssertEqual(project.planning?.subjects[0].referenceAssetIDs, [first])
    let before = project
    XCTAssertThrowsError(try project.attachPlanningReference(global, subjectID: UUID()))
    XCTAssertEqual(project, before)
  }
  func testMergeRemapsShotsAndKeepsImportIdentity() throws {
    var plan = ProjectPlanning()
    var a = PlanningSubject(name: "Dog", kind: .character)
    a.details = "Brown eye"; a.sourceKey = "run:cast:dog"
    var b = PlanningSubject(name: "Cyborg dog", kind: .character)
    b.details = "Titanium forelegs"; b.sourceKey = "other:subject:character:dog"
    var shot = PlanningShot(name: "A", frameCount: 24); shot.subjectIDs = [a.id, b.id]
    plan.subjects = [a,b]; plan.shots = [shot]
    try plan.mergeSubject(b.id, into: a.id)
    XCTAssertEqual(plan.subjects.count, 1)
    XCTAssertEqual(plan.shots[0].subjectIDs, [a.id])
    XCTAssertTrue(plan.subjects[0].sourceKeyAliases.contains(b.sourceKey))
    XCTAssertTrue(plan.subjects[0].suggestions.contains("Titanium forelegs"))
    XCTAssertFalse(plan.isSubjectApproved(a.id))
  }
  func testReferenceChangesInvalidateShotApproval() throws {
    var plan = ProjectPlanning()
    var s = PlanningSubject(name: "Dog", kind: .character); s.details = "Brown eye"
    plan.subjects = [s]; try plan.approveSubject(s.id)
    var shot = PlanningShot(name: "A", frameCount: 24)
    shot.action = "Sit"; shot.firstFrame = "Here"; shot.lastFrame = "Seated"; shot.subjectIDs = [s.id]
    plan.shots = [shot]; try plan.approveShot(shot.id)
    plan.subjects[0].referenceAssetIDs.append(UUID())
    XCTAssertFalse(plan.isShotApproved(shot.id))
  }
  func testInvalidTimingAndMissingSubjectBlockApproval() throws {
    var plan = ProjectPlanning()
    var shot = PlanningShot(name: "A", frameCount: 0)
    shot.action = "Walk"; shot.firstFrame = "Here"; shot.lastFrame = "There"
    plan.shots = [shot]
    XCTAssertThrowsError(try plan.approveShot(shot.id))
    plan.shots[0].frameCount = 24
    plan.shots[0].subjectIDs = [UUID()]
    XCTAssertThrowsError(try plan.approveShot(shot.id))
  }
}
