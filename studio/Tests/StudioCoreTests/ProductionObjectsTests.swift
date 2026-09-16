import XCTest
@testable import StudioCore

final class ProductionObjectsTests: XCTestCase {
  func testSetResolutionIncludesParentAndPropsButNotSiblingSets() throws {
    var environment = PlanningSubject(name: "Penthouse", kind: .environment)
    var pool = PlanningSubject(name: "Pool", kind: .set)
    var gallery = PlanningSubject(name: "Gallery", kind: .set)
    let chair = PlanningSubject(name: "Chair", kind: .prop)
    pool.environmentID = environment.id; gallery.environmentID = environment.id
    pool.relationships = [ObjectRelationship(targetID: chair.id, role: .contains, placement: "Left"),
                          ObjectRelationship(targetID: chair.id, role: .contains, placement: "Right")]
    environment.details = "Concrete architecture"
    var plan = ProjectPlanning(); plan.subjects = [environment, pool, gallery, chair]
    XCTAssertEqual(Set(try plan.resolvedObjects([pool.id]).map(\.id)), Set([pool.id, environment.id, chair.id]))
    XCTAssertEqual(try plan.resolvedObjects([environment.id], includeSets: true).count, 4)
    XCTAssertEqual(pool.relationships?.count, 2)
    XCTAssertThrowsError(try plan.removeSubject(chair.id))
    plan.subjects[1].relationships?.append(ObjectRelationship(targetID: UUID(), role: .uses))
    XCTAssertThrowsError(try plan.resolvedObjects([pool.id]))
  }

  func testLinkedDefinitionChangeInvalidatesShotApprovalAndCycleIsSafe() throws {
    var person = PlanningSubject(name: "Actor", kind: .character); person.details = "Face"
    var coat = PlanningSubject(name: "Coat", kind: .clothing); coat.details = "Blue wool"
    person.relationships = [ObjectRelationship(targetID: coat.id, role: .wears)]
    coat.relationships = [ObjectRelationship(targetID: person.id, role: .uses)]
    var plan = ProjectPlanning(); plan.subjects = [person, coat]
    for subject in plan.subjects { try plan.approveSubject(subject.id) }
    var shot = PlanningShot(name: "Walk", frameCount: 121)
    shot.action = "Walk"; shot.firstFrame = "Door"; shot.lastFrame = "Window"; shot.subjectIDs = [person.id]
    plan.shots = [shot]; try plan.approveShot(shot.id)
    XCTAssertTrue(plan.isShotApproved(shot.id))
    plan.subjects[1].details = "Red wool"
    XCTAssertFalse(plan.isShotApproved(shot.id))
  }

  func testLegacySubjectKeepsApprovalDigest() throws {
    var subject = PlanningSubject(name: "Pool", kind: .location); subject.details = "Stone pool"
    let oldRevision = planningDigest([subject.name, subject.kind.rawValue, subject.aliases, subject.details, subject.evidence, subject.suggestions])
    subject.approvedRevision = oldRevision
    let value = try JSONDecoder().decode(PlanningSubject.self, from: JSONEncoder().encode(subject))
    XCTAssertEqual(value.revision, oldRevision)
    XCTAssertNil(value.relationships)
    XCTAssertEqual(value.kind, .location)
  }
  func testDependencyReferenceApprovalBecomesStaleAndMergeRollsBack() throws {
    let imageURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try Data([1, 2, 3]).write(to: imageURL); defer { try? FileManager.default.removeItem(at: imageURL) }
    let image = MediaAsset(name: "Reference", kind: .image, path: imageURL.path)
    var actor = PlanningSubject(name: "Actor", kind: .character); actor.details = "Person"; actor.referenceAssetIDs = [image.id]
    var coat = PlanningSubject(name: "Coat", kind: .clothing); coat.details = "Blue"
    actor.relationships = [ObjectRelationship(targetID: coat.id, role: .wears)]
    var plan = ProjectPlanning(); plan.subjects = [actor, coat]
    try plan.approveSubject(actor.id); try plan.approveReferences(actor.id, assets: [image])
    XCTAssertTrue(plan.areReferencesApproved(actor.id, assets: [image]))
    plan.subjects[1].details = "Red"; try plan.approveSubject(coat.id)
    XCTAssertFalse(plan.isSubjectApproved(actor.id))
    XCTAssertFalse(plan.areReferencesApproved(actor.id, assets: [image]))
    var other = PlanningSubject(name: "Other actor", kind: .character); other.details = "Other"
    other.relationships = (0..<32).map { ObjectRelationship(targetID: coat.id, role: .uses, placement: String($0)) }
    plan.subjects.append(other)
    let before = plan
    XCTAssertThrowsError(try plan.mergeSubject(other.id, into: actor.id))
    XCTAssertEqual(plan, before)
  }

  func testWorkflowLinksMapIDsTransactionallyAndKeepIDsOnReimport() throws {
    let json = #"{"status":"paused","totalSeconds":0,"outputs":{},"steps":{"subjects_links":{"name":"Links","status":"completed","outputs":{"subjects":[{"id":"actor","kind":"character","name":"Actor","description":"Person","aliases":[],"evidence":[],"suggestions":[],"relationships":[{"id":"link_coat","targetID":"coat","role":"wears","placement":"Closed"}]},{"id":"coat","kind":"clothing","name":"Coat","description":"Blue wool","aliases":[],"evidence":[],"suggestions":[]}]}}}}"#
    let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: Data(json.utf8))
    var plan = ProjectPlanning(); try plan.importRun(run, sourceID: "run", sourceText: "Script")
    let actor = plan.subjects.first { $0.name == "Actor" }!
    let coat = plan.subjects.first { $0.name == "Coat" }!
    XCTAssertEqual(actor.relationships?.first?.targetID, coat.id)
    let originalLink = actor.relationships?.first?.id
    try plan.importRun(run, sourceID: "run", sourceText: "Script")
    XCTAssertEqual(plan.subjects.first { $0.id == actor.id }?.relationships?.first?.id, originalLink)
    let invalid = json.replacingOccurrences(of: "\"id\":\"coat\",\"kind\"", with: "\"id\":\"actor\",\"kind\"")
    let badRun = try JSONDecoder().decode(WorkflowRunSummary.self, from: Data(invalid.utf8))
    let before = plan
    XCTAssertThrowsError(try plan.importRun(badRun, sourceID: "bad", sourceText: "Script"))
    XCTAssertEqual(plan, before)
  }

  func testShotOverridesAndExportKeepDefinitionsStableAndMediaLinked() throws {
    var environment = PlanningSubject(name: "Penthouse", kind: .environment); environment.details = "Concrete"
    let image = MediaAsset(name: "Architecture", kind: .image, path: "/linked/architecture.png")
    environment.referenceAssetIDs = [image.id]
    var shot = PlanningShot(name: "Night", frameCount: 121); shot.subjectIDs = [environment.id]
    shot.appearanceOverrides = [ObjectStateOverride(subjectID: environment.id, state: "Night; rain on glass")]
    var plan = ProjectPlanning(); plan.subjects = [environment]; plan.shots = [shot]
    let snapshot = try plan.resolvedShotSnapshot(shot.id, assets: [image], requireApproval: false)
    XCTAssertEqual(snapshot.appearanceOverrides?.first?.state, "Night; rain on glass")
    XCTAssertEqual(snapshot.subjects.first?.details, "Concrete")
    let exported = plan.exportDocument(assets: [image])
    XCTAssertEqual(exported.referenceAssets.map(\.path), [image.path])
    XCTAssertEqual(exported.resolvedShots[shot.id.uuidString]?.revision, snapshot.revision)
    XCTAssertEqual(exported.format, "weetodd-shot-list-v2")
    XCTAssertTrue(exported.issues.isEmpty)
  }

  func testInvalidDraftNeverAppearsApprovedAndDanglingOverrideIsReported() throws {
    var set = PlanningSubject(name: "Pool", kind: .set); set.details = "Stone basin"
    var plan = ProjectPlanning(); plan.subjects = [set]
    XCTAssertFalse(plan.isSubjectApproved(set.id))
    plan.subjects[0].kind = .prop
    plan.subjects[0].relationships = [ObjectRelationship(targetID: UUID(), role: .uses)]
    XCTAssertFalse(plan.isSubjectApproved(set.id))
    var shot = PlanningShot(name: "Empty", frameCount: 121)
    shot.appearanceOverrides = [ObjectStateOverride(subjectID: set.id, state: "Night")]
    plan.shots = [shot]
    XCTAssertThrowsError(try plan.resolvedShotSnapshot(shot.id, assets: [], requireApproval: false))
    XCTAssertFalse(plan.exportDocument(assets: []).issues.isEmpty)
  }

}
