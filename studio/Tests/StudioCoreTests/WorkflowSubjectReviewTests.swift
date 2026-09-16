import XCTest
@testable import StudioCore

final class WorkflowSubjectReviewTests: XCTestCase {
  private let json = #"[{"id":"prop-1","kind":"prop","name":"Crystal","description":"Blue","aliases":[],"evidence":["Source"],"suggestions":["Proposal"]},{"id":"char-z","kind":"character","name":"Zoe","description":"Pilot","aliases":[],"evidence":[],"suggestions":[]},{"id":"env-1","kind":"location","name":"Roof","description":"Wet","aliases":[],"evidence":[],"suggestions":[]},{"id":"char-a","kind":"character","name":"Ada","description":"Engineer","aliases":[],"evidence":[],"suggestions":[]}]"#

  func testReviewGroupsByTypeThenNameAndPreservesSourceOrder() throws {
    let items = try JSONDecoder().decode([WorkflowSubjectProposal].self, from: Data(json.utf8))
    let groups = WorkflowSubjectGroup.group(items)
    XCTAssertEqual(groups.map(\.title), ["Characters", "Environments", "Sets", "Locations (unclassified)", "Props", "Clothing", "Outfits"])
    XCTAssertEqual(groups[0].subjects.map(\.name), ["Ada", "Zoe"])
    XCTAssertEqual(items.map(\.id), ["prop-1", "char-z", "env-1", "char-a"])
  }

  func testFocusedGroupingKeepsOnlyPopulatedKindsWithoutLosingSubjects() throws {
    let items = try JSONDecoder().decode([WorkflowSubjectProposal].self, from: Data(json.utf8))
    let groups = WorkflowSubjectGroup.group(items, includeEmpty: false)
    XCTAssertEqual(groups.map(\.title), ["Characters", "Locations (unclassified)", "Props"])
    XCTAssertEqual(groups.flatMap(\.subjects).map(\.id), ["char-a", "char-z", "env-1", "prop-1"])
    XCTAssertTrue(WorkflowSubjectGroup.group([], includeEmpty: false).isEmpty)
  }
  func testCollapsedNotesSummarizeAllIssueSourcesAndUnestablishedDesignProposals() throws {
    var subject = try JSONDecoder().decode([WorkflowSubjectProposal].self, from: Data(json.utf8))[0]
    subject.descriptionReview = .init(version: 1, status: "needs_attention", reviewedDescription: "Blue", criteria: [], proposedDetails: ["Gold setting"], issues: ["Image conflicts with source"], referenceAssets: [])
    subject.relationshipReview = .init(version: 1, status: "needs_attention", reviewedDescription: "Blue", issues: ["Missing owner"], missingObjects: ["Case"])
    subject.coverageReview = .init(version: 1, status: "needs_attention", reviewedDescription: "Blue", issues: ["Image conflicts with source", "No linked set"], missingObjects: [], libraryMatches: [])
    let attention = DirectorSubjectAttention(subject: subject)
    XCTAssertEqual(attention.issues, ["Image conflicts with source", "Missing owner", "No linked set"])
    XCTAssertEqual(attention.proposedDetailCount, 1)
    XCTAssertEqual(attention.missingObjectCount, 1)
    XCTAssertTrue(attention.needsAttention)
    XCTAssertTrue(DirectorSubjectAttention(subject: subject, currentDescription: "Red").hasOutdatedNotes)
    XCTAssertFalse(DirectorSubjectAttention(subject: subject, currentDescription: "Blue").hasOutdatedNotes)
  }
  func testOnlyNamesAndDescriptionsChangeInSubjectEditor() throws {
    let item = try JSONDecoder().decode([WorkflowSubjectProposal].self, from: Data(json.utf8))[0]
    let updated = try item.editing(name: "Data crystal", description: "Faceted blue crystal")
    XCTAssertEqual(updated.id, item.id)
    XCTAssertEqual(updated.evidence, ["Source"])
    XCTAssertEqual(updated.suggestions, ["Proposal"])
    XCTAssertEqual(updated.kind, .prop)
    XCTAssertEqual(updated.name, "Data crystal")
    XCTAssertEqual(updated.description, "Faceted blue crystal")
    XCTAssertThrowsError(try item.editing(name: "  ", description: "Blue"))
  }

  func testClassificationCorrectionKeepsIdentityAndEvidence() throws {
    var item = try JSONDecoder().decode([WorkflowSubjectProposal].self, from: Data(json.utf8))[2]
    let original = item
    item.kind = .set
    XCTAssertEqual(item.id, original.id)
    XCTAssertEqual(item.description, original.description)
    XCTAssertEqual(item.evidence, original.evidence)
    let group = WorkflowSubjectGroup.group([item]).first { $0.kind == .set }
    XCTAssertEqual(group?.subjects.first?.id, original.id)
  }

  func testImportRetainsExplicitSubjectApprovalsWithoutApprovingOtherSubjects() throws {
    let output = try JSONSerialization.jsonObject(with: Data(json.utf8))
    let data = try JSONSerialization.data(withJSONObject: [
      "status": "paused", "totalSeconds": 0, "outputs": [:],
      "steps": ["subjects": ["name": "Subjects", "status": "completed",
        "outputs": ["subjects": output],
        "items": ["prop-1": ["status": "completed", "approved": true]]]]
    ])
    let run = try JSONDecoder().decode(WorkflowRunSummary.self, from: data)
    XCTAssertEqual(run.preferredReviewStepID, "subjects")
    var plan = ProjectPlanning()
    try plan.importRun(run, sourceID: "run", sourceText: "Source")
    XCTAssertTrue(plan.isSubjectApproved(plan.subjects[0].id))
    XCTAssertFalse(plan.isSubjectApproved(plan.subjects[1].id))
    plan.subjects[0].details = "User correction"
    try plan.importRun(run, sourceID: "run", sourceText: "Source")
    XCTAssertEqual(plan.subjects[0].details, "User correction")
    XCTAssertFalse(plan.isSubjectApproved(plan.subjects[0].id))
  }
  func testDescriptionReviewProvenanceAndStaleApproval() throws {
    let data = Data(#"{"version":1,"status":"ready","reviewedDescription":"Steel armor.","criteria":["materials"],"proposedDetails":[],"referenceDetails":["Image 1 · materials: Steel armor."],"issues":[],"referenceAssets":["portrait"]}"#.utf8)
    let report = try JSONDecoder().decode(SubjectDescriptionReview.self, from: data)
    XCTAssertTrue(report.isReady(for: "Steel armor."))
    XCTAssertFalse(report.isReady(for: "Gold armor."))
    var plan = ProjectPlanning()
    var subject = PlanningSubject(name: "Warrior", kind: .character)
    subject.details = "Gold armor."; subject.descriptionReview = report
    plan.subjects = [subject]
    XCTAssertFalse(plan.subjects[0].descriptionReview!.isReady(for: plan.subjects[0].details))
    plan.subjects[0].details = "Steel armor."
    try plan.approveSubject(subject.id)
    XCTAssertTrue(plan.isSubjectApproved(subject.id))
    plan.subjects[0].descriptionReview?.status = "needs_attention"
    XCTAssertFalse(plan.subjects[0].descriptionReview!.isReady(for: plan.subjects[0].details))
    XCTAssertEqual(plan.subjects[0].descriptionReview?.referenceDetails?.count, 1)
  }

}
