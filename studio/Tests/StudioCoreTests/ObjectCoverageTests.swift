import XCTest
@testable import StudioCore

final class ObjectCoverageTests: XCTestCase {
  func testSemanticPhraseAnchorsRequireExactSourceAndLinkedIdentity() {
    let text = "She opens her jacket; her jacket catches the light."
    let target = DescriptionLinkTarget(id: "coat", name: "Copper jacket", description: "Copper leather")
    let mention = DescriptionMention(id: "m", targetID: "coat", phrase: "her jacket", occurrence: 1)
    let links = DescriptionLinks.ranges(in: text, targets: [target], mentions: [mention], sourceDescription: text)
    XCTAssertEqual(links.count, 1)
    XCTAssertEqual(links.first?.range.location, (text as NSString).range(of: "her jacket", options: .backwards).location)
    XCTAssertTrue(DescriptionLinks.ranges(in: text + "!", targets: [target], mentions: [mention], sourceDescription: text).isEmpty)
    XCTAssertTrue(DescriptionLinks.ranges(in: text, targets: [], mentions: [mention], sourceDescription: text).isEmpty)
  }
  func testCatalogFindsNameAliasAndTagAndKeepsPinnedIdentity() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    var coat = PlanningSubject(name: "Copper jacket", kind: .clothing)
    coat.details = "Copper leather"; coat.aliases = "pilot coat"; coat.tags = ["cyberpunk"]
    var plan = ProjectPlanning(); plan.subjects = [coat]
    let library = try ProductionLibrary(url: folder.appendingPathComponent("catalog.sqlite"))
    let package = try library.publish(rootID: coat.id, planning: plan, assets: [])
    for query in ["She wears the Copper jacket", "pilot coat", "cyberpunk"] {
      let candidates = try library.candidates(matching: query)
      XCTAssertEqual(candidates.first?.id, coat.id.uuidString)
      XCTAssertEqual(candidates.first?.version, package.version)
      XCTAssertEqual(candidates.first?.definitionRevision, coat.revision)
    }
    XCTAssertTrue(try library.candidates(matching: "swimming dolphin").isEmpty)
  }
  func testExplicitReuseRemapsLinksAndMentionsWithoutChangingLibraryDefinition() throws {
    var extracted = PlanningSubject(name: "Coat", kind: .clothing); extracted.details = "Draft"; extracted.sourceKey = "run:subject:clothing:coat"
    var reusable = PlanningSubject(name: "Copper jacket", kind: .clothing); reusable.details = "Copper leather"
    var actor = PlanningSubject(name: "Mara", kind: .character); actor.details = "Her jacket glows."
    actor.relationships = [ObjectRelationship(targetID: extracted.id, role: .wears)]
    actor.descriptionMentions = [DescriptionMention(id: "m", targetID: extracted.id.uuidString, phrase: "Her jacket", occurrence: 0)]
    actor.mentionSourceDescription = actor.details
    var plan = ProjectPlanning(); plan.subjects = [extracted, reusable, actor]
    var movie = StudioProject(); movie.planning = plan
    try movie.reuseLibraryObject(replacing: extracted.id, targetID: reusable.id)
    let updated = movie.planning!.subjects.first { $0.id == actor.id }!
    XCTAssertEqual(updated.relationships?.first?.targetID, reusable.id)
    XCTAssertEqual(updated.descriptionMentions?.first?.targetID, reusable.id.uuidString)
    XCTAssertEqual(movie.planning?.subjects.first { $0.id == reusable.id }?.details, "Copper leather")
    XCTAssertTrue(movie.planning!.subjects.first { $0.id == reusable.id }!.sourceKeyAliases.contains(extracted.sourceKey))
    XCTAssertEqual(movie.planning?.subjects.count, 2)
  }

  func testReuseSelectionRejectsChangedSourceAndDependencies() throws {
    let data = Data(#"{"id":"coat","name":"Coat","kind":"clothing","description":"Copper leather","aliases":[],"evidence":["Source"],"suggestions":[],"coverageReview":{"version":1,"status":"ready","reviewedDescription":"Copper leather","issues":[],"missingObjects":[],"libraryMatches":[{"objectID":"abc","packageID":"pkg","version":1,"reason":"Same coat","definitionRevision":"rev"}]}}"#.utf8)
    var subject = try JSONDecoder().decode(WorkflowSubjectProposal.self, from: data)
    var match = subject.coverageReview!.libraryMatches[0]; match.sourceRevision = subject.reuseSelectionRevision
    XCTAssertTrue(subject.validatesReuse(match))
    subject.description = "Blue silk"; XCTAssertFalse(subject.validatesReuse(match))
    var actor = PlanningSubject(name: "Actor", kind: .character); actor.details = "Person"
    var coat = PlanningSubject(name: "Coat", kind: .clothing); coat.details = "Copper leather"
    actor.relationships = [ObjectRelationship(targetID: coat.id, role: .wears)]
    var plan = ProjectPlanning(); plan.subjects = [actor, coat]
    let before = try plan.reusableObjectRevision(actor.id, assets: [])
    plan.subjects[1].details = "Blue silk"
    XCTAssertNotEqual(before, try plan.reusableObjectRevision(actor.id, assets: []))
  }

  func testGuidedReuseSurvivesCoverageRefreshButRejectsDifferentPinnedDefinition() throws {
    let data = Data(#"{"id":"coat","name":"Coat","kind":"clothing","description":"Copper leather","aliases":[],"evidence":["Source"],"suggestions":[],"reusedOriginalDescription":"An unspecified coat","reusedDefinition":{"objectID":"abc","packageID":"pkg","version":2,"scope":"global","definitionRevision":"rev2"}}"#.utf8)
    let decoded = try JSONDecoder().decode(WorkflowSubjectProposal.self, from: data)
    let subject = try JSONDecoder().decode(WorkflowSubjectProposal.self, from: JSONEncoder().encode(decoded))
    XCTAssertEqual(subject.reusedOriginalDescription, "An unspecified coat")
    XCTAssertNil(subject.coverageReview)
    var match = try XCTUnwrap(subject.reusedDefinition).libraryMatch
    match.sourceRevision = subject.reuseSelectionRevision
    XCTAssertTrue(subject.validatesReuse(match))
    match.version = 3
    XCTAssertFalse(subject.validatesReuse(match))
    match.version = 2; match.definitionRevision = "new definition"
    XCTAssertFalse(subject.validatesReuse(match))
    match.definitionRevision = "rev2"; match.scope = "movie"
    XCTAssertFalse(subject.validatesReuse(match))
  }

}
