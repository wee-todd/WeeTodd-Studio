import XCTest
@testable import StudioCore

final class CreativeBriefTests: XCTestCase {
  private func brief() throws -> CreativeBrief {
    try JSONDecoder().decode(CreativeBrief.self, from: Data(#"""
    {"sourceText":"A pilot enters.\nKeep this exact text.","facts":[{"text":"A pilot enters.","evidence":"A pilot enters."}],"questions":[{"id":"identity-1","prompt":"Which pilot identity?","options":["Original fictional adult pilot","Use the supplied reference"],"answer":"","requiresExplicitChoice":true},{"id":"camera-1","prompt":"Camera approach?","options":["Static","Let director decide"],"answer":"","requiresExplicitChoice":false}],"preferences":{"durationSeconds":30,"targetClipSeconds":5,"frameRate":24,"visualStyle":"Cinematic","presentation":"Landscape","cameraStyle":"Gentle","audioStyle":"Dialogue and ambience","designPolicy":"Propose details for review","constraints":"No captions"},"referenceObservations":[{"referenceAsset":"portrait","observations":["Blue jacket"],"unknownDetail":{"keep":true}}]}
    """#.utf8))
  }

  func testAnswerEditsPreserveImmutableEvidenceAndReferenceObservations() throws {
    let original = try brief()
    var preferences = original.preferences
    preferences.visualStyle = "Watercolor"
    let edited = try original.editing(answers: ["identity-1": "Original fictional adult pilot", "camera-1": "Let director decide"], preferences: preferences)
    XCTAssertTrue(edited.isReady)
    XCTAssertEqual(edited.sourceText, "A pilot enters.\nKeep this exact text.")
    XCTAssertEqual(edited.facts, original.facts)
    XCTAssertEqual(edited.referenceObservations, original.referenceObservations)
    XCTAssertEqual(edited.questions.map(\.id), ["identity-1", "camera-1"])
    XCTAssertEqual(edited.questions.map(\.prompt), original.questions.map(\.prompt))
    XCTAssertEqual(edited.questions.map(\.options), original.questions.map(\.options))
    XCTAssertEqual(edited.questions.map(\.requiresExplicitChoice), [true, false])
    XCTAssertEqual(edited.preferences.visualStyle, "Watercolor")
    XCTAssertEqual(original.questions.map(\.answer), ["", ""])
    let outputs = try edited.outputs()
    XCTAssertEqual(Set(outputs.keys), ["creative_brief"])
    let roundTrip = try JSONDecoder().decode(CreativeBrief.self, from: JSONEncoder().encode(outputs["creative_brief"]!))
    XCTAssertEqual(roundTrip, edited)
  }

  func testEveryQuestionNeedsAnAnswerAndIdentityCannotDelegate() throws {
    let original = try brief()
    XCTAssertFalse(original.isReady)
    let incomplete = try original.editing(answers: ["identity-1": "Original fictional adult pilot", "camera-1": " \n "], preferences: original.preferences)
    XCTAssertFalse(incomplete.isReady)
    for answer in ["Let director decide", "  LET DIRECTOR DECIDE  ", "You decide", "Either", "Whatever you like", "Surprise me!", "Director—decide"] {
      let delegated = try original.editing(answers: ["identity-1": answer, "camera-1": "Static"], preferences: original.preferences)
      XCTAssertFalse(delegated.isReady, answer)
    }
    XCTAssertThrowsError(try original.editing(answers: ["invented-id": "Answer"], preferences: original.preferences))
    let explicitAlternative = try original.editing(answers: ["identity-1": "Neither: an adult human pilot with a prosthetic arm.", "camera-1": "Static"], preferences: original.preferences)
    XCTAssertTrue(explicitAlternative.isReady)
  }

  func testInvalidPreferenceBoundsAndNonfiniteValuesCannotBeSaved() throws {
    let original = try brief()
    for duration in [0, 3601, Double.infinity, Double.nan] {
      var preferences = original.preferences; preferences.durationSeconds = duration
      XCTAssertThrowsError(try original.editing(answers: [:], preferences: preferences))
    }
    for seconds in [0, 61, Double.infinity, Double.nan] {
      var preferences = original.preferences; preferences.targetClipSeconds = seconds
      XCTAssertThrowsError(try original.editing(answers: [:], preferences: preferences))
    }
    for fps in [0, 121] {
      var preferences = original.preferences; preferences.frameRate = fps
      XCTAssertThrowsError(try original.editing(answers: [:], preferences: preferences))
    }
    var preferences = original.preferences; preferences.constraints = String(repeating: "x", count: 2001)
    XCTAssertThrowsError(try original.editing(answers: [:], preferences: preferences))
  }

  func testPreviewRetainsExactMultimodalPromptAndDraftWarnings() throws {
    let data = Data(#"""
    {"status":"draft","warnings":["Review continuity before rendering."],"prompts":[{"clipID":"clip-1","durationSeconds":5,"integrated_multimodal_description":"A pilot says ‘Hello’.","overall_soundscape":"Cabin hum","non_diegetic_music":"None","prompt":"[Scene]\nA pilot says ‘Hello’.\n[Sound]\nCabin hum","referenceAssets":["portrait"],"subjectIDs":["pilot"]}]}
    """#.utf8)
    let preview = try JSONDecoder().decode(H3PromptPreview.self, from: data)
    XCTAssertEqual(preview.status, "draft")
    XCTAssertEqual(preview.warnings, ["Review continuity before rendering."])
    XCTAssertEqual(preview.prompts[0].prompt, "[Scene]\nA pilot says ‘Hello’.\n[Sound]\nCabin hum")
    XCTAssertEqual(preview.prompts[0].integratedMultimodalDescription, "A pilot says ‘Hello’.")
    let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(preview)) as! [String: Any]
    let prompt = (encoded["prompts"] as! [[String: Any]])[0]
    XCTAssertEqual(prompt["overall_soundscape"] as? String, "Cabin hum")
    XCTAssertEqual(prompt["non_diegetic_music"] as? String, "None")
    XCTAssertEqual(prompt["referenceAssets"] as? [String], ["portrait"])
    XCTAssertNil(prompt["integratedMultimodalDescription"])
  }
}
