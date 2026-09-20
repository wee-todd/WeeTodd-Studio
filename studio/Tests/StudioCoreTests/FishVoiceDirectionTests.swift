import Foundation
import XCTest
@testable import StudioCore

final class FishVoiceDirectionTests: XCTestCase {
  private func draft() -> VoiceDraft {
    var draft = VoiceDraft(); draft.engine = .fishS2Pro; draft.modelPath = "/model"
    draft.referenceMode = .synthetic; draft.text = "Hello"
    return draft
  }

  private func direction(_ description: String) -> FishVoiceDirection {
    var value = FishVoiceDirection(); value.description = description
    return value
  }

  func testStoredDirectionBecomesSeparateRequestTagsWithoutChangingScript() throws {
    var draft = VoiceDraft(); draft.engine = .fishS2Pro; draft.modelPath = "/model"
    draft.referenceMode = .synthetic; draft.text = "[excited] Hello, 世界!"
    var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(draft)) as! [String: Any]
    object["fishDirection"] = ["pitch": "low voice", "pace": "slow delivery", "timbre": "warm voice",
                               "accent": " British ", "description": "measured storyteller"]
    draft = try JSONDecoder().decode(VoiceDraft.self, from: JSONSerialization.data(withJSONObject: object))
    let request = try draft.request()
    let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as! [String: Any]
    XCTAssertEqual(encoded["voice_direction"] as? [String],
                   ["low voice", "slow delivery", "warm voice", "with British accent", "measured storyteller"])
    XCTAssertEqual(request.text, "[excited] Hello, 世界!")
  }

  func testEmptyDirectionPreservesLegacyWireShapeAndOldDraftDecoding() throws {
    var draft = draft()
    let legacyDraft = try JSONEncoder().encode(draft)
    let legacyRequest = try draft.request()
    XCTAssertNil(try JSONDecoder().decode(VoiceDraft.self, from: legacyDraft).fishDirection)
    XCTAssertNil(try JSONDecoder().decode(VoiceRequest.self, from: JSONEncoder().encode(legacyRequest)).voiceDirection)
    draft.fishDirection = FishVoiceDirection()
    XCTAssertEqual(try draft.request(), legacyRequest)
    let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(draft.request())) as! [String: Any]
    XCTAssertNil(object["voice_direction"])
    draft.startDialogue()
    draft.dialogue?.speakers[0].fishDirection = nil
    let decoded = try JSONDecoder().decode(VoiceDraft.self, from: JSONEncoder().encode(draft))
    XCTAssertNil(decoded.dialogue?.speakers[0].fishDirection)
    XCTAssertNil(decoded.dialogue?.turns[0].fishDirection)
    let request = try decoded.request()
    XCTAssertNil(try JSONDecoder().decode(VoiceTurnRequest.self,
      from: JSONEncoder().encode(XCTUnwrap(request.turns?.first))).voiceDirection)
  }

  func testDraftSpeakerAndLineDirectionsRoundTrip() throws {
    var draft = draft(); draft.fishDirection = direction("gentle storyteller")
    draft.startDialogue()
    XCTAssertEqual(draft.dialogue?.speakers[0].fishDirection, draft.fishDirection)
    draft.dialogue?.turns[0].fishDirection = direction("energetic announcer")
    XCTAssertEqual(try JSONDecoder().decode(VoiceDraft.self, from: JSONEncoder().encode(draft)), draft)
    let request = try draft.request()
    XCTAssertEqual(try JSONDecoder().decode(VoiceRequest.self, from: JSONEncoder().encode(request)), request)
  }

  func testControlsValidateChoicesAndUnsafeTextBeforeTrimming() throws {
    for key in [\FishVoiceDirection.pitch, \.pace, \.timbre] {
      var value = FishVoiceDirection(); value[keyPath: key] = "unsupported"
      XCTAssertThrowsError(try value.tags())
    }
    for unsafe in ["[angry]", "warm]", "]\u{301}", "<|speaker:1|>", "soft\n", "\twarm", "\rwarm", "warm\u{0}",
                   "warm\u{200B}", "warm\u{2028}", "warm\u{2029}"] {
      for key in [\FishVoiceDirection.accent, \.description] {
        var value = FishVoiceDirection(); value[keyPath: key] = unsafe
        XCTAssertThrowsError(try value.tags(), unsafe)
      }
    }
    var value = FishVoiceDirection(); value.accent = String(repeating: "a", count: 81)
    XCTAssertThrowsError(try value.tags())
    value.accent = String(repeating: "a", count: 80)
    XCTAssertNoThrow(try value.tags())
    value.description = String(repeating: "e\u{301}", count: 60)
    XCTAssertNoThrow(try value.tags())
    value.description += "e\u{301}"
    XCTAssertThrowsError(try value.tags())
    value = FishVoiceDirection(); value.accent = "  "; value.description = "  "
    XCTAssertEqual(try value.tags(), [])
  }

  func testDirectionPrefixAndScriptShareUTF8Budget() throws {
    var draft = draft(); draft.fishDirection = direction("暖")
    // Three UTF-8 bytes plus brackets and a trailing space leave 31,994 bytes.
    draft.text = String(repeating: "a", count: 31_994)
    XCTAssertNoThrow(try draft.request())
    draft.text += "a"
    XCTAssertThrowsError(try draft.request())
    draft.fishDirection = nil
    XCTAssertNoThrow(try draft.request())
    draft.text = String(repeating: "暖", count: 10_667)
    XCTAssertThrowsError(try draft.request())
  }

  func testDialogueUsesSpeakerAndLineDirectionWithoutGlobalFallback() throws {
    var draft = draft(); draft.fishDirection = direction("first voice")
    draft.startDialogue()
    var second = VoiceSpeaker(name: "Second"); second.referenceMode = .synthetic
    second.fishDirection = direction("second voice")
    draft.dialogue?.speakers.append(second)
    draft.dialogue?.turns.append(VoiceTurn(speakerID: second.id, text: "Second line"))
    draft.fishDirection = direction("unrelated global direction")
    XCTAssertEqual(try draft.request().turns?.map(\.voiceDirection), [["first voice"], ["second voice"]])
    draft.dialogue?.turns[1].fishDirection = direction("line override")
    XCTAssertEqual(try draft.request().turns?[1].voiceDirection, ["line override"])
    draft.dialogue?.turns[1].fishDirection = FishVoiceDirection()
    XCTAssertNil(try draft.request().turns?[1].voiceDirection)
    draft.dialogue?.turns[1].fishDirection = nil
    XCTAssertEqual(try draft.request().turns?[1].voiceDirection, ["second voice"])
    draft.dialogue?.speakers[1].fishDirection = nil
    XCTAssertNil(try draft.request().turns?[1].voiceDirection)
  }

  func testChangingSpeakerClearsDirectionOverrideOnlyWhenSpeakerChanges() throws {
    var draft = draft(); draft.startDialogue()
    let firstID = try XCTUnwrap(draft.dialogue?.speakers[0].id)
    let lineID = try XCTUnwrap(draft.dialogue?.turns[0].id)
    let second = VoiceSpeaker(name: "Second"); draft.dialogue?.speakers.append(second)
    draft.dialogue?.turns[0].fishDirection = direction("first character")
    draft.dialogue?.assignSpeaker(firstID, to: lineID)
    XCTAssertNotNil(draft.dialogue?.turns[0].fishDirection)
    draft.dialogue?.assignSpeaker(second.id, to: lineID)
    XCTAssertNil(draft.dialogue?.turns[0].fishDirection)
  }

  func testQwenPreservesStoredFishControlsWithoutEmittingOrValidatingThem() throws {
    var draft = draft(); draft.engine = .qwen3TTS; draft.referenceMode = .customVoice
    draft.fishDirection = direction("[Fish-only stored setting]")
    XCTAssertNil(try draft.request().voiceDirection)
    XCTAssertNotNil(draft.fishDirection)
    draft.startDialogue()
    XCTAssertNil(try draft.request().turns?[0].voiceDirection)
    XCTAssertNotNil(draft.dialogue?.speakers[0].fishDirection)
    draft.engine = .fishS2Pro; draft.referenceMode = .synthetic; draft.usesDialogue = false
    XCTAssertThrowsError(try draft.request())
  }
}
