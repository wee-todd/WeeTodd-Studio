import Foundation
import XCTest
@testable import StudioCore

final class RippleTests: XCTestCase {
  func draft() -> RippleDraft {
    var clip = Clip(name: "Original", engine: .movie)
    clip.sourcePath = "/source.mp4"; clip.sourceIn = 3.25; clip.duration = 4.8
    var draft = RippleDraft(clip: clip, frameRate: 24)
    draft.references[0].path = "/edited.png"
    return draft
  }
  func testNewDraftUsesAuthorPromptAndAdjustableBaselineStrength() throws {
    var clip = Clip(); clip.prompt = "Unrelated text-to-video shot direction"
    var value = RippleDraft(clip: clip, frameRate: 24)
    XCTAssertEqual(value.prompt, "Use the reference video for motion, timing, camera movement, composition, and unchanged scene content, while consistently propagating the visual edit established in the first frame throughout the video.")
    XCTAssertEqual(value.loraStrength, 1.35)
    value.prompt = "Custom restyle"; value.loraStrength = 0.75
    XCTAssertEqual(value.prompt, "Custom restyle"); XCTAssertEqual(value.loraStrength, 0.75)
  }
  func testOneAndNineDistinctFramesAcceptedButTenDuplicateAndRangeRejected() throws {
    var value = draft()
    try value.validate()
    value.references = (0..<9).map { RippleReference(frame: $0 * 8, path: "/\($0).png") }
    try value.validate()
    value.references.append(RippleReference(frame: 80, path: "/tenth.png"))
    XCTAssertThrowsError(try value.validate())
    value.references.removeLast(); value.references[8].frame = value.references[0].frame
    XCTAssertThrowsError(try value.validate())
    value.references[8].frame = value.frameCount
    XCTAssertThrowsError(try value.validate())
    value.references[8].frame = -1
    XCTAssertThrowsError(try value.validate())
  }
  func testFirstFrameIsRequired() {
    var value = draft(); value.references[0].frame = 1
    XCTAssertThrowsError(try value.validate())
  }
  func testInspectionOmitsIncompleteReferencePlaceholders() throws {
    var value = draft(); value.references[0].path = ""
    XCTAssertEqual((try value.bridgeObject(requireReferences: false)["references"] as? [[String: Any]])?.count, 0)
  }
  func testFrameCountCoversFractionalEditorialFrameWithoutChangingTrim() {
    let value = draft()
    XCTAssertEqual(value.frameCount, 116)
    XCTAssertEqual(value.duration, 4.8)
    XCTAssertEqual(value.sourceIn, 3.25)
  }
  func testRequestUsesRelativeFramesOnlyAndPreservesAudioByDefault() throws {
    var value = draft(); value.references.append(RippleReference(frame: 15, path: "/other.png"))
    let body = try value.bridgeObject()
    XCTAssertEqual(body["source_start"] as? Double, 3.25)
    XCTAssertEqual(body["audio_policy"] as? String, "preserve")
    XCTAssertEqual((body["references"] as? [[String: Any]])?.last?["frame"] as? Int, 15)
    XCTAssertNil(body["rippleAdapterPath"])
    XCTAssertNil(body["rippleProfileID"])
    XCTAssertNil((body["references"] as? [[String: Any]])?.first?["originalPath"])
    value.audioPolicy = .silent
    XCTAssertEqual(try value.bridgeObject()["audio_policy"] as? String, "silent")
  }
  func testEmptyPromptIsValidAndIncompleteDraftCanInspect() throws {
    var value = draft(); value.prompt = ""; try value.validate()
    value.references[0].path = ""
    XCTAssertThrowsError(try value.validate())
    XCTAssertNoThrow(try value.bridgeObject(requireReferences: false))
  }
  func testInvalidStrengthAndNonFiniteOrExcessiveSettingsRejected() {
    for strength in [-0.1, 1.1, Double.nan] {
      var value = draft(); value.references[0].strength = strength
      XCTAssertThrowsError(try value.validate())
    }
    for strength in [0, -1, 3.1, Double.infinity] {
      var value = draft(); value.loraStrength = strength
      XCTAssertThrowsError(try value.validate())
    }
    var value = draft(); value.width = 1921; XCTAssertThrowsError(try value.validate())
    value = draft(); value.frameRate = 61; XCTAssertThrowsError(try value.validate())
    value = draft(); value.duration = .nan; XCTAssertThrowsError(try value.validate())
  }
  func testDraftAndTakePersistAndLegacyClipNeedsNeither() throws {
    var clip = Clip(engine: .movie); clip.sourcePath = "/source.mp4"
    let legacy = try JSONEncoder().encode(clip)
    XCTAssertNil(try JSONDecoder().decode(Clip.self, from: legacy).rippleDraft)
    clip.rippleDraft = draft()
    clip.rippleTakes = [RippleTake(draft: draft(), path: "/take.mp4", receiptPath: "/receipt.json", artifactsDirectory: "/artifacts", hasAudio: false)]
    XCTAssertEqual(try JSONDecoder().decode(Clip.self, from: JSONEncoder().encode(clip)), clip)
  }
  func testPortablePathsMapEverySourcePairTakeAndReceipt() throws {
    var project = StudioProject(); var clip = Clip()
    var value = draft(); value.references[0].originalPath = "/original.png"
    clip.rippleDraft = value
    clip.rippleTakes = [RippleTake(draft: value, path: "/take.mp4", receiptPath: "/receipt.json", artifactsDirectory: "/artifacts", hasAudio: true)]
    project.clips = [clip]
    ProjectStorage.mapPaths(&project) { $0.isEmpty ? $0 : "mapped" + $0 }
    let updated = try XCTUnwrap(project.clips[0].rippleDraft)
    XCTAssertEqual(updated.sourcePath, "mapped/source.mp4")
    XCTAssertEqual(updated.references[0].originalPath, "mapped/original.png")
    XCTAssertEqual(updated.references[0].path, "mapped/edited.png")
    XCTAssertEqual(project.clips[0].rippleTakes?[0].draft, updated)
    XCTAssertEqual(project.clips[0].rippleTakes?[0].receiptPath, "mapped/receipt.json")
    XCTAssertEqual(project.clips[0].rippleTakes?[0].artifactsDirectory, "mapped/artifacts")
  }
  func testDraftDoesNotInvalidateOrdinaryGenerationFingerprint() {
    var clip = Clip(); let fingerprint = clip.generationFingerprint
    clip.rippleDraft = draft()
    XCTAssertEqual(clip.generationFingerprint, fingerprint)
  }
}
