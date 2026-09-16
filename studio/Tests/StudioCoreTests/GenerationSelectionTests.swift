import XCTest
@testable import StudioCore

final class GenerationSelectionTests: XCTestCase {
  func testMotionReviewShowsLTXSourceVideoAndEffectiveTaskWithoutChangingSavedTask() {
    for engine in [Engine.ltx23, .ltx25] {
      var clip = Clip(engine: engine)
      clip.generationSelection = GenerationSelection(task: "t2v")
      clip.continuity = ClipContinuity(mode: "motion")
      XCTAssertEqual(clip.reviewMediaCount, 1)
      XCTAssertEqual(clip.displayTask, "Video extension")
      XCTAssertEqual(clip.inferredTask, "t2v")
      clip.continuity = nil
      XCTAssertEqual(clip.reviewMediaCount, 0)
      XCTAssertEqual(clip.displayTask, "Text to video")
    }
    var h3 = Clip(engine: .h3)
    h3.continuity = ClipContinuity(mode: "motion")
    XCTAssertEqual(h3.reviewMediaCount, 0) // H3 consumes native context, not a video attachment.
  }

  func testReviewAttachmentsExcludeDisabledLoRAsAndReplacedFirstFrame() {
    var clip = Clip(engine: .ltx25)
    let first = Attachment(assetID: UUID(), role: .first)
    let last = Attachment(assetID: UUID(), role: .last)
    var lora = Attachment(assetID: UUID(), role: .lora)
    lora.enabled = false
    clip.attachments = [first, last, lora]
    clip.continuity = ClipContinuity(mode: "frame")
    XCTAssertEqual(clip.reviewAttachments.map(\.id), [last.id])
    XCTAssertEqual(clip.reviewMediaCount, 2)
    XCTAssertEqual(clip.reviewLoRACount, 0)
    clip.continuity = nil
    XCTAssertEqual(clip.reviewAttachments.map(\.id), [first.id, last.id])
    XCTAssertEqual(clip.attachments.count, 3)
  }

  func testAutomaticModelRecoveryPreservesTaskControlsAndMedia() {
    var clip = Clip(engine: .h3)
    clip.profileID = "missing.json"
    clip.attachments = [Attachment(assetID: UUID(), role: .first)]
    clip.selectAutomaticModelComponents()
    XCTAssertEqual(clip.profileID, "auto")
    XCTAssertEqual(clip.inferredTask, "fflf")
    XCTAssertNotNil(clip.generationSelection)
    clip.profileID = "incompatible.json"
    clip.generationSelection?.steps = 19
    clip.selectAutomaticModelComponents()
    XCTAssertEqual(clip.generationSelection?.steps, 19)
    XCTAssertEqual(clip.attachments.count, 1)
  }

  func testProviderRoundTripRestoresLastLocalModelWithoutChangingItsSettings() throws {
    var clip = Clip(engine: .ltx25)
    clip.profileID = "chosen-ltx25.json"
    clip.generationSelection = GenerationSelection(task: "fflf", preset: .custom)
    clip.generationSelection?.steps = 8
    clip.selectGenerationProvider(.drawThings)
    XCTAssertEqual(clip.generationProvider, .drawThings)
    clip = try JSONDecoder().decode(Clip.self, from: JSONEncoder().encode(clip))
    clip.selectGenerationProvider(.local)
    XCTAssertEqual(clip.engine, .ltx25)
    XCTAssertEqual(clip.profileID, "chosen-ltx25.json")
    XCTAssertEqual(clip.generationSelection?.steps, 8)
    XCTAssertEqual(clip.inferredTask, "fflf")
  }

  func testNewLocalModelUsesAutomaticComponentsAndExistingMediaTask() {
    var clip = Clip(engine: .drawThings)
    clip.attachments = [Attachment(assetID: UUID(), role: .first),
                        Attachment(assetID: UUID(), role: .last)]
    clip.selectLocalModel(.h3)
    XCTAssertEqual(clip.engine, .h3)
    XCTAssertEqual(clip.profileID, "auto")
    XCTAssertEqual(clip.inferredTask, "fflf")
    XCTAssertEqual(clip.generationSelection?.preset, .balanced)
    XCTAssertEqual(clip.attachments.count, 2)
  }

  func testTaskAndPresetChangesPreserveChosenModelAndMedia() {
    var clip = Clip(engine: .h3)
    clip.profileID = "/recipes/chosen-model.json"
    clip.attachments = [Attachment(assetID: UUID(), role: .first)]
    let attachments = clip.attachments
    clip.selectGenerationPreset(.speed)
    XCTAssertEqual(clip.profileID, "/recipes/chosen-model.json")
    clip.generationSelection?.steps = 12
    clip.selectGenerationTask("fflf")
    XCTAssertEqual(clip.profileID, "/recipes/chosen-model.json")
    XCTAssertEqual(clip.attachments, attachments)
    XCTAssertEqual(clip.generationSelection?.task, "fflf")
    XCTAssertEqual(clip.generationSelection?.preset, .speed)
    XCTAssertEqual(clip.generationSelection?.steps, 12)
  }

  func testBackendSwitchRestoresNativeModelAndParametersAfterRoundTrip() throws {
    var clip = Clip(engine: .h3)
    clip.profileID = "/recipes/h3.json"
    clip.generationSelection = GenerationSelection(task: "fflf", preset: .custom)
    clip.generationSelection?.steps = 19
    clip.selectGenerationEngine(.drawThings)
    XCTAssertNil(clip.generationSelection)
    clip = try JSONDecoder().decode(Clip.self, from: JSONEncoder().encode(clip))
    clip.selectGenerationEngine(.ltx25)
    XCTAssertEqual(clip.profileID, "auto")
    clip.selectGenerationEngine(.h3)
    XCTAssertEqual(clip.profileID, "/recipes/h3.json")
    XCTAssertEqual(clip.generationSelection?.steps, 19)
    XCTAssertEqual(clip.generationSelection?.task, "fflf")
  }
  func testLegacyClipPreservesCustomRecipe() throws {
    var clip = Clip(engine: .h3)
    clip.profileID = "/recipes/exact.json"
    var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(clip)) as! [String: Any]
    object.removeValue(forKey: "generationSelection")
    let decoded = try JSONDecoder().decode(Clip.self, from: JSONSerialization.data(withJSONObject: object))
    XCTAssertNil(decoded.generationSelection)
    XCTAssertEqual(decoded.profileID, clip.profileID)
  }
  func testExplicitTaskPreservesAttachmentsAndReset() throws {
    var clip = Clip(engine: .h3)
    clip.attachments = [Attachment(assetID: UUID(), role: .reference)]
    clip.generationSelection = GenerationSelection(task: "t2v", preset: .balanced)
    XCTAssertEqual(clip.inferredTask, "t2v")
    XCTAssertEqual(clip.attachments.count, 1)
    let fingerprint = clip.generationFingerprint
    clip.generationSelection?.steps = 27
    XCTAssertTrue(clip.generationSelection!.isModified)
    XCTAssertNotEqual(fingerprint, clip.generationFingerprint)
    clip.generationSelection?.resetOverrides()
    XCTAssertEqual(fingerprint, clip.generationFingerprint)
    let decoded = try JSONDecoder().decode(Clip.self, from: JSONEncoder().encode(clip))
    XCTAssertEqual(decoded.generationSelection, clip.generationSelection)
  }
}

extension GenerationSelectionTests {
  func testAccelerationDefaultsAndOverrideRoundTrip() throws {
    var defaults = AccelerationSettings()
    XCTAssertEqual(defaults.h3MemoryPolicy, "automatic")
    XCTAssertEqual(defaults.h3ProjectionBackend, "auto")
    defaults.h3MemoryPolicy = "paged"
    defaults.h3ProjectionBackend = "mlx"
    XCTAssertEqual(try JSONDecoder().decode(AccelerationSettings.self,
      from: JSONEncoder().encode(defaults)), defaults)
  }
}

extension GenerationSelectionTests {
  func testReferencedAssetRelinkAndLoRAMetadataInvalidateResolution() {
    var asset = MediaAsset(name: "Reference", kind: .image, path: "/first.png")
    var clip = Clip(engine: .h3)
    clip.attachments = [Attachment(assetID: asset.id, role: .first)]
    let original = GenerationSelection.assetFingerprint(for: clip, assets: [asset])
    asset.path = "/relinked.png"
    XCTAssertNotEqual(original, GenerationSelection.assetFingerprint(for: clip, assets: [asset]))
    let relinked = GenerationSelection.assetFingerprint(for: clip, assets: [asset])
    asset.loraModel = .ltx23
    XCTAssertNotEqual(relinked, GenerationSelection.assetFingerprint(for: clip, assets: [asset]))
    XCTAssertNotEqual(relinked, GenerationSelection.assetFingerprint(for: clip, assets: []))
  }
}

extension GenerationSelectionTests {
  func testLargerWorkspacePagingSelectionRoundTripAndReset() throws {
    var clip = Clip(engine: .h3)
    clip.generationSelection = GenerationSelection()
    let original = clip.generationFingerprint
    clip.generationSelection?.memoryPolicy = "pagedNormal"
    let restored = try JSONDecoder().decode(Clip.self, from: JSONEncoder().encode(clip))
    XCTAssertEqual(restored.generationSelection?.memoryPolicy, "pagedNormal")
    XCTAssertNotEqual(restored.generationFingerprint, original)
    XCTAssertEqual(AccelerationSettings.memoryPolicyLabel("pagedNormal"), "Paged · larger workspace")
    clip.generationSelection?.resetOverrides()
    XCTAssertEqual(clip.generationFingerprint, original)
  }
}
