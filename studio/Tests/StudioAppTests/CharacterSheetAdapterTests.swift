import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class CharacterSheetAdapterTests: XCTestCase {
  @MainActor private func fixture(names: [String]) async throws -> StudioStore {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let connection = DrawThingsConnection(id: "local", name: "Draw Things Local", selfHostedConfirmed: true)
    store.drawThingsConnections = [connection]
    await store.drawThingsDiscovery.load(connection) {
      ["models": [["id": "flux", "name": "FLUX", "family": "flux"]],
       "capabilities": ["flux": ["operations": ["image": [:]]]],
       "loras": names.enumerated().map { index, name in
         ["id": "lora-\(index)", "name": name, "family": "flux", "compatibleModelIDs": ["flux"]] as [String: Any]
       }]
    }
    var context = ReferenceSheetContext(subjectKey: "actor", name: "Actor", kind: .character,
      description: "Green eyes, curly black hair, embroidered red robe.")
    var definition = CharacterSheetDefinition.newDraft()
    definition.appearance.setText("identity.species", "human")
    definition.appearance.setText("identity.type", "human")
    context.characterDefinition = definition
    var draft = store.makeReferenceImageDraft(context, previousDraft: nil)
    draft.modelID = "flux"; store.imageDraft = draft
    return store
  }
  @MainActor func testLeavingFourPanelTemplateRemovesOnlyItsAdapterAndCanReturn() async throws {
    let store = try await fixture(names: ["4-panel turnaround", "Watercolor"])
    store.configureCharacterSheetAdapter()
    var draft = try XCTUnwrap(store.imageDraft)
    draft.loras.append(DrawThingsLoRA(modelID: "lora-1", weight: 0.4))
    var context = try XCTUnwrap(draft.referenceSheet); context.template = .portrait
    context.apply(to: &draft); store.configureCharacterSheetAdapter(&draft)
    XCTAssertNil(draft.characterSheetLoRAID)
    XCTAssertEqual(draft.loras.map(\.modelID), ["lora-1"])
    context.template = .characterSheet; context.apply(to: &draft)
    store.configureCharacterSheetAdapter(&draft)
    XCTAssertEqual(draft.loras.map(\.modelID), ["lora-1", "lora-0"])
    XCTAssertEqual(draft.loras.first?.weight, 0.4)
  }
  @MainActor func testInstalledKreaFourViewPairDefaultsOnColdCatalogAndDoesNotReplaceSavedModel() async throws {
    let store = try await fixture(names: [])
    let local = try XCTUnwrap(store.drawThingsConnections.first)
    store.drawThingsDiscovery.invalidate(local.id)
    let context = ReferenceSheetContext(subjectKey: "new", name: "Nell", kind: .character, description: "Freckled elf in blue wool.")
    store.imageDraft = store.makeReferenceImageDraft(context, previousDraft: nil)
    XCTAssertEqual(store.imageDraft?.modelID, "")
    await store.drawThingsDiscovery.load(local) { [
      "models": [["id": "krea", "name": "Krea 2 Turbo", "family": "krea2"], ["id": "other", "name": "Other", "family": "flux"]],
      "capabilities": ["krea": ["operations": ["image": [:]]], "other": ["operations": ["image": [:]]]],
      "loras": [["id": "installed-four-view.ckpt", "name": "Krea2_Character_Design_4-View_V1 (Krea 2)", "family": "krea2", "compatibleModelIDs": ["krea"]]]
    ] }
    store.configureCharacterSheetAdapter()
    XCTAssertEqual(store.imageDraft?.modelID, "krea")
    XCTAssertEqual(store.imageDraft?.steps, 8)
    XCTAssertEqual(store.imageDraft?.guidance, 1)
    XCTAssertEqual(store.imageDraft?.loras.map(\.modelID), ["installed-four-view.ckpt"])
    store.imageDraft?.modelID = "other"
    store.configureCharacterSheetAdapter()
    XCTAssertEqual(store.imageDraft?.modelID, "other")
    XCTAssertNotNil(store.characterSheetIssue(try XCTUnwrap(store.imageDraft)))
  }

  @MainActor func testKnownInstalledAdapterWinsOverOtherFourPanelVariants() async throws {
    let store = try await fixture(names: ["Other 4-panel adapter", "Krea2_Character_Design_4-View_V1 (Krea 2)"])
    store.configureCharacterSheetAdapter()
    XCTAssertEqual(store.imageDraft?.characterSheetLoRAID, "lora-1")
    XCTAssertEqual(store.imageDraft?.loras.map(\.modelID), ["lora-1"])
  }

  @MainActor func testCatalogArrivalSelectsAdapterOnceAndPreservesDisabledChoice() async throws {
    let store = try await fixture(names: ["4 Panel turnaround"])
    store.configureCharacterSheetAdapter()
    store.configureCharacterSheetAdapter()
    XCTAssertEqual(store.imageDraft?.loras.count, 1)
    XCTAssertEqual(store.imageDraft?.characterSheetLoRAID, "lora-0")
    XCTAssertNil(store.characterSheetIssue(try XCTUnwrap(store.imageDraft)))
    store.imageDraft?.loras[0].enabled = false
    store.configureCharacterSheetAdapter()
    XCTAssertFalse(try XCTUnwrap(store.imageDraft?.loras[0].isEnabled))
    XCTAssertNotNil(store.characterSheetIssue(try XCTUnwrap(store.imageDraft)))
    store.imageDraft?.loras[0].enabled = true; store.imageDraft?.loras[0].weight = 0
    XCTAssertThrowsError(try store.imageDraft?.request(id: "zero-weight"))
  }
  @MainActor func testAmbiguousOrUnrecognizedNamesRequireExplicitCompatibleSelection() async throws {
    for names in [["4-panel A", "four_panel B"], ["Custom turnaround adapter"]] {
      let store = try await fixture(names: names)
      store.configureCharacterSheetAdapter()
      XCTAssertTrue(try XCTUnwrap(store.imageDraft?.loras.isEmpty))
      XCTAssertNotNil(store.characterSheetIssue(try XCTUnwrap(store.imageDraft)))
      store.selectCharacterSheetAdapter("missing")
      XCTAssertNil(store.imageDraft?.characterSheetLoRAID)
      store.selectCharacterSheetAdapter("lora-0")
      XCTAssertNil(store.characterSheetIssue(try XCTUnwrap(store.imageDraft)))
      let payload = try await store.imagePayload(try XCTUnwrap(store.imageDraft), connection: store.drawThingsConnections.first)
      let request = try XCTUnwrap(payload["drawThingsRequest"] as? [String: Any])
      XCTAssertEqual((request["loras"] as? [[String: Any]])?.first?["modelID"] as? String, "lora-0")
    }
  }
  @MainActor func testModelOrPromptChangeInvalidatesRequiredAdapterContract() async throws {
    let store = try await fixture(names: ["4-panel turnaround"])
    store.configureCharacterSheetAdapter()
    var draft = try XCTUnwrap(store.imageDraft)
    draft.modelID = "incompatible"
    XCTAssertNotNil(store.characterSheetIssue(draft))
    draft = try XCTUnwrap(store.imageDraft); draft.prompt = "A portrait instead"
    XCTAssertThrowsError(try draft.request(id: "wrong-prefix"))
    draft = try XCTUnwrap(store.imageDraft); draft.selectProvider(.nativeMLX)
    XCTAssertNotNil(store.characterSheetIssue(draft))
  }
  @MainActor func testCatalogDefaultDoesNotMutateOrdinaryWorkspaceAndSheetDraftRestores() async throws {
    let store = try await fixture(names: ["4-panel turnaround"])
    store.configureCharacterSheetAdapter()
    let sheet = try XCTUnwrap(store.imageDraft)
    store.imageWorkspaceLibrary.record(sheet, preview: nil)
    let restored = store.makeReferenceImageDraft(try XCTUnwrap(sheet.referenceSheet), previousDraft: nil)
    XCTAssertEqual(restored, sheet)
    var ordinary = DrawThingsImageDraft(destination: sheet.destination)
    ordinary.profileID = "local"; ordinary.modelID = "flux"; ordinary.prompt = "Keep my painting"
    store.imageDraft = ordinary
    store.configureCharacterSheetAdapter()
    XCTAssertEqual(store.imageDraft, ordinary)
  }
}
