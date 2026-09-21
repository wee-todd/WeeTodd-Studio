import XCTest
import StudioCore
@testable import WeeToddStudio

@MainActor final class CharacterCatalogTests: XCTestCase {
  private func inventory(detailIDs: [String]) -> [String: Any] {
    ["models": [["id": "krea", "name": "Krea 2 Turbo"],
      ["id": "klein", "name": "FLUX.2 klein 9B"]],
     "capabilities": ["krea": ["operations": ["image": [:]]],
       "klein": ["operations": ["image": [:]]]],
     "loras": [["id": "krea2_character_design_4view_v1.ckpt", "name": "Four view",
       "compatibleModelIDs": ["krea"]]] + detailIDs.map {
         ["id": $0, "name": $0, "compatibleModelIDs": ["klein"]] as [String: Any]
       }]
  }

  func testDetailLoRADiscoveryAcceptsBothSpellingsAndPreservesInstalledID() async throws {
    for detailID in ["highresolution9b_lora_f16.ckpt", "hichresolution9b_lora_f16.ckpt"] {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: root) }
      let catalog = inventory(detailIDs: [detailID])
      let store = StudioStore(dataDirectory: root, restoreSession: false,
        invocation: { _, _, _, _ in catalog })
      store.drawThingsConnections = [.init(id: "local")]
      let controller = CharacterSheetSessionController(document: .init(), store: store,
        storage: .init(root: root))
      await controller.refreshCatalog()
      XCTAssertNil(controller.error)
      XCTAssertEqual(controller.document.refinement.detailLoRAID, detailID)
      try await controller.validateRecipeDependencies()
    }
  }

  func testAmbiguousDetailLoRAsRequireSelectionWithoutRewritingAnExplicitID() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let ids = ["highresolution9b_lora_f16.ckpt", "hichresolution9b_lora_f16.ckpt"]
    let catalog = inventory(detailIDs: ids)
    let store = StudioStore(dataDirectory: root, restoreSession: false,
      invocation: { _, _, _, _ in catalog })
    store.drawThingsConnections = [.init(id: "local")]
    let controller = CharacterSheetSessionController(document: .init(), store: store,
      storage: .init(root: root))
    await controller.refreshCatalog()
    XCTAssertEqual(controller.document.refinement.detailLoRAID, "")
    controller.document.refinement.detailLoRAID = ids[0]
    await controller.refreshCatalog()
    XCTAssertEqual(controller.document.refinement.detailLoRAID, ids[0])
    try await controller.validateRecipeDependencies()
  }

  func testEmptyDiscoveryExplainsModelBrowsingAndGenerationPreservesFailure() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    var commands = [String]()
    let store = StudioStore(dataDirectory: root, restoreSession: false,
      invocation: { command, _, _, _ in
        commands.append(command)
        return ["models": [], "loras": []]
      })
    store.drawThingsConnections = [.init(id: "local")]
    let controller = CharacterSheetSessionController(document: .init(), store: store,
      storage: .init(root: root))
    await controller.refreshCatalog()
    let discoveryError = controller.error
    XCTAssertTrue(discoveryError?.contains("Enable Model Browsing") ?? false)
    controller.launch { try await controller.generateInitial() }
    await controller.workTask?.value
    XCTAssertEqual(controller.error, discoveryError)
    XCTAssertEqual(commands, ["dt-discover"])
  }
}
