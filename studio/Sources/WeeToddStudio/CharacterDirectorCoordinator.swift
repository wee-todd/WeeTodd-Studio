import AppKit
import StudioCore
import SwiftUI

private struct CharacterDirectorContextIdentity: Codable {
  var description: String
  var definition: CharacterSheetDefinition?
}

/// Owns the standalone Character Director sessions and their windows.
/// Documents live outside movie storage and never borrow `StudioStore.imageDraft`.
@MainActor final class CharacterDirectorCoordinator: NSObject, NSWindowDelegate {
  private unowned let store: StudioStore
  private let storage: CharacterSheetDocumentStore
  private var sessions: [UUID: CharacterSheetSessionController] = [:]
  private var windows: [UUID: NSWindow] = [:]
  var recentDocuments: [CharacterSheetDocument] { storage.documents() }

  init(store: StudioStore) {
    self.store = store
    storage = CharacterSheetDocumentStore(root: store.dataDirectory.appendingPathComponent("Character Director"))
    super.init()
  }

  func session(context: ReferenceSheetContext? = nil) -> CharacterSheetSessionController {
    let requestedSignature = context.flatMap(contextSignature)
    if let key = context?.subjectKey,
       let existing = sessions.values.first(where: {
         $0.document.subjectKey == key && contextMatches($0.document, context: context!, signature: requestedSignature)
       }) { return existing }

    let persisted = context.flatMap { requested in
      storage.documents().first {
        $0.subjectKey == requested.subjectKey && contextMatches($0, context: requested, signature: requestedSignature)
      }
    }
    var document = persisted ?? CharacterSheetDocument(title: context?.name ?? "New character")
    if persisted == nil, let context {
      document.subjectKey = context.subjectKey
      document.originalDescription = context.description
      if let definition = context.characterDefinition { document.definition = definition }
      var sheetContext = context
      sheetContext.template = .characterSheet
      sheetContext.characterDefinition = document.definition
      sheetContext.apply(to: &document.draft)
      try? storage.save(document)
      if let requestedSignature { try? saveContextSignature(requestedSignature, documentID: document.id) }
    } else if let context, let requestedSignature, loadContextSignature(documentID: document.id) == nil,
              document.originalDescription == context.description {
      // Adopt legacy documents only when their captured text still matches, then make future matching exact.
      try? saveContextSignature(requestedSignature, documentID: document.id)
    }
    return owner(for: document)
  }

  func session(documentID: UUID) -> CharacterSheetSessionController? {
    if let existing = sessions[documentID] { return existing }
    do { return owner(for: try storage.load(id: documentID)) }
    catch {
      store.error = "Character Director document \(documentID.uuidString) could not be opened: \(error.localizedDescription)"
      return nil
    }
  }

  func open(documentID: UUID? = nil) {
    let controller: CharacterSheetSessionController
    if let documentID {
      guard let restored = session(documentID: documentID) else { return }
      controller = restored
    } else { controller = session() }
    if let existing = windows[controller.id] {
      existing.makeKeyAndOrderFront(nil)
      NSApp.activate(ignoringOtherApps: true)
      return
    }
    let content = CharacterDirectorWindow(controller: controller).environmentObject(store)
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1480, height: 920),
      styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
    window.identifier = NSUserInterfaceItemIdentifier(controller.id.uuidString)
    window.title = "Character Director · \(controller.document.title)"
    window.contentMinSize = NSSize(width: 1040, height: 700)
    window.contentView = NSHostingView(rootView: content)
    window.isReleasedWhenClosed = false
    window.delegate = self
    windows[controller.id] = window
    window.center()
    window.makeKeyAndOrderFront(nil)
    NSApp.activate(ignoringOtherApps: true)
  }

  func windowShouldClose(_ sender: NSWindow) -> Bool {
    guard let raw = sender.identifier?.rawValue, let id = UUID(uuidString: raw),
          let controller = sessions[id], controller.running else { return true }
    let alert = NSAlert()
    alert.messageText = "A Character Director job is still running."
    alert.informativeText = "Keep it running in the background, or cancel the job before closing this window."
    alert.alertStyle = .warning
    alert.addButton(withTitle: "Keep Running")
    alert.addButton(withTitle: "Cancel Job")
    if alert.runModal() == .alertSecondButtonReturn { controller.cancel() }
    return true
  }

  func windowWillClose(_ notification: Notification) {
    guard let window = notification.object as? NSWindow,
          let raw = window.identifier?.rawValue, let id = UUID(uuidString: raw) else { return }
    windows.removeValue(forKey: id)
  }

  private func owner(for document: CharacterSheetDocument) -> CharacterSheetSessionController {
    if let existing = sessions[document.id] { return existing }
    let controller = CharacterSheetSessionController(document: document, store: store, storage: storage)
    sessions[document.id] = controller
    return controller
  }

  private func contextSignature(_ context: ReferenceSheetContext) -> String? {
    try? CharacterArtifactHash.value(CharacterDirectorContextIdentity(description: context.description,
      definition: context.characterDefinition))
  }

  private func contextMatches(_ document: CharacterSheetDocument, context: ReferenceSheetContext,
                              signature: String?) -> Bool {
    guard document.originalDescription == context.description, let signature else { return false }
    if let captured = loadContextSignature(documentID: document.id) { return captured == signature }
    // A legacy document has no captured identity. Reuse only when the supplied canonical
    // definition still equals its saved definition; ambiguity creates a new document.
    return context.characterDefinition == nil || context.characterDefinition == document.definition
  }

  private func signatureURL(documentID: UUID) -> URL {
    storage.directory(id: documentID).appendingPathComponent("subject-context.sha256")
  }
  private func loadContextSignature(documentID: UUID) -> String? {
    try? String(contentsOf: signatureURL(documentID: documentID), encoding: .utf8)
      .trimmingCharacters(in: .whitespacesAndNewlines)
  }
  private func saveContextSignature(_ signature: String, documentID: UUID) throws {
    let url = signatureURL(documentID: documentID)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data((signature + "\n").utf8).write(to: url, options: .atomic)
  }
}

/// Embeds the same document-scoped editor in a movie reference workflow.
struct CharacterSheetEmbeddedHost: View {
  @EnvironmentObject private var store: StudioStore
  let context: ReferenceSheetContext
  let onUse: (MediaAsset) async -> Bool

  var body: some View {
    let controller = store.characterDirector.session(context: context)
    CharacterDirectorWindow(controller: controller, embedded: true)
      .environmentObject(store)
      .onAppear {
        controller.onUse = { asset in
          guard let candidate = asset.generation?.referenceSheet,
                candidate.subjectKey == context.subjectKey,
                candidate.characterDefinition == controller.document.definition else {
            controller.error = "This candidate was generated from an older character definition. Keep it as a saved take or generate a new candidate from the current fields."
            return false
          }
          return await onUse(asset)
        }
      }
      .onDisappear { controller.onUse = nil }
  }
}
