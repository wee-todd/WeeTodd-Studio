import AppKit
import ImageIO
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

/// A frame-scoped image workspace; generated candidates are attached only on explicit selection.
struct RippleReferenceGenerator: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  let clipID: UUID
  let referenceID: UUID
  @State private var lease: ReferenceWorkspaceLease?
  @State private var context: RippleImageContext?
  @State private var started = false
  @State private var failure: String?

  var body: some View {
    Group {
      if let context, started {
        ImageGenerationEditor(onClose: { dismiss() }, onUseReference: { asset in
          adopt(asset.path, context: context)
        }, referenceTools: AnyView(referenceTools(context)), backLabel: "Back to Ripple Director")
      } else {
        VStack(spacing: 16) {
          if let failure { Text(failure).textSelection(.enabled); Button("Close") { dismiss() } }
          else { ProgressView("Extracting source frame…") }
        }
      }
    }
    .frame(minWidth: 980, idealWidth: 1480, minHeight: 680, idealHeight: 960)
    .background(ReferenceEditorWindow())
    .interactiveDismissDisabled(store.operationBusy)
    .task { await begin() }
    .onDisappear { _ = lease?.restore(store: store) }
    .alert("Ripple reference", isPresented: Binding(get: { started && failure != nil }, set: { if !$0 { failure = nil } })) {
      Button("OK") { failure = nil }
    } message: { Text(failure ?? "") }
  }

  private func referenceTools(_ context: RippleImageContext) -> some View {
    HStack(spacing: 14) {
      WorkspaceImage(path: context.originalPath, fill: false).frame(width: 90, height: 60)
      VStack(alignment: .leading, spacing: 4) {
        Text("Reference frame \(context.frame) · \(Double(context.frame) / context.frameRate, specifier: "%.3f") s").font(.headline)
        Text("Restyle the captured frame with Draw Things, then choose Use as reference.")
          .font(.caption).foregroundStyle(.secondary)
      }
      Spacer()
      Menu("Existing image asset") {
        ForEach(store.allAssets.filter { $0.kind == .image }) { asset in
          Button(asset.name) { adopt(asset.path, context: context) }
        }
      }.disabled(!store.allAssets.contains { $0.kind == .image })
      Button("Import edited frame…") {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { adopt(url.path, context: context) }
      }
    }.padding(12).disabled(store.operationBusy)
  }

  private func begin() async {
    guard !started, context == nil else { return }
    let session = store.documentSessionID
    guard let clip = store.project.clips.first(where: { $0.id == clipID }),
      let reference = clip.rippleDraft?.references.first(where: { $0.id == referenceID }) else {
      failure = "This reference was removed. Return to the Ripple Director."; return
    }
    if reference.originalPath.isEmpty { await store.extractRippleFrame(referenceID: referenceID) }
    guard session == store.documentSessionID,
      let current = store.project.clips.first(where: { $0.id == clipID }), let ripple = current.rippleDraft,
      let ref = ripple.references.first(where: { $0.id == referenceID }),
      !ref.originalPath.isEmpty else {
      failure = "The source frame could not be extracted, or the movie changed. Check the source clip and try again."; return
    }
    let captured = RippleImageContext(clipID: clipID, draft: ripple, reference: ref)
    guard captured.matches(current) else {
      failure = "The source clip changed. Reopen Ripple using the current source."; return
    }
    let savedLease = ReferenceWorkspaceLease(store: store)
    lease = savedLease
    let draft = store.makeRippleImageDraft(captured, width: ripple.width, height: ripple.height,
                                           previousDraft: savedLease.previousDraft)
    store.referenceSheetOpen = true
    store.restoringImageWorkspace = true
    store.imageDraft = draft; store.imageEstimate = nil
    store.imagePreviewPath = store.imageWorkspaceLibrary.sessions[draft.storageKey]?.draft == draft
      ? store.imageWorkspaceLibrary.sessions[draft.storageKey]?.previewPath : nil
    store.restoringImageWorkspace = false; store.persistImageWorkspace()
    context = captured; started = true
  }

  private func adopt(_ path: String, context: RippleImageContext) {
    do {
      guard let lease else { throw StudioError.invalid("Reopen the reference editor.") }
      try lease.validate(store: store)
      try store.adoptRippleImage(path, context: context)
      dismiss()
    } catch { failure = error.localizedDescription }
  }
}

@MainActor extension StudioStore {
  func makeRippleImageDraft(_ context: RippleImageContext, width: Int, height: Int,
                            previousDraft: DrawThingsImageDraft?) -> DrawThingsImageDraft {
    let destination = ImageAssetDestination(scope: .clip, projectID: project.id, owner: context.clipID)
    var draft = previousDraft ?? DrawThingsImageDraft(destination: destination)
    draft.destination = destination; draft.referenceSheet = nil; draft.rippleReference = context
    draft.name = "Ripple · frame \(context.frame)"
    draft.prompt = ""; draft.moodboard = []; draft.canvas = ImageWorkspaceInput(path: context.originalPath)
    draft.width = width; draft.height = height
    if let saved = imageWorkspaceLibrary.sessions[draft.storageKey], saved.draft.rippleReference == context {
      return saved.draft
    }
    if imageWorkspaceLibrary.referenceProvider == .nativeMLX {
      draft.selectProvider(.nativeMLX); draft.nativeImage = imageWorkspaceLibrary.referenceNativeImage ?? NativeImageSettings()
      draft.width = width; draft.height = height
      return draft
    }
    if let preferred = imageWorkspaceLibrary.referenceConnectionID {
      draft.profileID = drawThingsConnections.contains { $0.id == preferred } ? preferred : ""
    } else if !drawThingsConnections.contains(where: { $0.id == draft.profileID }) {
      draft.profileID = ""
    }
    return draft
  }

  func adoptRippleImage(_ path: String, context: RippleImageContext) throws {
    guard let index = project.clips.firstIndex(where: { $0.id == context.clipID }),
      context.matches(project.clips[index]),
      imageDraft?.rippleReference == context,
      let refIndex = project.clips[index].rippleDraft?.references.firstIndex(where: { $0.id == context.referenceID }) else {
      throw StudioError.invalid("The movie, source frame or reference changed. The image remains available; reopen this frame before attaching it.")
    }
    let url = URL(fileURLWithPath: path)
    guard let image = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(image) > 0 else {
      throw StudioError.invalid("Choose a readable edited image.")
    }
    change { $0.clips[index].rippleDraft?.references[refIndex].path = path }
    notice = "Edited image assigned to Ripple frame \(context.frame)."
  }
}
