import AppKit
import StudioCore
import SwiftUI

struct VoiceScriptEditor: View {
  @EnvironmentObject var store: StudioStore
  @Binding var text: String
  var engine: VoiceEngine
  var contextID: String
  var customVoice = false
  @State private var selection = NSRange(location: 0, length: 0)
  @State private var custom = ""
  @State private var showCustom = false
  @State private var showAutoTag = false
  @State private var error: String?
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      if engine == .fishS2Pro {
        HStack {
          Menu("Add tag") {
            ForEach(FishVoiceTags.groups, id: \.0) { group in
              Menu(group.0) { ForEach(group.1, id: \.self) { tag in Button(tag.capitalized) { insert(tag) } } }
            }
            Divider(); Button("Custom delivery…") { showCustom = true }
          }
          ForEach(["whisper", "excited", "pause", "sigh"], id: \.self) { tag in
            Button(tag.capitalized) { insert(tag) }.controlSize(.small)
          }
          Spacer()
          Button { showAutoTag = true } label: { Label("Auto Tag", systemImage: "sparkles") }
            .disabled(store.operationBusy || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        Text("Click a tag to insert before the cursor or selection. Tags affect the words that follow; you can edit or delete them in the script.")
          .font(.caption).foregroundStyle(.secondary)
      } else if customVoice {
        Text("Use the Delivery instructions to guide emotion. Fish inline tags are not supported by Qwen.").font(.caption).foregroundStyle(.secondary)
        if FishVoiceTags.containsTags(text) {
          Text("Remove the bracket tags before generating with Qwen.").font(.caption).foregroundStyle(.orange)
        }
      } else {
        Text("Qwen Base follows the delivery in your reference sample and the meaning of the script. Use an expressive sample with its transcript for emotion; inline Fish tags are not supported.")
          .font(.caption).foregroundStyle(.secondary)
        if FishVoiceTags.containsTags(text) {
          Text("This script contains bracket tags. Remove them before generating with Qwen Base.").font(.caption).foregroundStyle(.orange)
        }
      }
      VoiceTextEditor(text: $text, selection: $selection).frame(minHeight: 100)
      if let error { Text(error).font(.caption).foregroundStyle(.red) }
    }
    .popover(isPresented: $showCustom) {
      VStack(alignment: .leading, spacing: 12) {
        Text("Describe the delivery").font(.headline)
        TextField("e.g. speaking slowly, almost hesitant", text: $custom).frame(width: 320)
        Button("Insert tag") { insert(custom); if error == nil { custom = ""; showCustom = false } }.disabled(custom.isEmpty)
      }.padding()
    }
    .sheet(isPresented: $showAutoTag) {
      VoiceAutoTagView(text: $text, contextID: contextID).environmentObject(store)
    }
  }
  private func insert(_ tag: String) {
    do {
      let result = try FishVoiceTags.inserting(tag, into: text, at: selection)
      text = result.text; selection = result.cursor; error = nil
    } catch { self.error = error.localizedDescription }
  }
}

/// AppKit selection works on the app's macOS 14 minimum and survives toolbar focus.
struct VoiceTextEditor: NSViewRepresentable {
  @Binding var text: String
  @Binding var selection: NSRange
  @Environment(\.isEnabled) private var enabled
  func makeCoordinator() -> Coordinator { Coordinator(self) }
  func makeNSView(context: Context) -> NSScrollView {
    let scroll = NSScrollView(); scroll.hasVerticalScroller = true; scroll.autohidesScrollers = true
    scroll.borderType = .bezelBorder
    let view = NSTextView(); view.isRichText = false; view.importsGraphics = false; view.allowsUndo = true
    view.isAutomaticQuoteSubstitutionEnabled = false; view.isAutomaticDashSubstitutionEnabled = false
    view.isVerticallyResizable = true; view.isHorizontallyResizable = false; view.autoresizingMask = [.width]
    view.textContainer?.widthTracksTextView = true; view.textContainerInset = NSSize(width: 8, height: 8)
    view.font = .systemFont(ofSize: NSFont.systemFontSize); view.delegate = context.coordinator
    view.setAccessibilityLabel("Voice script"); scroll.documentView = view
    return scroll
  }
  func updateNSView(_ scroll: NSScrollView, context: Context) {
    context.coordinator.parent = self
    guard let view = scroll.documentView as? NSTextView else { return }
    context.coordinator.updating = true; defer { context.coordinator.updating = false }
    view.isEditable = enabled
    if view.string != text { view.string = text }
    let size = (text as NSString).length
    let start = min(selection.location, size)
    let range = NSRange(location: start, length: min(selection.length, size - start))
    if view.selectedRange() != range { view.setSelectedRange(range); view.scrollRangeToVisible(range) }
    guard !view.hasMarkedText(), let storage = view.textStorage else { return }
    let all = NSRange(location: 0, length: storage.length)
    storage.addAttribute(.foregroundColor, value: NSColor.textColor, range: all)
    if let expression = try? NSRegularExpression(pattern: #"\[[^\[\]\r\n]+\]"#) {
      for match in expression.matches(in: text, range: all) { storage.addAttribute(.foregroundColor, value: NSColor.systemTeal, range: match.range) }
    }
    view.typingAttributes = [.font: NSFont.systemFont(ofSize: NSFont.systemFontSize), .foregroundColor: NSColor.textColor]
  }
  final class Coordinator: NSObject, NSTextViewDelegate {
    var parent: VoiceTextEditor; var updating = false
    init(_ parent: VoiceTextEditor) { self.parent = parent }
    func textDidChange(_ notification: Notification) {
      guard !updating, let view = notification.object as? NSTextView else { return }
      parent.selection = view.selectedRange(); parent.text = view.string
    }
    func textViewDidChangeSelection(_ notification: Notification) {
      guard !updating, let view = notification.object as? NSTextView else { return }
      parent.selection = view.selectedRange()
    }
  }
}

struct VoiceAutoTagView: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  @Binding var text: String
  let contextID: String
  @AppStorage("qwen35PromptModelPath") private var modelPath = ""
  @State private var source = ""
  @State private var proposal: String?
  @State private var snapshot: VoiceTagContext?
  @State private var running = false
  @State private var error: String?
  @State private var showSetup = false
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack { Text("Auto Tag · Fish").font(.title2); Spacer(); Button("Done") { dismiss() }.disabled(running) }
      Text("Local Qwen3.5 suggests delivery from the script’s words. It does not listen to the reference recording. Review the result before applying.").foregroundStyle(.secondary)
      HStack {
        Text(modelPath.isEmpty ? "Set up the local assistant to continue." : LocalPromptModels.label(for: modelPath))
        Spacer(); Button("Set up assistant…") { showSetup = true }.disabled(running || store.operationBusy)
      }
      Text("Suggested script").font(.headline)
      ScrollView { Text(proposal ?? text).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(10) }
        .frame(minHeight: 160, maxHeight: 300).background(Theme.raised)
      if let error { Text(error).foregroundStyle(.red) }
      HStack {
        if running { ProgressView().controlSize(.small); Text(store.bridge.message).font(.caption); Button("Cancel") { store.bridge.cancel() } }
        else { Button(proposal == nil ? "Suggest tags" : "Try again") { Task { await suggest() } }.disabled(modelPath.isEmpty || store.operationBusy || text.utf8.count > 12000) }
        Spacer()
        Button("Apply tags") { apply() }.buttonStyle(.borderedProminent).disabled(proposal == nil || running)
      }
      if text.utf8.count > 12000 { Text("Split this script into shorter dialogue lines before auto tagging (12,000 UTF-8 bytes per request).").font(.caption).foregroundStyle(.orange) }
    }.padding(24).frame(width: 650)
      .onAppear { if modelPath.isEmpty { modelPath = LocalPromptModels.discover().first?.path ?? "" } }
      .sheet(isPresented: $showSetup) { AssistantModelSetupView(store: store, currentModelPath: modelPath) { modelPath = $0 } }
      .interactiveDismissDisabled(running)
      .onDisappear { if running { store.bridge.cancel() } }
  }
  private func suggest() async {
    guard !store.operationBusy else { return }
    source = text
    let original = source
    running = true; error = nil; proposal = nil
    defer { running = false }
    do {
      snapshot = try VoiceTagContext(project: store.project, documentSessionID: store.documentSessionID,
        lineID: store.project.voiceDraft?.usesDialogue == true ? UUID(uuidString: contextID) : nil)
      let response = try await store.bridge.invoke("assist-prompt", runtime: store.runtime, payload: ["textRequest": [
        "modelPath": modelPath, "systemPrompt": FishVoiceTags.systemPrompt,
        "prompt": "Suggest delivery tag insertions for this script:\n" + original, "maxTokens": 1024, "images": []]])
      guard response["truncated"] as? Bool != true, let output = response["text"] as? String else {
        throw StudioError.invalid("Auto Tag did not finish its suggestion. Your script is unchanged; try a shorter line.")
      }
      proposal = try FishVoiceTags.applyingSuggestions(output, to: original)
    } catch { self.error = error.localizedDescription }
  }
  private func apply() {
    guard let proposal else { return }
    do {
      guard let snapshot, text == source else { throw StudioError.invalid("The script changed. Suggest tags again.") }
      try snapshot.validate(project: store.project, documentSessionID: store.documentSessionID)
    } catch { self.error = error.localizedDescription; return }
    text = proposal; dismiss()
  }
}
