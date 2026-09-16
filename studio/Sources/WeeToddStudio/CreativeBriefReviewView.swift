import StudioCore
import SwiftUI

struct CreativeBriefReviewView: View {
  @Binding var draft: DirectorBriefDraft?
  let value: CreativeBrief
  let approved: Bool
  let busy: Bool
  let onSave: ([String: JSONValue]) async -> Bool
  let onDirtyChange: (Bool) -> Void
  private var baseline: CreativeBrief {
    get { (draft ?? DirectorBriefDraft(value)).baseline }
    nonmutating set { var edit = draft ?? DirectorBriefDraft(value); edit.baseline = newValue; draft = edit }
  }
  private var answers: [String: String] {
    get { (draft ?? DirectorBriefDraft(value)).answers }
    nonmutating set { var edit = draft ?? DirectorBriefDraft(value); edit.answers = newValue; draft = edit }
  }
  private var preferences: CreativeBriefPreferences {
    get { (draft ?? DirectorBriefDraft(value)).preferences }
    nonmutating set { var edit = draft ?? DirectorBriefDraft(value); edit.preferences = newValue; draft = edit }
  }
  private var duration: String {
    get { (draft ?? DirectorBriefDraft(value)).duration }
    nonmutating set { var edit = draft ?? DirectorBriefDraft(value); edit.duration = newValue; draft = edit }
  }
  private var clipDuration: String {
    get { (draft ?? DirectorBriefDraft(value)).clipDuration }
    nonmutating set { var edit = draft ?? DirectorBriefDraft(value); edit.clipDuration = newValue; draft = edit }
  }
  private var frameRate: String {
    get { (draft ?? DirectorBriefDraft(value)).frameRate }
    nonmutating set { var edit = draft ?? DirectorBriefDraft(value); edit.frameRate = newValue; draft = edit }
  }
  @State private var saving = false
  @State private var error: String?

  private var dirty: Bool {
    preferences != baseline.preferences || baseline.questions.contains { answers[$0.id] != $0.answer }
      || duration != Self.number(baseline.preferences.durationSeconds)
      || clipDuration != Self.number(baseline.preferences.targetClipSeconds)
      || frameRate != String(baseline.preferences.frameRate)
  }
  private var editedPreferences: CreativeBriefPreferences? {
    guard let seconds = Double(duration), let clipSeconds = Double(clipDuration), let fps = Int(frameRate) else { return nil }
    var edited = preferences
    edited.durationSeconds = seconds; edited.targetClipSeconds = clipSeconds; edited.frameRate = fps
    return edited
  }
  private var preferenceIssues: [String] {
    editedPreferences?.validationIssues ?? ["Enter numbers for the movie duration, clip duration, and frame rate."]
  }

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 18) {
        Text("Creative brief").font(.title2.bold())
        Text("Review your original story, answer each question, and adjust the creative preferences. Save your answers before approving the brief.")
          .font(.callout).foregroundStyle(.secondary)
        facts
        preferenceFields
        questions
        if !value.referenceObservations.isEmpty {
          DisclosureGroup("Reference image observations") {
            ForEach(value.referenceObservations.indices, id: \.self) { index in
              Text(Self.observation(value.referenceObservations[index])).font(.caption)
                .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
            }
          }
        }
        DisclosureGroup("Original source") {
          Text(value.sourceText).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
        }
        if baseline != value && dirty {
          Text("This draft belongs to an earlier saved brief. Keep it for reference or reset it before saving.").font(.caption).foregroundStyle(.orange)
          Button("Reset draft to saved brief") { reset(value) }
        }
        ForEach(preferenceIssues, id: \.self) { Text($0).font(.caption).foregroundStyle(.orange) }
        HStack {
          if saving { ProgressView().controlSize(.small) }
          Text(approved ? "Approved · unlock to edit" : (dirty ? "Unsaved changes" : "Answers saved"))
            .font(.caption).foregroundStyle(.secondary)
          Spacer()
          Button("Save answers") { Task { await save() } }
            .disabled(!dirty || approved || busy || saving || !preferenceIssues.isEmpty || baseline != value)
        }
        if let error { Text(error).font(.caption).foregroundStyle(.red) }
      }.padding(12)
    }
    .onAppear { onDirtyChange(dirty) }
    .onChange(of: dirty) { _, changed in onDirtyChange(changed) }
    .onChange(of: value) { _, newValue in if !dirty { reset(newValue) } }
  }

  private var facts: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("Story you provided").font(.headline)
      if value.facts.isEmpty { Text("No source passages are available.").font(.caption).foregroundStyle(.secondary) }
      ForEach(value.facts.indices, id: \.self) { index in
        VStack(alignment: .leading, spacing: 4) {
          Text(value.facts[index].text).textSelection(.enabled)
          if value.facts[index].evidence != value.facts[index].text {
            Text("Evidence: \(value.facts[index].evidence)").font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
          }
        }.frame(maxWidth: .infinity, alignment: .leading)
      }
    }
  }

  private var preferenceFields: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("Creative preferences").font(.headline)
      Text("\(duration) seconds · about \(clipDuration) seconds per clip · \(frameRate) fps")
        .font(.caption).foregroundStyle(.secondary)
      LabeledContent("Movie duration (seconds)") { TextField("Seconds", text: draftBinding(\.duration)).frame(maxWidth: 140) }
      LabeledContent("Target clip duration (seconds)") { TextField("Seconds per clip", text: draftBinding(\.clipDuration)).frame(maxWidth: 140) }
      LabeledContent("Frame rate (fps)") { TextField("Frames per second", text: draftBinding(\.frameRate)).frame(maxWidth: 140) }
      preferenceText("Visual style", text: draftBinding(\.preferences).visualStyle)
      preferenceText("Presentation", text: draftBinding(\.preferences).presentation)
      preferenceText("Camera style", text: draftBinding(\.preferences).cameraStyle)
      preferenceText("Audio style", text: draftBinding(\.preferences).audioStyle)
      preferenceText("Design policy", text: draftBinding(\.preferences).designPolicy)
      preferenceText("Constraints", text: draftBinding(\.preferences).constraints)
    }.textFieldStyle(.roundedBorder).disabled(approved || busy || saving)
  }

  private func preferenceText(_ title: String, text: Binding<String>) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(title).font(.subheadline)
      TextField(title, text: text, axis: .vertical).lineLimit(1...4).accessibilityLabel(title)
    }
  }

  private var questions: some View {
    VStack(alignment: .leading, spacing: 14) {
      Text("Decisions to review").font(.headline)
      if value.questions.isEmpty { Text("No open questions.").font(.caption).foregroundStyle(.secondary) }
      ForEach(value.questions) { question in
        VStack(alignment: .leading, spacing: 6) {
          Text(question.prompt).font(.subheadline.bold()).textSelection(.enabled)
          if question.requiresExplicitChoice {
            Text("Choose the identity explicitly. This decision cannot be delegated to the director.")
              .font(.caption).foregroundStyle(.secondary)
          }
          if !question.options.isEmpty {
            Picker("Suggested answer", selection: Binding(get: {
              question.options.firstIndex(of: answers[question.id] ?? "") ?? -1
            }, set: { index in
              answers[question.id] = question.options.indices.contains(index) ? question.options[index] : ""
            })) {
              Text("Write your own answer").tag(-1)
              ForEach(question.options.indices, id: \.self) { index in Text(question.options[index]).tag(index) }
            }
          }
          TextField("Your answer", text: Binding(get: { answers[question.id] ?? "" }, set: { answers[question.id] = $0 }), axis: .vertical)
            .lineLimit(1...5).textFieldStyle(.roundedBorder).accessibilityLabel("Answer: \(question.prompt)")
          if !answered(question) { Text("Answer required before approval").font(.caption).foregroundStyle(.orange) }
        }.padding(10).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
      }
    }.disabled(approved || busy || saving)
  }

  private func draftBinding<T>(_ key: WritableKeyPath<DirectorBriefDraft, T>) -> Binding<T> {
    Binding(get: { (draft ?? DirectorBriefDraft(value))[keyPath: key] }, set: { newValue in
      var edit = draft ?? DirectorBriefDraft(value); edit[keyPath: key] = newValue; draft = edit
    })
  }
  private func answered(_ question: CreativeBrief.Question) -> Bool {
    var edited = question; edited.answer = answers[question.id] ?? ""
    return edited.isAnswered
  }
  private func save() async {
    guard let editedPreferences else { return }
    saving = true; error = nil
    defer { saving = false }
    do {
      let edited = try baseline.editing(answers: answers, preferences: editedPreferences)
      if await onSave(try edited.outputs()) { reset(edited) }
      else { error = "Could not save your answers. See the workflow error below." }
    } catch { self.error = error.localizedDescription }
  }
  private func reset(_ brief: CreativeBrief) {
    baseline = brief; preferences = brief.preferences
    answers = Dictionary(brief.questions.map { ($0.id, $0.answer) }, uniquingKeysWith: { first, _ in first })
    duration = Self.number(brief.preferences.durationSeconds)
    clipDuration = Self.number(brief.preferences.targetClipSeconds)
    frameRate = String(brief.preferences.frameRate)
    error = nil; onDirtyChange(false)
  }
  private static func number(_ value: Double) -> String {
    let text = String(value)
    return text.hasSuffix(".0") ? String(text.dropLast(2)) : text
  }
  private static func observation(_ value: JSONValue) -> String {
    switch value {
    case .string(let text): return text
    case .array(let entries): return entries.map(observation).joined(separator: "\n")
    case .object(let fields): return fields.keys.sorted().map { "\($0): \(observation(fields[$0]!))" }.joined(separator: "\n")
    case .integer(let number): return String(number)
    case .number(let number): return String(number)
    case .boolean(let flag): return flag ? "Yes" : "No"
    case .null: return ""
    }
  }
}
