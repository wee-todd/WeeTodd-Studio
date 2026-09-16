import AppKit
import StudioCore
import SwiftUI

struct WorkflowModelTurnsView: View {
  let directory: String
  let stepID: String
  let name: String
  let legacy: [WorkflowModelTurn]
  @Environment(\.dismiss) private var dismiss
  @State private var rows: [WorkflowTurnArchive.Row] = []
  @State private var total = 0
  @State private var limit = 100
  @State private var selected: String?
  @State private var detail: WorkflowModelTurn?
  @State private var tab = "Response"
  @State private var error: String?
  @State private var loadedRevision: Int?

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text("Model turns · " + name).font(.title2)
        Spacer()
        Button("Done") { dismiss() }
      }
      Text("Full recorded messages. New turns are chronological; older saved responses retain their checkpoint order, which may not be execution order.")
        .font(.caption).foregroundStyle(.secondary)
      HSplitView {
        VStack {
          List(selection: $selected) {
            if !legacy.isEmpty {
              Section("Older saved responses (\(legacy.count))") {
                ForEach(Array(legacy.enumerated()), id: \.element.id) { index, turn in
                  VStack(alignment: .leading) {
                    Text("Saved response \(index + 1)")
                    Text(turn.label).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                    if let seconds = turn.seconds { Text(String(format: "%.2f s", seconds)).font(.caption.monospacedDigit()) }
                  }.tag(turn.id)
                }
              }
            }
            Section("Recorded turns (\(total))") {
              ForEach(rows) { row in
                VStack(alignment: .leading, spacing: 4) {
                  Text("Turn \(row.ordinal) · \(row.kind == "reuse" ? "Reused response" : row.status.capitalized)")
                  Text(row.label).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                  if let seconds = row.seconds { Text(String(format: "%.2f s", seconds)).font(.caption.monospacedDigit()) }
                }.tag(row.id)
              }
            }
          }
          if rows.count < total { Button("Load next 100") { limit += 100; Task { await refresh() } } }
        }.frame(minWidth: 250, idealWidth: 290, maxWidth: 350)
        VStack(alignment: .leading, spacing: 12) {
          if let detail {
            HStack {
              Text(detail.status.capitalized).font(.headline)
              if let seconds = detail.seconds { Text(String(format: "%.2f seconds", seconds)).monospacedDigit() }
              if let started = detail.startedAt { Text(Date(timeIntervalSince1970: started), style: .time) }
              Spacer()
              Button("Copy text") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(contents(detail), forType: .string)
              }
            }
            if let error = detail.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
            if detail.kind == "legacy" { Text("Older record: missing messages, timestamps and validation results cannot be reconstructed.").font(.caption).foregroundStyle(.orange) }
            Picker("Turn detail", selection: $tab) {
              ForEach(["System", "Input", "Response", "Images", "Details"], id: \.self) { Text($0).tag($0) }
            }.pickerStyle(.segmented)
            ScrollView {
              Text(contents(detail)).font(.system(.body, design: .monospaced)).textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading).padding(12)
            }.background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
            Text("Returned text may still fail validation. Details shows the recorded JSON/schema check; semantic checks and human approval are separate.")
              .font(.caption).foregroundStyle(.secondary)
          } else { Text("Select a model turn.").foregroundStyle(.secondary); Spacer() }
        }.padding(.leading, 12).frame(minWidth: 580)
      }
      if let error { Text(error).foregroundStyle(.orange).font(.caption).textSelection(.enabled) }
      Text("Messages are local text records; images are referenced, not copied. New turns are stored separately from the abbreviated step overview.").font(.caption).foregroundStyle(.secondary)
    }.padding(22).frame(width: 1120, height: 760)
      .task {
        await refresh()
        if selected == nil { selected = legacy.first?.id ?? rows.first?.id }
        while !Task.isCancelled {
          do { try await Task.sleep(for: .seconds(2)) } catch { break }
          await refresh()
        }
      }
      .task(id: selected) { await loadSelected() }
  }

  private func contents(_ turn: WorkflowModelTurn) -> String {
    switch tab {
    case "System": return turn.system ?? "System message was not retained in this record."
    case "Input": return turn.prompt ?? "Input message was not retained in this record."
    case "Response": return turn.response ?? (turn.status == "running" ? "Waiting for the model response…" : "No response was recorded.")
    case "Images":
      guard let images = turn.images else { return "Image references were not retained in this record." }
      return images.isEmpty ? "No images were sent." : images.map { id in
        id + (turn.imageBindings?[id].map { "\n" + $0 } ?? "\nFile binding unavailable in this record.")
      }.joined(separator: "\n\n")
    default: return turn.details
    }
  }

  @MainActor private func refresh() async {
    let directory = directory, stepID = stepID, limit = limit
    do {
      let page = try await Task.detached(priority: .utility) {
        try WorkflowTurnArchive.list(directory: directory, stepID: stepID, limit: limit)
      }.value
      rows = page.rows; total = page.total; error = nil
      if selected == nil { selected = legacy.first?.id ?? rows.first?.id }
      if let row = rows.first(where: { $0.id == selected }), loadedRevision != row.revision { await loadSelected() }
    } catch { self.error = error.localizedDescription }
  }

  @MainActor private func loadSelected() async {
    guard let id = selected else { return }
    if let turn = legacy.first(where: { $0.id == id }) { detail = turn; loadedRevision = nil; return }
    let directory = directory
    let revision = rows.first(where: { $0.id == id })?.revision
    do {
      let turn = try await Task.detached(priority: .utility) {
        try WorkflowTurnArchive.read(directory: directory, id: id)
      }.value
      guard selected == id else { return }
      detail = turn; loadedRevision = revision
    } catch { self.error = error.localizedDescription }
  }
}
