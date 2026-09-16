import AppKit
import StudioCore
import SwiftUI

struct WorkflowHistoryView: View {
  let directory: String
  let workflowBusy: Bool
  let inspectStep: (String) -> Void
  @Environment(\.dismiss) private var dismiss
  @State private var snapshot: WorkflowHistorySnapshot?
  @State private var selected: String?
  @State private var error: String?
  @State private var modified: Date?
  @State private var showTurns = false

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack {
        Text("Workflow execution history").font(.title2)
        Spacer()
        Button("Reveal records") { NSWorkspace.shared.open(URL(fileURLWithPath: directory)) }
        Button("Done") { dismiss() }
      }
      if let snapshot {
        HStack {
          Text(snapshot.status.replacingOccurrences(of: "_", with: " ").capitalized).font(.headline)
          Text("Recorded workflow time: \(duration(snapshot.totalSeconds))").foregroundStyle(.secondary)
          Spacer()
          if workflowBusy { ProgressView().controlSize(.small); Text("Updates every 2 seconds").font(.caption) }
        }
        HSplitView {
          List(snapshot.definition.steps, selection: $selected) { spec in
            VStack(alignment: .leading, spacing: 5) {
              Text(spec.name)
              Text(snapshot.steps[spec.id]?.status.replacingOccurrences(of: "_", with: " ") ?? "Not run").font(.caption).foregroundStyle(.secondary)
              if let seconds = snapshot.steps[spec.id]?.seconds {
                Text(duration(seconds)).font(.caption.monospacedDigit())
              }
            }.padding(.vertical, 3).tag(spec.id)
          }.frame(minWidth: 220, idealWidth: 250, maxWidth: 320)
          ScrollView {
            VStack(alignment: .leading, spacing: 14) {
              if let spec = snapshot.definition.steps.first(where: { $0.id == selected }) {
                Text(spec.name).font(.title3)
                Text(WorkflowOperationDescription.text(spec.operation))
                Text(spec.operation).font(.caption.monospaced()).foregroundStyle(.secondary)
                if let step = snapshot.steps[spec.id] {
                  if let seconds = step.seconds { Text("Time stored for this step: \(duration(seconds))") }
                  Text("\(step.savedResponses) saved model responses, including child records. This is retained output, not a lifetime call count.")
                    .font(.caption).foregroundStyle(.secondary)
                  if let error = step.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
                  Button("Model turns…") { showTurns = true }
                  Button("View step result") { inspectStep(spec.id); dismiss() }.disabled(workflowBusy)
                }
                Divider()
                let entries = snapshot.executionHistory.filter { $0.stepID == spec.id }
                if entries.isEmpty {
                  Text("No detailed execution history was recorded for this step. Older checkpoints show their saved timing and responses above; discarded retries and past regenerations cannot be reconstructed.")
                    .foregroundStyle(.secondary)
                } else {
                  Text("Recorded activity · chronological").font(.headline)
                  ForEach(entries) { entry in activity(entry) }
                }
              } else { Text("Select a step to inspect its purpose, timing and recorded activity.").foregroundStyle(.secondary) }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(12)
          }.frame(minWidth: 500)
        }
        if snapshot.historyOmitted > 0 { Text("\(snapshot.historyOmitted) older activity entries omitted to keep records small.").font(.caption).foregroundStyle(.secondary) }
        Text("Step times include local work, model loading and validation. Calls are nested within steps: do not add their times to step totals. Human approval waiting time is excluded. History retains the latest 40 activities and 20 call/retry details per activity.")
          .font(.caption).foregroundStyle(.secondary)
      } else { Text(error ?? "Waiting for the first saved checkpoint…").foregroundStyle(.secondary); Spacer() }
      if snapshot != nil, let error { Text(error).font(.caption).foregroundStyle(.orange) }
    }.padding(22).frame(width: 1060, height: 730)
      .sheet(isPresented: $showTurns) {
        if let selected, let spec = snapshot?.definition.steps.first(where: { $0.id == selected }) {
          WorkflowModelTurnsView(directory: directory, stepID: selected, name: spec.name,
                                 legacy: snapshot?.steps[selected]?.legacyTurns ?? [])
        }
      }
      .task(id: directory) {
        while !Task.isCancelled {
          await refresh()
          do { try await Task.sleep(for: .seconds(2)) } catch { break }
        }
      }
  }

  @ViewBuilder private func activity(_ entry: WorkflowHistorySnapshot.Entry) -> some View {
    VStack(alignment: .leading, spacing: 7) {
      HStack {
        Text(Date(timeIntervalSince1970: entry.startedAt), style: .time)
        Text(entry.reason).fontWeight(.medium)
        Spacer()
        Text(entry.status.capitalized)
      }
      SwiftUI.TimelineView(.periodic(from: .now, by: 1)) { time in
        Text("\(duration(entry.elapsed(at: time.date))) · \(entry.modelCalls) model requests · \(entry.reusedCalls) reused responses · \(entry.retries) step retries")
          .font(.caption.monospacedDigit())
      }
      if entry.timingIncomplete == true { Text("Process ended without final timing; displayed time is partial.").font(.caption).foregroundStyle(.orange) }
      if entry.status == "running", !workflowBusy { Text("No active workflow request in this window. This may be an interrupted run or a job running elsewhere.").font(.caption).foregroundStyle(.orange) }
      if let message = entry.message { Text(message).font(.caption).foregroundStyle(.secondary) }
      if let error = entry.error { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
      if !entry.events.isEmpty {
        DisclosureGroup("Call and retry details") {
          ForEach(entry.events) { event in
            VStack(alignment: .leading, spacing: 4) {
              HStack {
                Text(event.kind.capitalized + " · " + (event.status ?? "Recorded")).bold()
                if let seconds = event.seconds { Text(duration(seconds)) }
                Spacer()
                if let key = event.requestKey { Text(key).font(.caption.monospaced()).help("Same request text and image IDs produce the same fingerprint; this does not establish equivalent model/runtime or image file contents.") }
              }
              Text(event.message)
              if let purpose = event.purpose { Text(purpose).foregroundStyle(.secondary) }
              if let error = event.error { Text(error).foregroundStyle(.red) }
            }.font(.caption).textSelection(.enabled).padding(.vertical, 5)
            Divider()
          }
          if let count = entry.eventsOmitted, count > 0 { Text("\(count) earlier details omitted; counters still cover the entire activity.").font(.caption) }
        }
      }
    }.padding(12).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
  }

  private func duration(_ seconds: Double) -> String {
    seconds >= 60 ? String(format: "%dm %.1fs", Int(seconds / 60), seconds.truncatingRemainder(dividingBy: 60)) : String(format: "%.1fs", seconds)
  }

  @MainActor private func refresh() async {
    let previous = modified
    let file = URL(fileURLWithPath: directory).appendingPathComponent("run.json")
    do {
      let update = try await Task.detached(priority: .utility) { () -> (Date?, WorkflowHistorySnapshot)? in
        guard FileManager.default.fileExists(atPath: file.path) else { return nil }
        let info = try file.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
        guard (info.fileSize ?? Int.max) <= 2 * 1024 * 1024 else { throw StudioError.invalid("Checkpoint exceeds the 2 MiB inspection limit.") }
        if let date = info.contentModificationDate, date == previous { return nil }
        let data = try Data(contentsOf: file)
        guard data.count <= 2 * 1024 * 1024 else { throw StudioError.invalid("Checkpoint exceeds the inspection limit.") }
        return (info.contentModificationDate, try JSONDecoder().decode(WorkflowHistorySnapshot.self, from: data))
      }.value
      if let update {
        modified = update.0; snapshot = update.1; error = nil
        if selected == nil { selected = update.1.executionHistory.last?.stepID ?? update.1.definition.steps.first?.id }
      }
    } catch { self.error = "Could not refresh history: \(error.localizedDescription)" }
  }
}
