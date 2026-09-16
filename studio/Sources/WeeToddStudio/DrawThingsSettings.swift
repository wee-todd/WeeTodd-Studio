import AppKit
import StudioCore
import SwiftUI

struct DrawThingsSettings: View {
  var onClose: (() -> Void)? = nil
  @EnvironmentObject var store: StudioStore
  @State private var draft = DrawThingsConnection()
  @State private var credential = ""
  @State private var message = ""
  @State private var accessingCredential = false
  var body: some View {
    VStack(alignment: .leading, spacing: 18) {
      HStack {
        Text("Draw Things Connections").font(.title2)
        Spacer()
        Button("Done") { if let onClose { onClose() } else { store.showDrawThings = false } }
      }
      HStack(alignment: .top, spacing: 22) {
        VStack(alignment: .leading) {
          ForEach(store.drawThingsConnections) { connection in
            Button(connection.name) { draft = connection; credential = ""; message = "" }
              .buttonStyle(.borderless)
          }
          Divider()
          Button("Add Connection") { draft = DrawThingsConnection(); credential = ""; message = "" }
        }.frame(width: 170, alignment: .leading)
        Form {
          TextField("Name", text: $draft.name)
          Picker("Route", selection: $draft.route) {
            Text("Self-hosted gRPC").tag("grpc")
            Text("DT+ App Bridge").tag("dtBridge")
            Text("Draw Things Cloud API").tag("dtCloud")
          }.onChange(of: draft.route) { _, route in
            draft.selfHostedConfirmed = false
            if route == "dtCloud" { draft.host = "compute.drawthings.ai"; draft.port = 443; draft.useTLS = true }
            else { draft.host = "127.0.0.1"; draft.port = 7859; draft.useTLS = false }
          }
          TextField("Host", text: $draft.host).disabled(draft.route == "dtCloud")
          TextField("Port", value: $draft.port, format: .number.grouping(.never)).disabled(draft.route == "dtCloud")
          Toggle("TLS with hostname verification", isOn: $draft.useTLS).disabled(draft.route == "dtCloud")
          if draft.route == "grpc" {
            Toggle("This server has cloud offload disabled", isOn: Binding(
              get: { draft.selfHostedConfirmed ?? false }, set: { draft.selfHostedConfirmed = $0 }))
          }
          SecureField(draft.route == "dtCloud" ? "API key" : "Shared secret (optional)", text: $credential)
          Text("Credentials stay in Keychain. Authorized access is reused until you quit or clear session access. Leave blank to keep the saved credential.")
            .font(.caption).foregroundStyle(.secondary)
          if draft.route == "dtCloud" {
            Link("Open Draw Things API dashboard", destination: URL(string: "https://api.drawthings.ai/dashboard")!)
          } else if draft.route == "dtBridge" {
            Text("Draw Things must stay open in Bridge Mode. Free-only generation remains unavailable until the bridge exposes a verifiable billing route.")
              .font(.caption).foregroundStyle(.secondary)
          }
          HStack {
            Button("Save") { Task { await save() } }
            Button("Test Connection") { Task { if await save() { await store.testDrawThings(draft) } } }
              .disabled(store.bridge.busy)
            if let reference = draft.credentialRef {
              Button("Remove Credential") {
                Task {
                  accessingCredential = true
                  do {
                    _ = try await BackgroundCredential.read {
                      try DrawThingsCredential.remove(reference); return nil
                    }
                    draft.credentialRef = nil; credential = ""
                    accessingCredential = false
                    _ = await save()
                  } catch { accessingCredential = false; message = error.localizedDescription }
                }
              }
            }
          }
          Button("Clear Session Access") {
            Task {
              accessingCredential = true
              _ = try? await BackgroundCredential.read {
                DrawThingsCredential.clearSession(); return nil
              }
              accessingCredential = false
              message = "Session access cleared. The next request will read Keychain again."
            }
          }.help("Forget credentials held in memory for all Draw Things connections. Saved keys remain in Keychain.")
          if accessingCredential { ProgressView("Waiting for Keychain…").controlSize(.small) }
          if let catalog = store.drawThingsCatalogs[draft.id] {
            Text("\((catalog["capabilities"] as? [String: Any])?.count ?? 0) supported models · \(store.drawThingsDiscovery.errors[draft.id] == nil ? "connection verified" : "last loaded catalog")")
              .font(.caption).foregroundStyle(.secondary)
            if let account = catalog["account"] as? [String: Any] {
              if let quota = account["monthlyQuota"] as? [String: Any],
                let remaining = quota["remainingRequests"] as? NSNumber {
                Text("Free requests remaining: \(remaining.stringValue)").font(.caption)
              }
              if let reason = account["reason"] as? String { Text(reason).font(.caption).foregroundStyle(.secondary) }
            }
          }
          if store.drawThingsDiscovery.loading.contains(draft.id) { ProgressView("Loading model catalog…").controlSize(.small) }
          if let failure = store.drawThingsDiscovery.errors[draft.id] { Text(failure).font(.caption).foregroundStyle(.orange) }
          if !message.isEmpty { Text(message).font(.caption).foregroundStyle(.secondary) }
        }.formStyle(.grouped).frame(maxWidth: .infinity)
      }.disabled(accessingCredential)
      Divider()
      HStack {
        Text("Transport helper").font(.caption)
        TextField("WeeToddDrawThings", text: Binding(get: { store.runtime.drawThingsHelperPath ?? "" },
          set: { store.runtime.drawThingsHelperPath = $0 })).textFieldStyle(.roundedBorder)
        Button("Import…") {
          let panel = NSOpenPanel(); panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
          if panel.runModal() == .OK, let url = panel.url {
            store.runtime.drawThingsHelperPath = url.path
            store.saveRuntime(reloadProfiles: false)
          }
        }
      }
    }.padding(24).frame(width: 840, height: 660)
      .onAppear { if let first = store.drawThingsConnections.first { draft = first } }
  }
  @discardableResult func save() async -> Bool {
    guard !accessingCredential else { return false }
    accessingCredential = true
    defer { accessingCredential = false }
    guard !draft.name.trimmingCharacters(in: .whitespaces).isEmpty,
      !draft.host.isEmpty, (1...65535).contains(draft.port) else { message = "Enter a name, host, and valid port."; return false }
    do {
      if !credential.isEmpty {
        let reference = draft.credentialRef ?? draft.id
        let value = credential
        _ = try await BackgroundCredential.read {
          try DrawThingsCredential.save(value, reference: reference); return nil
        }
        draft.credentialRef = reference; credential = ""
      }
      if let index = store.drawThingsConnections.firstIndex(where: { $0.id == draft.id }) {
        if store.drawThingsConnections[index] != draft { store.drawThingsDiscovery.invalidate(draft.id) }
        store.drawThingsConnections[index] = draft
      } else { store.drawThingsConnections.append(draft) }
      store.saveDrawThingsConnections()
      store.saveRuntime(reloadProfiles: false)
      message = "Connection saved."
      return true
    } catch { message = error.localizedDescription; return false }
  }
}
