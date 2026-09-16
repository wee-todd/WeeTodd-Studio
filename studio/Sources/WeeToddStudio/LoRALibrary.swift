import AppKit
import StudioCore
import SwiftUI

struct LoRALibrary: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  @State private var browsingEngine: Engine = .ltx25
  @State private var importModel: LoRAModel = .ltx25
  @State private var importProfile = "standard"
  @State private var importLayout = "auto"
  @State private var importGrid: String?
  @State private var draft: LoRAGroup?
  @State private var search = ""
  var engine: Engine { store.selectedClip?.engine ?? browsingEngine }
  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      HStack {
        VStack(alignment: .leading) {
          Text("LoRAs & Groups").font(.title2.bold())
          Text(
            store.selectedClip.map { "For \($0.name) · \($0.engine.label)" } ?? "Reusable library"
          )
          .foregroundStyle(.secondary)
        }
        Spacer()
        Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
      }
      if store.selectedClip == nil {
        Picker("Model", selection: $browsingEngine) {
          ForEach(Engine.allCases.filter { $0 != .movie && $0 != .drawThings }) {
            Text($0.label).tag($0)
          }
        }.onChange(of: browsingEngine) { _, value in
          draft = nil
          importModel = LoRAModel(rawValue: value.rawValue) ?? .ltx25
        }
      }
      if engine == .movie {
        ContentUnavailableView(
          "Select a generated clip", systemImage: "slider.horizontal.3",
          description: Text("Movie assets do not use LoRAs."))
      } else {
        HStack(alignment: .top, spacing: 20) {
          library.frame(maxWidth: .infinity, maxHeight: .infinity)
          Divider()
          groups.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        if let message = store.error {
          HStack(alignment: .top) {
            Label(message, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(
              .red)
            Spacer()
            Button("Dismiss") { store.error = nil }
          }
        } else {
          Text(store.notice).font(.caption).foregroundStyle(.secondary).lineLimit(2)
        }
        Text(
          "Files stay linked. Applying a group copies its settings into the clip. Model and recipe checks run before generation; control and reference adapters belong in task recipes."
        )
        .font(.caption).foregroundStyle(.secondary)
      }
    }.padding(24).frame(width: 800, height: 580)
      .onAppear { importModel = LoRAModel(rawValue: engine.rawValue) ?? .ltx25 }
  }
  var library: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("LoRA library").font(.headline)
      HStack {
        Picker("Trained model", selection: $importModel) {
          ForEach(LoRAModel.allCases.filter { $0.supports(engine) }) { Text($0.label).tag($0) }
        }
        Button("Import…") {
          store.chooseLoRAImports(model: importModel,
            profile: importModel == .h3 ? importProfile : "standard",
            layout: importModel == .h3 ? importLayout : "auto",
            adalnInputGrid: importModel == .h3 && importProfile == "turbo" ? importGrid : nil)
        }
      }.controlSize(.small)
      if importModel == .h3 {
        Picker("Adapter", selection: $importProfile) {
          Text("Standard").tag("standard")
          Text("H3 Turbo (4 steps)").tag("turbo")
        }
        if importProfile == "turbo" {
          Text("Turbo requires its four-step schedule. Validation checks task compatibility and any auxiliary weights before generation.")
            .font(.caption).foregroundStyle(.secondary)
          DisclosureGroup("Adapter file details") {
            Picker("Tensor layout", selection: $importLayout) {
              Text("Auto").tag("auto")
              Text("Native interleaved").tag("native_interleaved")
              Text("Contiguous QKV").tag("contiguous_qkv")
            }
            HStack {
              Button("Link AdaLN grid…") { chooseImportGrid() }
              if let importGrid {
                Text(URL(fileURLWithPath: importGrid).lastPathComponent).lineLimit(1).help(importGrid)
                Button("Clear") { self.importGrid = nil }
              }
            }
            Text("Link the adapter's AdaLN input grid when its training format requires a separate file.")
              .font(.caption2).foregroundStyle(.secondary)
          }.controlSize(.small)
        }
      }
      TextField("Search LoRAs", text: $search).textFieldStyle(.roundedBorder)
      ScrollView {
        VStack(alignment: .leading, spacing: 10) {
          ForEach(
            store.compatibleLoRAs(for: engine).filter {
              search.isEmpty || $0.name.localizedCaseInsensitiveContains(search)
            }
          ) { asset in
            VStack(alignment: .leading, spacing: 6) {
              Text(asset.name).font(.subheadline.bold()).lineLimit(2)
              Text(asset.loraModel?.label ?? "Choose trained model").font(.caption).foregroundStyle(
                .secondary)
              if asset.loraProfile == "turbo" {
                Text("H3 Turbo · requires 4 steps").font(.caption).foregroundStyle(.orange)
              }
              HStack {
                Button("Add to current stack") { store.applyLoRAMembers([LoRAMember(asset: asset)]) }
                  .disabled(store.selectedClip == nil)
                if draft != nil {
                  Button("Add to group") { draft?.members.append(LoRAMember(asset: asset)) }
                    .disabled(
                      draft?.members.contains { $0.fileKey == LoRAMember(asset: asset).fileKey }
                        == true)
                }
                Menu {
                  Button("Relink…") { store.relink(asset) }
                  Button("Remove from library", role: .destructive) { removeLibraryAsset(asset) }
                } label: {
                  Image(systemName: "ellipsis")
                }
                .menuStyle(.borderlessButton).frame(width: 22)
              }.controlSize(.small)
            }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
              .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
          }
          let unknown = store.allAssets.filter {
            $0.kind == .lora && $0.loraModel == nil
              && ($0.scope != .clip || $0.owner == store.selectedClipID)
          }
          if !unknown.isEmpty {
            Text("Imports needing a trained model").font(.subheadline.bold())
            ForEach(unknown) { asset in
              HStack {
                Text(asset.name).lineLimit(2)
                Menu("Set model") {
                  ForEach(LoRAModel.allCases) { model in
                    Button(model.label) { store.setLoRAModel(model, for: asset) }
                  }
                }
              }.font(.caption)
            }
          }
          if store.compatibleLoRAs(for: engine).isEmpty {
            Text(
              "Import a LoRA trained for \(engine.label)\(engine == .ltx25 ? " or LTX 2.3" : "")."
            )
            .font(.callout).foregroundStyle(.secondary).padding(.vertical)
          }
        }
      }
    }
  }
  @ViewBuilder var groups: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text("Groups · \(engine.label)").font(.headline)
        Spacer()
        Button("New group") { draft = LoRAGroup(name: "New group", engine: engine) }.disabled(
          draft != nil)
      }
      Button("Save current stack as group") { draft = store.currentLoRAGroup() }
        .disabled(draft != nil || store.selectedClip?.attachments.contains { $0.role == .lora } != true)
      if let value = draft {
        TextField(
          "Group name", text: Binding(get: { draft?.name ?? "" }, set: { draft?.name = $0 })
        )
        .textFieldStyle(.roundedBorder)
        Text("Add LoRAs from the library, then set their strengths.").font(.caption)
          .foregroundStyle(.secondary)
        ScrollView {
          VStack(alignment: .leading, spacing: 12) {
            ForEach(value.members) { member in
              VStack(alignment: .leading, spacing: 6) {
                HStack {
                  Toggle(member.asset.name, isOn: Binding(get: {
                    draft?.members.first { $0.id == member.id }?.isEnabled ?? member.isEnabled
                  }, set: { enabled in
                    if let index = draft?.members.firstIndex(where: { $0.id == member.id }) {
                      draft?.members[index].enabled = enabled
                    }
                  })).lineLimit(2)
                  Spacer()
                  Button {
                    draft?.members.removeAll { $0.id == member.id }
                  } label: {
                    Image(systemName: "minus.circle")
                  }.help("Remove from group")
                }
                LoRAStrength(
                  value: Binding(
                    get: {
                      draft?.members.first { $0.id == member.id }?.strength ?? member.strength
                    },
                    set: { strength in
                      if let index = draft?.members.firstIndex(where: { $0.id == member.id }) {
                        draft?.members[index].strength = strength
                      }
                    }))
              }
            }
          }
        }
        HStack {
          Button("Cancel") { draft = nil }
          Spacer()
          Button("Save group") {
            if let group = draft, store.saveLoRAGroup(group) { draft = nil }
          }.disabled(
            value.members.isEmpty || value.name.trimmingCharacters(in: .whitespaces).isEmpty)
        }
      } else {
        ScrollView {
          VStack(alignment: .leading, spacing: 12) {
            ForEach(store.loraGroups.filter {
              $0.engine == engine && (search.isEmpty || $0.name.localizedCaseInsensitiveContains(search)
                || $0.members.contains { $0.asset.name.localizedCaseInsensitiveContains(search) })
            }) { group in
              VStack(alignment: .leading, spacing: 6) {
                Text(group.name).font(.subheadline.bold())
                Text(
                  group.members.map { "\($0.asset.name) · \(String(format: "%.2f", $0.strength))\($0.isEnabled ? "" : " · disabled")" }
                    .joined(separator: "\n")
                )
                .font(.caption).foregroundStyle(.secondary)
                HStack {
                  Menu("Apply group") {
                    ForEach(LoRAGroupApplicationMode.allCases) { mode in
                      Button(mode.label) { store.applyLoRAGroup(group, mode: mode) }
                    }
                  }.disabled(store.selectedClip == nil)
                  Button("Edit") { draft = group }
                  Spacer()
                  Button {
                    store.deleteLoRAGroup(group.id)
                  } label: {
                    Image(systemName: "trash")
                  }.help("Delete group template; applied clips keep their settings")
                }.controlSize(.small)
              }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
            }
          }
        }
      }
    }
  }
  func chooseImportGrid() {
    let panel = NSOpenPanel()
    panel.title = "Link H3 Turbo AdaLN input grid"
    panel.canChooseDirectories = false
    panel.allowedContentTypes = [.init(filenameExtension: "safetensors") ?? .data]
    if panel.runModal() == .OK { importGrid = panel.url?.path }
  }
  func removeLibraryAsset(_ asset: MediaAsset) {
    if store.project.clips.contains(where: { $0.attachments.contains { $0.assetID == asset.id } }) {
      store.error = "Remove the clip attachment before removing its asset link."
    } else if asset.scope == .global {
      store.globalAssets.removeAll { $0.id == asset.id }
      store.saveGlobals()
    } else {
      store.change { $0.assets.removeAll { $0.id == asset.id } }
    }
  }
}

struct LoRAStrength: View {
  @Binding var value: Double
  var body: some View {
    HStack {
      Text("Strength").font(.caption)
      Slider(value: $value, in: 0...2, step: 0.01).accessibilityLabel("LoRA strength")
      TextField("LoRA strength", value: $value, format: .number.precision(.fractionLength(2)))
        .textFieldStyle(.roundedBorder).frame(width: 58)
    }
  }
}

struct ClipLoRAInspector: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        SmallLabel(text: "LoRAs")
        Spacer()
        Button("Add / Groups…") { store.showLoRALibrary = true }.controlSize(.small)
      }
      ForEach(clip.attachments.filter { $0.role == .lora }) { attachment in
        let asset = store.allAssets.first { $0.id == attachment.assetID }
        VStack(alignment: .leading, spacing: 6) {
          HStack {
            Toggle(asset?.name ?? "Missing LoRA", isOn: Binding(get: {
              store.selectedClip?.attachments.first { $0.id == attachment.id }?.isEnabled ?? attachment.isEnabled
            }, set: { enabled in
              store.editClip { selected in
                if let index = selected.attachments.firstIndex(where: { $0.id == attachment.id }) {
                  selected.attachments[index].enabled = enabled
                }
              }
            })).font(.caption.bold()).lineLimit(2)
            Spacer()
            Button {
              store.editClip { $0.attachments.removeAll { $0.id == attachment.id } }
            } label: {
              Image(systemName: "xmark")
            }.help("Remove LoRA")
          }
          if asset?.loraProfile == "turbo" {
            Text(attachment.isEnabled ? "H3 Turbo · requires 4 steps" : "H3 Turbo · disabled")
              .font(.caption2).foregroundStyle(.secondary)
          }
          if let group = attachment.loraGroupName, let groupID = attachment.loraGroupID {
            HStack {
              Label(group, systemImage: "folder").font(.caption2)
              Spacer()
              Button("Remove group") {
                store.editClip {
                  $0.attachments.removeAll { $0.role == .lora && $0.loraGroupID == groupID }
                }
              }.font(.caption2).buttonStyle(.link)
            }
          }
          if asset?.loraModel?.supports(clip.engine) != true {
            Text("Choose a compatible LoRA or set its trained model in the library.").font(
              .caption2
            ).foregroundStyle(.red)
          }
          LoRAStrength(
            value: Binding(
              get: {
                store.selectedClip?.attachments.first { $0.id == attachment.id }?.strength
                  ?? attachment.strength
              },
              set: { strength in
                store.editClip { c in
                  if let index = c.attachments.firstIndex(where: { $0.id == attachment.id }) {
                    c.attachments[index].strength = strength
                  }
                }
              }))
        }.padding(9).background(Theme.raised, in: RoundedRectangle(cornerRadius: 7))
      }
    }
  }
}
