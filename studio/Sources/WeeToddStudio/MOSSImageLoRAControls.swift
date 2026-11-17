import AppKit
import StudioCore
import SwiftUI

struct MOSSImageLoRAControls: View {
  @EnvironmentObject var store: StudioStore
  @State private var groupName = "MOSS image stack"
  private var members: [LoRAMember] { store.imageDraft?.mossImage?.loras ?? [] }
  var body: some View {
    Section("LoRAs & Groups") {
      Button("Link MOSS adapters…") { store.chooseLoRAImports(model: .mossImage) }.disabled(store.operationBusy)
      Menu("Add from library") {
        ForEach(store.localLoRALibraryAssets.filter { $0.loraModel == .mossImage }) { asset in
          Button(asset.name + (asset.loraProfile == "moss_sft_4step_v2" ? " · SFT / 4 Euler steps / CFG 1 / strength 1" : "")) { apply([.init(asset: asset)]) }
        }
      }.disabled(store.operationBusy)
      ForEach(members) { member in
        VStack(alignment: .leading) {
          HStack {
            Toggle(member.asset.name, isOn: Binding(get: { member.isEnabled }, set: { enabled in
              edit { try $0.setMOSSAdapterEnabled(id: member.id, enabled: enabled) }
            }))
            Button { edit { draft in
              draft.mossImage?.loras?.removeAll { $0.id == member.id }; try draft.reconcileMOSSFastRecipe()
            } } label: { Image(systemName: "xmark.circle") }
          }
          TextField("Strength", value: Binding(get: { member.strength }, set: { strength in edit { draft in
            guard let i = draft.mossImage?.loras?.firstIndex(where: { $0.id == member.id }) else { return }
            draft.mossImage?.loras?[i].strength = strength; try draft.reconcileMOSSFastRecipe()
          } }), format: .number)
          if member.asset.loraProfile == "moss_sft_4step_v2" {
            Text("Four-step SFT v2 · applies strength 1, CFG 1, shift 5 and release Euler. Disabling restores your stock recipe.").font(.caption2).foregroundStyle(.orange)
          }
        }.disabled(store.operationBusy)
      }
      if !members.isEmpty {
        TextField("Group name", text: $groupName)
        Button("Save current stack as group") {
          var group = LoRAGroup(name: groupName, engine: .drawThings, members: members); group.imageProvider = .nativeMOSS
          _ = store.saveLoRAGroup(group)
        }
      }
      ForEach(store.loraGroups.filter { $0.imageProvider == .nativeMOSS }) { group in
        HStack {
          Menu(group.name) {
            Button("Add to current stack") { apply(group.members) }
            Button("Replace current stack") { apply(group.members, mode: .replace) }
          }
          Button { store.deleteLoRAGroup(group.id) } label: { Image(systemName: "trash") }
        }
      }
      Link("Four-step adapter source", destination: URL(string: "https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps/tree/2ded95c37a029360245926fe72e475e30039bbfb")!)
      Text("Adapters stay in place. Target shapes and content identity are checked before generation; unsupported ConvRot/GPTQ formats are not selectable.").font(.caption2).foregroundStyle(.secondary)
    }
  }
  private func edit(_ body: (inout DrawThingsImageDraft) throws -> Void) {
    guard var draft = store.imageDraft else { return }
    do { try body(&draft); store.imageDraft = draft; store.imageEstimate = nil } catch { store.error = error.localizedDescription }
  }
  private func apply(_ members: [LoRAMember], mode: LoRAGroupApplicationMode = .add) {
    edit { try $0.applyMOSSLoRAs(members, mode: mode) }
  }
}
