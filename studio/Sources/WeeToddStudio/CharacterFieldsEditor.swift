import StudioCore
import SwiftUI

struct CharacterFieldsEditor: View {
  @ObservedObject var controller: CharacterSheetSessionController
  @State private var search = ""
  @State private var compact = false
  @State private var hideEmpty = false

  private let sectionNames = [
    1: "Required layout", 2: "Character identity", 3: "Body / proportions",
    4: "Face / head", 5: "Hair", 6: "Clothing",
    7: "Accessories / distinctive features", 8: "Materials / surface detail",
    9: "Style / rendering", 10: "Camera / lens", 11: "Lighting",
    12: "Turnaround consistency", 13: "Composition / framing"
  ]

  var body: some View {
    VStack(spacing: 0) {
      HStack {
        TextField("Search character fields", text: $search).textFieldStyle(.roundedBorder)
        Toggle("Compact", isOn: $compact).toggleStyle(.checkbox)
        Toggle("Hide empty groups", isOn: $hideEmpty).toggleStyle(.checkbox)
      }.padding(.bottom, 10)
      ForEach(1...13, id: \.self) { section in
        let definitions = visibleDefinitions(section)
        if shouldShow(section, definitions: definitions) {
          DisclosureGroup(sectionNames[section] ?? "Section \(section)") {
            if section == 1 { immutable("4-view turnaround: front, side, back, facial close-up") }
            else if section == 12 { immutable("One consistent identity, proportions, clothing, materials and local colors across all views") }
            else if section == 13 { immutable("One row of four distinct panels on a plain solid white background") }
            else if section == 9 { presetPicker }
            ForEach(definitions, id: \.key) { field in
              if field.key != "style.presetID" { fieldRow(field, path: field.key) }
            }
            if section == 6 { repeatable("garments", records: controller.document.definition.appearance.orderedGarments) }
            if section == 7 {
              repeatable("accessories", records: controller.document.definition.appearance.orderedAccessories)
              repeatable("features", records: controller.document.definition.appearance.orderedFeatures)
            }
            if section == 8 { repeatable("surfaces", records: controller.document.definition.appearance.orderedSurfaces) }
          }.padding(.vertical, 5)
        }
      }
    }
  }

  private var presetPicker: some View {
    HStack {
      Text("Preset").frame(width: compact ? 130 : 190, alignment: .leading)
      Picker("Preset", selection: Binding(get: { controller.document.definition.settings.stylePresetID },
        set: controller.selectStyle)) {
        ForEach(CharacterStylePresetRegistry.all) { Text($0.label).tag($0.id) }
      }.labelsHidden().frame(maxWidth: 320)
      Text("Required").font(.caption).foregroundStyle(.secondary)
      Spacer()
    }
  }

  @ViewBuilder private func repeatable(_ collection: String, records: [CharacterRepeatableRecord]) -> some View {
    Divider().padding(.top, 6)
    HStack { Text(collection.capitalized).font(.headline); Spacer(); Button("Add") { controller.addRecord(collection) } }
    ForEach(records) { record in
      GroupBox("\(collection.dropLast().capitalized) \(record.order + 1)") {
        let definitions = CharacterFieldCatalog.shared.fields.filter { $0.key.hasPrefix("\(collection)[].") }
        ForEach(definitions, id: \.key) { definition in
          let suffix = String(definition.key.split(separator: ".").last ?? "")
          fieldRow(definition, path: "\(collection)[\(record.id.uuidString)].\(suffix)")
        }
        HStack { Spacer(); Button("Remove", role: .destructive) { controller.removeRecord(collection, id: record.id) } }
      }.padding(.vertical, 4)
    }
  }

  private func fieldRow(_ field: CharacterFieldDefinition, path: String) -> some View {
    let entry = controller.document.definition.entry(at: path)
    return VStack(alignment: .leading, spacing: 3) {
      HStack(alignment: .firstTextBaseline) {
        Text(field.label.replacingOccurrences(of: #"([a-z])([A-Z])"#, with: "$1 $2", options: .regularExpression).capitalized)
          .frame(width: compact ? 130 : 190, alignment: .leading)
        TextField("Optional", text: Binding(get: { entry?.displayString ?? "" }, set: { controller.setText(path, $0) }))
          .textFieldStyle(.roundedBorder)
        if !field.suggestions.isEmpty {
          Menu {
            ForEach(field.suggestions, id: \.self) { value in Button(value) { controller.setText(path, value) } }
          } label: { Image(systemName: "chevron.down.circle") }.help("Common values; custom text remains allowed")
        }
        Picker("State", selection: Binding(get: { entry?.state ?? .unspecified }, set: { controller.setState(path, $0) })) {
          Text("Unspecified").tag(CharacterFieldState.unspecified)
          Text("Value").tag(CharacterFieldState.value)
          if field.canBeAbsent { Text("Absent").tag(CharacterFieldState.explicitlyAbsent) }
          if !field.applicability.isEmpty { Text("N/A").tag(CharacterFieldState.notApplicable) }
        }.labelsHidden().frame(width: 110)
        Toggle("Required", isOn: Binding(get: {
          controller.document.definition.settings.requiredFieldPaths.contains(path)
        }, set: { controller.setRequired(path, $0) })).toggleStyle(.checkbox)
      }
      if let evidence = entry?.evidence {
        Text("\(entry?.source.rawValue ?? "") · \(evidence.summary)")
          .font(.caption2).foregroundStyle(.secondary).padding(.leading, compact ? 138 : 198)
      }
    }
  }

  private func immutable(_ text: String) -> some View {
    Label(text, systemImage: "lock.fill").font(.callout).foregroundStyle(.secondary)
      .frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 5)
  }

  private func visibleDefinitions(_ section: Int) -> [CharacterFieldDefinition] {
    CharacterFieldCatalog.shared.fields.filter {
      $0.section == section && (search.isEmpty || ($0.label + " " + $0.key).localizedCaseInsensitiveContains(search))
        && !$0.key.contains("[]")
    }
  }
  private func shouldShow(_ section: Int, definitions: [CharacterFieldDefinition]) -> Bool {
    if [1, 12, 13].contains(section) { return search.isEmpty }
    if !search.isEmpty { return !definitions.isEmpty }
    if !hideEmpty { return true }
    return definitions.contains { controller.document.definition.entry(at: $0.key)?.state != .unspecified }
      || [6, 7, 8, 9].contains(section)
  }
}
