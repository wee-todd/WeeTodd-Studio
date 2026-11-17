import ImageLayout
import SwiftUI

struct ImageLayoutInspector: View {
  let region: ImageLayoutRegion
  let canvasWidth, canvasHeight: Int
  let onChange: (ImageLayoutRegion) -> Void
  private func binding<T>(_ key: WritableKeyPath<ImageLayoutRegion, T>) -> Binding<T> {
    Binding(get: { region[keyPath: key] }, set: { value in var copy = region; copy[keyPath: key] = value; onChange(copy) })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Picker("Type", selection: Binding(get: { region.type }, set: { type in
        var copy = region; copy.type = type
        if type == .text && copy.text == nil { copy.text = "TEXT" }
        if type != .text { copy.text = nil }
        onChange(copy)
      })) {
        Text("Object").tag(ImageLayoutRegionType.object); Text("Text").tag(ImageLayoutRegionType.text); Text("Character").tag(ImageLayoutRegionType.character)
      }
      Text("Description").font(.caption.bold())
      TextField("Appearance and details", text: binding(\.caption), axis: .vertical).lineLimit(2...5).textFieldStyle(.roundedBorder)
      if region.type == .text {
        Text("Exact words").font(.caption.bold())
        TextEditor(text: Binding(get: { region.text ?? "" }, set: { text in var copy = region; copy.text = text; onChange(copy) }))
          .font(.system(size: 12)).frame(height: 65).overlay(Rectangle().stroke(.secondary.opacity(0.3)))
      }
      Grid(alignment: .leading) {
        GridRow { number("X", field: \.x, dimension: canvasWidth); number("Y", field: \.y, dimension: canvasHeight) }
        GridRow { number("Width", field: \.width, dimension: canvasWidth); number("Height", field: \.height, dimension: canvasHeight) }
      }
      Text("Pixel bounds · arrows move 1 px; Shift moves 10 px. Region colors are editor guides.").font(.caption2).foregroundStyle(.secondary)
      if region.tags != nil || region.palette != nil {
        DisclosureGroup("Imported tags & palette") {
          if let tags = region.tags { Text("Tags: " + tags.joined(separator: ", ")).textSelection(.enabled) }
          if let palette = region.palette { Text("Palette: " + palette.joined(separator: ", ")).textSelection(.enabled) }
        }.font(.caption2)
      }
    }.font(.caption)
  }
  private func number(_ label: String, field: WritableKeyPath<NormalizedLayoutRect, Double>, dimension: Int) -> some View {
    VStack(alignment: .leading, spacing: 2) {
      Text(label).font(.caption2)
      TextField(label, value: Binding(get: { region.bounds[keyPath: field] * Double(dimension) }, set: { pixels in
        guard pixels.isFinite else { return }
        var copy = region; copy.bounds[keyPath: field] = pixels / Double(dimension)
        copy.bounds.x = max(0, min(1 - 1 / Double(canvasWidth), copy.bounds.x))
        copy.bounds.y = max(0, min(1 - 1 / Double(canvasHeight), copy.bounds.y))
        copy.bounds.width = max(1 / Double(canvasWidth), min(1 - copy.bounds.x, copy.bounds.width))
        copy.bounds.height = max(1 / Double(canvasHeight), min(1 - copy.bounds.y, copy.bounds.height))
        onChange(copy)
      }), format: .number.grouping(.never).precision(.fractionLength(0...2))).textFieldStyle(.roundedBorder)
    }
  }
}
