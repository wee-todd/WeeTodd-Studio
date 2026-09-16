import AppKit
import StudioCore
import SwiftUI

/// Thumbnail-sized in its parent, with an accessible full-image inspection action.
struct PreviewableImage: View {
  let path: String
  var title: String? = nil
  @State private var selection: ImagePreviewSelection?

  var body: some View {
    Button { selection = ImagePreviewSelection(path: path, title: title) } label: {
      WorkspaceImage(path: path).contentShape(Rectangle())
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Preview \(title ?? URL(fileURLWithPath: path).lastPathComponent)")
    .accessibilityIdentifier("image-preview:" + ImagePreviewSelection(path: path).id + ":" + (title ?? "image"))
    .help("Click to preview the original image")
    .sheet(item: $selection) { item in
      ImagePreview(path: item.path, title: item.title).id(item.id)
    }
  }
}

/// Loads the original file only while open; thumbnail decoding remains independent.
struct ImagePreview: View {
  let path: String
  var title: String? = nil
  @Environment(\.dismiss) private var dismiss
  @Environment(\.displayScale) private var displayScale
  @State private var image: NSImage?
  @State private var loading = true
  @State private var fit = true
  @State private var zoom = 1.0

  private var pixels: CGSize {
    guard let image else { return .zero }
    let rep = image.representations.max { $0.pixelsWide < $1.pixelsWide }
    return CGSize(width: rep?.pixelsWide ?? Int(image.size.width), height: rep?.pixelsHigh ?? Int(image.size.height))
  }
  var body: some View {
    VStack(spacing: 0) {
      HStack {
        VStack(alignment: .leading, spacing: 3) {
          Text(title ?? URL(fileURLWithPath: path).lastPathComponent).font(.headline).lineLimit(1)
          if image != nil { Text("\(Int(pixels.width)) × \(Int(pixels.height)) pixels · original image").font(.caption).foregroundStyle(.secondary) }
        }
        Spacer()
        Button("Fit") { fit = true }.disabled(image == nil)
        Button("Actual Size") { fit = false; zoom = 1 }.disabled(image == nil)
          .help("One image pixel per display pixel")
        Slider(value: Binding(get: { zoom }, set: { zoom = $0; fit = false }), in: 0.1...4)
          .frame(width: 150).disabled(image == nil).accessibilityLabel("Preview zoom")
        Text(fit ? "Fit" : "\(Int(zoom * 100))%")
          .monospacedDigit().frame(width: 48).accessibilityLabel("Zoom level")
        Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
      }.padding(16)
      Divider()
      GeometryReader { geometry in
        if let image {
          let actual = CGSize(width: pixels.width / displayScale, height: pixels.height / displayScale)
          let fitScale = min(max(1, geometry.size.width - 32) / max(1, actual.width),
                             max(1, geometry.size.height - 32) / max(1, actual.height))
          let scale = fit ? fitScale : zoom
          ScrollView([.horizontal, .vertical]) {
            Image(nsImage: image).resizable().interpolation(.high)
              .frame(width: actual.width * scale, height: actual.height * scale)
              .padding(16)
              .frame(minWidth: geometry.size.width, minHeight: geometry.size.height)
              .accessibilityLabel("Full image preview")
          }
        } else {
          VStack(spacing: 12) {
            if loading { ProgressView("Loading original image…") }
            else {
              Image(systemName: "photo.badge.exclamationmark").font(.largeTitle)
              Text("The image could not be opened.").font(.headline)
              Text("The file may have moved or become unavailable.").foregroundStyle(.secondary)
              Text(path).font(.caption).textSelection(.enabled)
            }
          }.frame(maxWidth: .infinity, maxHeight: .infinity).padding()
        }
      }.background(Color(nsColor: .underPageBackgroundColor))
    }
    .frame(minWidth: 900, idealWidth: 1100, minHeight: 640, idealHeight: 800)
    .task(id: path) {
      image = nil; loading = true; fit = true; zoom = 1
      let loaded = await Task.detached(priority: .userInitiated) { NSImage(contentsOfFile: path) }.value
      guard !Task.isCancelled else { return }
      image = loaded; loading = false
    }
  }
}
