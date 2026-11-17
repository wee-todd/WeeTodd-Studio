import AppKit
import ImageLayout
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct ImageLayoutEditor: View {
  @Binding var layout: ImageLayoutDraft
  var expandedPresentation = false
  var initialAnalysisBefore: ImageLayoutDraft? = nil
  var onAnalysisRegistered: (() -> Void)? = nil
  @State private var selection: UUID?
  @State private var drawing = false
  @State private var newType: ImageLayoutRegionType = .object
  @State private var zoom = 1.0
  @State private var history = ImageLayoutHistory()
  @State private var dragStart: ImageLayoutDraft?
  @State private var dragRegion: UUID?
  @State private var dragHandle: LayoutResizeHandle?
  @State private var expanded = false
  @State private var beforeExpanded: ImageLayoutDraft?
  @State private var failure: String?
  @FocusState private var canvasFocused: Bool

  private var selected: ImageLayoutRegion? { layout.regions.first { $0.id == selection } }
  private var compiled: String? { try? MOSSLayoutCompiler.compile(layout).effectivePrompt }
  var body: some View {
    VStack(spacing: 8) {
      toolbar
      HSplitView {
        GeometryReader { geometry in
          if let viewport = try? LayoutViewport(canvasWidth: layout.width, canvasHeight: layout.height,
            viewWidth: geometry.size.width, viewHeight: geometry.size.height, zoom: zoom) {
            canvas(viewport: viewport)
          }
        }.frame(minWidth: 250, minHeight: 240)
        inspector.frame(minWidth: 200, idealWidth: 230, maxWidth: 280)
      }
      if let failure { Text(failure).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
    }
    .sheet(isPresented: $expanded, onDismiss: {
      if let beforeExpanded { history.record(before: beforeExpanded, after: layout) }
      beforeExpanded = nil
    }) {
      VStack {
        HStack { Text("Layout canvas").font(.headline); Spacer(); Button("Done") { expanded = false }.keyboardShortcut(.cancelAction) }.padding()
        ImageLayoutEditor(layout: $layout, expandedPresentation: true).padding()
      }.frame(minWidth: 1000, minHeight: 700)
    }
    .onAppear { registerAnalysisUndo() }
    .onChange(of: initialAnalysisBefore) { _, _ in registerAnalysisUndo() }
    .onChange(of: layout) { before, after in history.observeAnalysisReplacement(from: before, to: after) }
  }
  private func registerAnalysisUndo() {
    guard let before = initialAnalysisBefore else { return }
    history.observeAnalysisReplacement(from: before, to: layout)
    onAnalysisRegistered?()
  }
  private var toolbar: some View {
    HStack(spacing: 8) {
      Picker("Tool", selection: $drawing) { Text("Select").tag(false); Text("Draw").tag(true) }
        .labelsHidden().pickerStyle(.segmented).frame(width: 120)
      Picker("Region", selection: $newType) {
        Text("Object").tag(ImageLayoutRegionType.object); Text("Text").tag(ImageLayoutRegionType.text); Text("Character").tag(ImageLayoutRegionType.character)
      }.labelsHidden().frame(width: 105)
      Button { duplicate() } label: { Image(systemName: "plus.square.on.square") }.help("Duplicate selected region").disabled(selected == nil)
      Button { removeSelected() } label: { Image(systemName: "trash") }.help("Delete selected region").disabled(selected == nil)
      Button { if let previous = history.undo(current: layout) { layout = previous } } label: { Image(systemName: "arrow.uturn.backward") }
        .help("Undo").keyboardShortcut("z", modifiers: .command).disabled(!history.canUndo)
      Button { if let next = history.redo(current: layout) { layout = next } } label: { Image(systemName: "arrow.uturn.forward") }
        .help("Redo").keyboardShortcut("z", modifiers: [.command, .shift]).disabled(!history.canRedo)
      Menu {
        Button("Import layout or workflow…") { importLayout() }
        Button("Export editable layout…") { exportLayout() }
        Button("Copy MOSS prompt") { copyPrompt() }.disabled(compiled == nil)
        Divider()
        Button("Load guide image…") { chooseGuide() }
        if layout.guideImagePath != nil { Button("Remove guide") { edit { $0.guideImagePath = nil } } }
        Divider()
        Button("Clear regions") { edit { $0.regions = [] }; selection = nil }.disabled(layout.regions.isEmpty)
      } label: { Image(systemName: "ellipsis.circle") }.help("Layout actions")
      Spacer(minLength: 0)
      Button("Fit") { zoom = 1 }
      Slider(value: $zoom, in: 0.25...3).frame(width: 65).help("Canvas zoom")
      if !expandedPresentation {
        Button { beforeExpanded = layout; expanded = true } label: { Image(systemName: "arrow.up.left.and.arrow.down.right") }.help("Expand layout editor")
      }
    }.controlSize(.small)
  }
  private func canvas(viewport v: LayoutViewport) -> some View {
    ZStack(alignment: .topLeading) {
      Color.black.opacity(0.85)
      Rectangle().fill(.white).frame(width: v.canvasWidth * v.scale, height: v.canvasHeight * v.scale)
        .offset(x: v.originX, y: v.originY)
      if let path = layout.guideImagePath {
        CachedImageThumbnail(path: path, maximumPixelSize: 2048)
          .aspectRatio(contentMode: .fit).frame(width: v.canvasWidth * v.scale, height: v.canvasHeight * v.scale)
          .opacity(layout.guideOpacity ?? 0.5).offset(x: v.originX, y: v.originY).allowsHitTesting(false)
      }
      ForEach(Array(layout.regions.enumerated()), id: \.element.id) { index, region in
        let origin = v.viewPoint(x: region.bounds.x, y: region.bounds.y)
        let width = region.bounds.width * v.canvasWidth * v.scale, height = region.bounds.height * v.canvasHeight * v.scale
        Rectangle().fill(color(region.type).opacity(0.08)).overlay(alignment: .topLeading) {
          Text("\(index + 1) · \(region.type == .text ? region.text ?? "Text" : region.caption)")
            .font(.system(size: 11, weight: .medium)).lineLimit(2).padding(3)
            .background(color(region.type).opacity(0.9)).foregroundStyle(.white)
        }.overlay(Rectangle().stroke(color(region.type), lineWidth: region.id == selection ? 2 : 1))
          .frame(width: width, height: height).offset(x: origin.x, y: origin.y)
          .allowsHitTesting(false)
        if region.id == selection {
          ForEach(LayoutResizeHandle.allCases, id: \.rawValue) { handle in
            let point = handlePoint(handle, rect: region.bounds, viewport: v)
            Rectangle().fill(.white).overlay(Rectangle().stroke(color(region.type)))
              .frame(width: 7, height: 7).offset(x: point.x - 3.5, y: point.y - 3.5).allowsHitTesting(false)
          }
        }
      }
    }.clipped().contentShape(Rectangle()).focusable().focusEffectDisabled().focused($canvasFocused)
      .accessibilityLabel("Layout canvas. Draw regions or select and move them with arrow keys.")
      .gesture(DragGesture(minimumDistance: 0).onChanged { value in
        if dragStart == nil { beginDrag(at: value.startLocation, viewport: v) }
        updateDrag(value, viewport: v)
      }.onEnded { value in
        if dragStart == nil { beginDrag(at: value.startLocation, viewport: v) }
        updateDrag(value, viewport: v)
        if let dragStart { history.record(before: dragStart, after: layout) }
        dragStart = nil; dragRegion = nil; dragHandle = nil
      })
      .onKeyPress(phases: .down) { key in
        guard selected != nil else { return .ignored }
        if key.key == .delete || ["\u{7f}", "\u{8}", "\u{f728}"].contains(key.characters) { removeSelected(); return .handled }
        guard [KeyEquivalent.leftArrow, .rightArrow, .upArrow, .downArrow].contains(key.key) else { return .ignored }
        let distance = key.modifiers.contains(.shift) ? 10.0 : 1.0
        let dx = key.key == .leftArrow ? -distance : key.key == .rightArrow ? distance : 0
        let dy = key.key == .upArrow ? -distance : key.key == .downArrow ? distance : 0
        editRegion { $0.bounds = ImageLayoutEditing.move($0.bounds, dx: dx / Double(layout.width), dy: dy / Double(layout.height)) }
        return .handled
      }
  }
  private var inspector: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("REGIONS · \(layout.regions.count)").font(.caption.bold())
      List(selection: $selection) {
        ForEach(Array(layout.regions.enumerated()), id: \.element.id) { index, region in
          Label("\(index + 1). \(region.type == .text ? region.text ?? "Text" : region.caption)", systemImage: region.type == .text ? "textformat" : region.type == .character ? "person" : "square")
            .font(.caption).lineLimit(2).padding(.leading, region.parentID == nil ? 0 : 12).tag(region.id)
        }
      }.frame(minHeight: 70, idealHeight: 115, maxHeight: 150)
      if let region = selected {
        if let parent = region.parentID, let index = layout.regions.firstIndex(where: { $0.id == parent }) {
          Button("Part of region \(index + 1) · Select parent") { selection = parent }.font(.caption2)
        } else {
          let count = ImageLayoutEditing.descendants(of: region.id, in: layout).count - 1
          if count > 0 { Text("Moving or resizing this subject also transforms its \(count) detail boxes.").font(.caption2).foregroundStyle(.secondary) }
        }
        ImageLayoutInspector(region: region, canvasWidth: layout.width, canvasHeight: layout.height) { replacement in
          editRegion { $0 = replacement }
        }
      } else {
        Text("Draw a box for each object, character or line of text. Select a region to edit its words and bounds.")
          .font(.caption).foregroundStyle(.secondary)
      }
      if layout.guideImagePath != nil {
        HStack { Text("Guide opacity").font(.caption); Slider(value: Binding(get: { layout.guideOpacity ?? 0.5 }, set: { value in edit { $0.guideOpacity = value } }), in: 0...1) }
        Text("Guide image is an editor overlay.").font(.caption2).foregroundStyle(.secondary)
      }
      if let source = layout.sourceReconstruction {
        HStack {
          CachedImageThumbnail(path: source.path, maximumPixelSize: 96).aspectRatio(contentMode: .fit).frame(width: 45, height: 45)
          VStack(alignment: .leading) {
            Text("Generation source").font(.caption.bold())
            Text(URL(fileURLWithPath: source.path).lastPathComponent).font(.caption2).lineLimit(1).help(source.path)
          }
        }
        Toggle("Source reconstruction", isOn: Binding(get: { layout.sourceReconstruction?.enabled ?? false }, set: { value in edit { $0.sourceReconstruction?.enabled = value } }))
          .font(.caption)
        if source.enabled {
          HStack {
            Text("Change strength"); Text(source.noiseLevel, format: .number.precision(.fractionLength(2)))
          }.font(.caption)
          Slider(value: Binding(get: { layout.sourceReconstruction?.noiseLevel ?? 0.08 }, set: { value in edit { $0.sourceReconstruction?.noiseLevel = value } }), in: 0.01...0.95, step: 0.01)
          Text("Lower strength preserves the source and limits large edits. Reanalyze to replace this source; loading a guide changes only the overlay.")
            .font(.caption2).foregroundStyle(.secondary)
        }
      }
      DisclosureGroup("MOSS prompt") {
        ScrollView { Text(compiled ?? "Add a scene description and complete the regions.").font(.system(size: 10, design: .monospaced)).textSelection(.enabled) }.frame(maxHeight: 100)
      }.font(.caption)
      if compiled == nil, !layout.summary.caption.isEmpty {
        Text((try? MOSSLayoutCompiler.compile(layout)) == nil ? validationMessage() : "").font(.caption2).foregroundStyle(.orange)
      }
      Spacer(minLength: 0)
    }.padding(.leading, 8)
  }
  private func color(_ type: ImageLayoutRegionType) -> Color { type == .text ? .blue : type == .character ? .orange : .green }
  private func edit(_ body: (inout ImageLayoutDraft) -> Void) {
    let before = layout; var after = before; body(&after); history.record(before: before, after: after); layout = after
  }
  private func editRegion(_ body: (inout ImageLayoutRegion) -> Void) {
    guard var replacement = selected else { return }
    body(&replacement)
    do {
      let after = try ImageLayoutEditing.replacing(replacement, in: layout)
      history.record(before: layout, after: after); layout = after
    } catch { failure = error.localizedDescription }
  }
  private func duplicate() {
    guard var region = selected, layout.regions.count < 128 else { return }
    region.id = UUID(); region.bounds = ImageLayoutEditing.move(region.bounds, dx: 0.02, dy: 0.02)
    edit { $0.regions.append(region) }; selection = region.id
  }
  private func removeSelected() {
    guard let selection else { return }
    let removed = ImageLayoutEditing.descendants(of: selection, in: layout)
    edit { $0.regions.removeAll { removed.contains($0.id) } }; self.selection = nil
  }
  private func handlePoint(_ handle: LayoutResizeHandle, rect: NormalizedLayoutRect, viewport: LayoutViewport) -> LayoutPoint {
    let x = handle.changesLeft ? rect.x : handle.changesRight ? rect.x + rect.width : rect.x + rect.width / 2
    let y = handle.changesTop ? rect.y : handle.changesBottom ? rect.y + rect.height : rect.y + rect.height / 2
    return viewport.viewPoint(x: x, y: y)
  }
  private func beginDrag(at location: CGPoint, viewport v: LayoutViewport) {
    canvasFocused = true; dragStart = layout
    let point = v.normalizedPoint(x: location.x, y: location.y)
    guard (0...1).contains(point.x), (0...1).contains(point.y) else { return }
    if drawing {
      guard layout.regions.count < 128 else { return }
      let region = ImageLayoutRegion(type: newType, caption: newType == .text ? "Describe the lettering" : newType == .character ? "Describe the character" : "Describe the object",
        text: newType == .text ? "TEXT" : nil, bounds: .init(x: min(point.x, 1 - 1 / Double(layout.width)), y: min(point.y, 1 - 1 / Double(layout.height)), width: 1 / Double(layout.width), height: 1 / Double(layout.height)))
      var next = layout; next.regions.append(region); layout = next; selection = region.id; dragRegion = region.id
      return
    }
    if let selected, let handle = LayoutResizeHandle.allCases.first(where: { handle in
      let p = handlePoint(handle, rect: selected.bounds, viewport: v)
      return abs(p.x - location.x) <= 7 && abs(p.y - location.y) <= 7
    }) { dragRegion = selected.id; dragHandle = handle; return }
    let hit = layout.regions.reversed().first { r in point.x >= r.bounds.x && point.x <= r.bounds.x + r.bounds.width && point.y >= r.bounds.y && point.y <= r.bounds.y + r.bounds.height }
    selection = hit?.id; dragRegion = hit?.id
  }
  private func updateDrag(_ value: DragGesture.Value, viewport v: LayoutViewport) {
    guard let dragRegion, let index = layout.regions.firstIndex(where: { $0.id == dragRegion }) else { return }
    var next = layout
    if drawing {
      let start = v.normalizedPoint(x: value.startLocation.x, y: value.startLocation.y)
      let end = v.normalizedPoint(x: value.location.x, y: value.location.y)
      let x0 = max(0, min(1 - 1 / Double(layout.width), min(start.x, end.x)))
      let y0 = max(0, min(1 - 1 / Double(layout.height), min(start.y, end.y)))
      next.regions[index].bounds = .init(x: x0, y: y0,
        width: max(1 / Double(layout.width), min(1, max(start.x, end.x)) - x0),
        height: max(1 / Double(layout.height), min(1, max(start.y, end.y)) - y0))
    } else if let start = dragStart, var original = start.regions.first(where: { $0.id == dragRegion }) {
      let dx = value.translation.width / (v.canvasWidth * v.scale), dy = value.translation.height / (v.canvasHeight * v.scale)
      if let dragHandle { original.bounds = ImageLayoutEditing.resize(original.bounds, handle: dragHandle, dx: dx, dy: dy, minimumWidth: 1 / Double(layout.width), minimumHeight: 1 / Double(layout.height)) }
      else { original.bounds = ImageLayoutEditing.move(original.bounds, dx: dx, dy: dy) }
      do { next = try ImageLayoutEditing.replacing(original, in: start) }
      catch { failure = error.localizedDescription; return }
    }
    layout = next
  }
  private func validationMessage() -> String {
    do { _ = try MOSSLayoutCompiler.compile(layout); return "" } catch { return error.localizedDescription }
  }
  private func importLayout() {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    do {
      let file = try FileHandle(forReadingFrom: url); defer { try? file.close() }
      let data = try file.read(upToCount: 2 * 1024 * 1024 + 1) ?? Data()
      guard data.count <= 2 * 1024 * 1024 else { throw ImageLayoutError.invalid("Layout import exceeds 2 MiB.") }
      let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
      let imported: ImageLayoutDraft
      if root?["format"] as? String == "weetodd-image-layout-v1" { imported = try MOSSLayoutInterchange.importEditableDraft(data) }
      else if root?["nodes"] != nil { imported = try MOSSLayoutInterchange.importPixaromaWorkflow(data) }
      else if root?["bbox_coord_space"] as? String == "abs_centralized" { imported = try MOSSLayoutInterchange.importCompiled(data, width: layout.width, height: layout.height) }
      else { imported = try MOSSLayoutInterchange.importOfficialDraft(data) }
      edit { $0 = imported }; selection = imported.regions.first?.id; failure = nil
    } catch { failure = error.localizedDescription }
  }
  private func exportLayout() {
    let panel = NSSavePanel(); panel.allowedContentTypes = [.json]; panel.nameFieldStringValue = "MOSS layout.json"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    do { try MOSSLayoutInterchange.exportEditableDraft(layout).write(to: url, options: .atomic); failure = nil } catch { failure = error.localizedDescription }
  }
  private func copyPrompt() { guard let compiled else { return }; NSPasteboard.general.clearContents(); NSPasteboard.general.setString(compiled, forType: .string) }
  private func chooseGuide() {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]
    if panel.runModal() == .OK, let url = panel.url { edit { $0.guideImagePath = url.path; $0.guideOpacity = 0.5 } }
  }
}
