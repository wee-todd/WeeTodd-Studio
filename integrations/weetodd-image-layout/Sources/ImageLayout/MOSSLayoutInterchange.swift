import Foundation

public enum MOSSLayoutInterchange {
  private struct EditableDraft: Codable {
    var format: String
    var layout: ImageLayoutDraft
  }
  /// Native editing preserves IDs, ownership and provenance; official export remains interoperable.
  public static func exportEditableDraft(_ draft: ImageLayoutDraft) throws -> Data {
    try MOSSLayoutCompiler.validate(draft)
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return try encoder.encode(EditableDraft(format: "weetodd-image-layout-v1", layout: draft))
  }
  public static func importEditableDraft(_ data: Data) throws -> ImageLayoutDraft {
    let root = try LayoutJSON.parse(data)
    try fields(root, allowed: ["format", "layout"], required: ["format", "layout"])
    guard root["format"]?.string == "weetodd-image-layout-v1" else {
      throw ImageLayoutError.invalid("Unsupported editable layout version.")
    }
    let draft = try JSONDecoder().decode(EditableDraft.self, from: data).layout
    try MOSSLayoutCompiler.validate(draft)
    return draft
  }
  public static func importOfficialDraft(_ data: Data) throws -> ImageLayoutDraft {
    let root = try LayoutJSON.parse(data)
    if let caption = root["caption"]?.string {
      try fields(root, allowed: ["caption"], required: ["caption"])
      let draft = ImageLayoutDraft(width: 1024, height: 1024, caption: caption)
      try MOSSLayoutCompiler.validate(draft); return draft
    }
    try fields(root, allowed: ["width", "height", "bbox_coord_space", "bbox_format", "summary", "items"],
      required: ["width", "height", "bbox_coord_space", "bbox_format", "summary", "items"])
    guard let width = root["width"]?.integer, let height = root["height"]?.integer,
      root["bbox_coord_space"]?.string == "abs", root["bbox_format"]?.string == "xyxy" else {
      throw ImageLayoutError.invalid("Editable drafts require integer dimensions, abs coordinates and xyxy boxes.")
    }
    return try decode(root, width: width, height: height, centered: false)
  }
  public static func importCompiled(_ data: Data, width: Int, height: Int) throws -> ImageLayoutDraft {
    let root = try LayoutJSON.parse(data)
    try fields(root, allowed: ["summary", "items", "bbox_coord_space", "bbox_format"], required: ["summary", "items", "bbox_coord_space", "bbox_format"])
    guard root["bbox_coord_space"]?.string == "abs_centralized", root["bbox_format"]?.string == "yxyx" else {
      throw ImageLayoutError.invalid("Compiled layouts require abs_centralized yxyx coordinates and matching dimensions.")
    }
    return try decode(root, width: width, height: height, centered: true)
  }
  public static func exportOfficialDraft(_ draft: ImageLayoutDraft) throws -> Data {
    try MOSSLayoutCompiler.validate(draft)
    if draft.regions.isEmpty { return Data(("{\"caption\":" + MOSSLayoutCompiler.quote(draft.summary.caption) + "}").utf8) }
    let items = draft.regions.map { region -> String in
      let r = region.bounds, w = Double(draft.width), h = Double(draft.height)
      let box = [r.x * w, r.y * h, (r.x + r.width) * w, (r.y + r.height) * h]
      let numbers = box.map { String($0) }.joined(separator: ",")
      return MOSSLayoutCompiler.itemJSON(region, box: "[" + numbers + "]")
    }
    let text = "{\"width\":\(draft.width),\"height\":\(draft.height),\"bbox_coord_space\":\"abs\",\"bbox_format\":\"xyxy\",\"summary\":"
      + MOSSLayoutCompiler.summaryJSON(draft.summary) + ",\"items\":[" + items.joined(separator: ",") + "]}"
    return Data(text.utf8)
  }
  public static func importPixaromaWorkflow(_ data: Data) throws -> ImageLayoutDraft {
    guard let nodes = try LayoutJSON.parse(data)["nodes"]?.array else { throw ImageLayoutError.invalid("Workflow has no nodes.") }
    let layouts = nodes.filter { $0["type"]?.string == "PixaromaLayoutBoxes" }
    guard layouts.count == 1, let node = layouts.first, let widgets = node["widgets_values"]?.array, widgets.count >= 2,
      let width = widgets[0].integer, let height = widgets[1].integer,
      let boxes = node["properties"]?["layoutBoxes"]?["boxes"]?.array else {
      throw ImageLayoutError.invalid("Choose a workflow with one saved Pixaroma layout canvas.")
    }
    let promptNodes = nodes.filter { $0["type"]?.string == "PixaromaPrompt" }
    guard promptNodes.count == 1, let promptNode = promptNodes.first else { throw ImageLayoutError.invalid("Workflow prompt is ambiguous.") }
    let caption = promptNode["properties"]?["promptState"]?["text"]?.string
      ?? promptNode["widgets_values"]?.array?.first?.string ?? ""
    let characterText = nodes.first { $0["type"]?.string == "MOSSImageLayoutPrompt" }?["widgets_values"]?.array?.last?.string ?? ""
    let characters = Set(characterText.split { !$0.isNumber }.compactMap { Int($0) })
    let regions = try boxes.enumerated().map { index, box -> ImageLayoutRegion in
      guard let x = box["x"]?.double, let y = box["y"]?.double, let w = box["w"]?.double,
        let h = box["h"]?.double, let type = box["type"]?.string, ["obj", "text", "character"].contains(type),
        let description = box["desc"]?.string else { throw ImageLayoutError.invalid("Invalid saved workflow box.") }
      let label: ImageLayoutRegionType = type == "text" ? .text : (type == "character" || characters.contains(index + 1) ? .character : .object)
      return .init(type: label, caption: description, text: label == .text ? box["text"]?.string : nil,
        bounds: .init(x: x, y: y, width: w, height: h))
    }
    let draft = ImageLayoutDraft(width: width, height: height, caption: caption, regions: regions)
    try MOSSLayoutCompiler.validate(draft); return draft
  }
  private static func decode(_ root: LayoutJSON, width: Int, height: Int, centered: Bool) throws -> ImageLayoutDraft {
    try ImageLayoutGeometry.validateCanvas(width: width, height: height)
    guard let summary = root["summary"], let caption = summary["caption"]?.string,
      let items = root["items"]?.array, !items.isEmpty, items.count <= 128 else { throw ImageLayoutError.invalid("Layout needs a scene description and 1–128 items.") }
    try fields(summary, allowed: ["caption", "meta", "canvas", "palette"], required: ["caption"])
    var draft = ImageLayoutDraft(width: width, height: height, caption: caption)
    draft.summary = .init(caption: caption, meta: try strings(summary["meta"]),
      canvas: try optionalString(summary["canvas"]), palette: try strings(summary["palette"]))
    draft.regions = try items.map { item in
      try fields(item, allowed: ["label", "caption", "text", "bbox", "tags", "palette"], required: ["label", "caption", "bbox"])
      guard let label = item["label"]?.string.flatMap(ImageLayoutRegionType.init(rawValue:)),
        let caption = item["caption"]?.string, let values = item["bbox"]?.array, values.count == 4 else { throw ImageLayoutError.invalid("Invalid layout item.") }
      let numbers = values.compactMap(\.double)
      guard numbers.count == 4 else { throw ImageLayoutError.invalid("Box edges must be finite numbers.") }
      let x0 = centered ? numbers[1] + Double(width / 2) : numbers[0]
      let y0 = centered ? numbers[0] + Double(height / 2) : numbers[1]
      let x1 = centered ? numbers[3] + Double(width / 2) : numbers[2]
      let y1 = centered ? numbers[2] + Double(height / 2) : numbers[3]
      return ImageLayoutRegion(type: label, caption: caption, text: try optionalString(item["text"]),
        bounds: .init(x: x0 / Double(width), y: y0 / Double(height), width: (x1 - x0) / Double(width), height: (y1 - y0) / Double(height)),
        tags: try strings(item["tags"]), palette: try strings(item["palette"]))
    }
    try MOSSLayoutCompiler.validate(draft); return draft
  }
  private static func fields(_ json: LayoutJSON, allowed: Set<String>, required: Set<String>) throws {
    guard let object = json.object, Set(object.keys).isSubset(of: allowed), required.isSubset(of: Set(object.keys)) else {
      throw ImageLayoutError.invalid("Layout has missing or unsupported fields.")
    }
  }
  private static func optionalString(_ value: LayoutJSON?) throws -> String? {
    guard let value else { return nil }
    guard let text = value.string else { throw ImageLayoutError.invalid("Expected layout text.") }
    return text
  }
  private static func strings(_ value: LayoutJSON?) throws -> [String]? {
    guard let value else { return nil }
    guard let array = value.array else { throw ImageLayoutError.invalid("Expected an array of text.") }
    let strings = array.compactMap(\.string)
    guard strings.count == array.count else { throw ImageLayoutError.invalid("Expected an array of text.") }
    return strings
  }
}
