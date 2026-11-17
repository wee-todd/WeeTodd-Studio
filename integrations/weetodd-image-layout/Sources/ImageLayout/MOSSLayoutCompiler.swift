import CryptoKit
import Foundation

public struct CompiledMOSSLayout: Equatable, Sendable {
  public let effectivePrompt: String
  public let width, height: Int
  public var contentIdentity: String {
    SHA256.hash(data: Data("\(width)x\(height)\n\(effectivePrompt)".utf8)).map { String(format: "%02x", $0) }.joined()
  }
}

public enum MOSSLayoutCompiler {
  public static func compile(_ draft: ImageLayoutDraft) throws -> CompiledMOSSLayout {
    try validate(draft)
    let dense = ["moss-image-layout-dense-v2","moss-image-layout-dense-v3","moss-image-layout-dense-v4"].contains(draft.analysis?.recipe ?? "")
    var summary = draft.summary
    if dense { summary.caption = ImageLayoutAnalysis.normalizingSpatialLanguage(summary.caption) }
    let prompt: String
    if draft.regions.isEmpty { prompt = draft.summary.caption }
    else {
      let items = try draft.regions.map { authored -> String in
        var region = authored
        if dense {
          region.caption = ImageLayoutAnalysis.normalizingSpatialLanguage(region.caption)
          if region.type == .text { region.caption = ImageLayoutTypography.editableStyle(region.caption) }
          if region.type != .text && region.parentID == nil && (region.type == .character || region.bounds.width*region.bounds.height < 0.45) {
            let x = region.bounds.x + region.bounds.width/2
            let location = x < 0.25 ? "far left side" : x < 0.38 ? "left side" : x > 0.75 ? "far right side" : x > 0.62 ? "right side" : "center"
            region.caption = "Positioned in the \(location) of the image. " + region.caption
          }
        }
        let bounds = try ImageLayoutGeometry.pixelBounds(region.bounds, width: draft.width, height: draft.height)
        return itemJSON(region, box: "[" + bounds.centeredYXYX.map(String.init).joined(separator: ",") + "]")
      }
      prompt = "{\"summary\":" + summaryJSON(summary) + ",\"items\":[" + items.joined(separator: ",")
        + "],\"bbox_coord_space\":\"abs_centralized\",\"bbox_format\":\"yxyx\"}"
    }
    return .init(effectivePrompt: prompt, width: draft.width, height: draft.height)
  }
  static func validate(_ draft: ImageLayoutDraft) throws {
    try ImageLayoutGeometry.validateCanvas(width: draft.width, height: draft.height)
    try nonempty(draft.summary.caption, "Scene description")
    if let canvas = draft.summary.canvas, !["single", "multi_panel", "collage"].contains(canvas) {
      throw ImageLayoutError.invalid("Unsupported summary canvas type.")
    }
    try strings(draft.summary.meta); try strings(draft.summary.palette)
    guard draft.regions.count <= 128, Set(draft.regions.map(\.id)).count == draft.regions.count else {
      throw ImageLayoutError.invalid("Use up to 128 uniquely identified regions.")
    }
    try ImageLayoutEditing.validateParents(draft)
    for region in draft.regions {
      try nonempty(region.caption, "Region description")
      if region.type == .text, region.text == nil { throw ImageLayoutError.invalid("Text regions require literal text.") }
      if let text = region.text { try nonempty(text, "Literal text") }
      try strings(region.tags); try strings(region.palette)
      _ = try ImageLayoutGeometry.pixelBounds(region.bounds, width: draft.width, height: draft.height)
    }
  }
  static func nonempty(_ text: String, _ name: String) throws {
    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw ImageLayoutError.invalid("\(name) must not be empty.") }
  }
  static func strings(_ values: [String]?) throws { for value in values ?? [] { try nonempty(value, "Tag or palette entry") } }
  static func quote(_ text: String) -> String {
    // Foundation handles escaping and Unicode while retaining a stable explicit object field order here.
    String(data: try! JSONSerialization.data(withJSONObject: text, options: [.fragmentsAllowed, .withoutEscapingSlashes]), encoding: .utf8)!
  }
  static func array(_ strings: [String]) -> String { "[" + strings.map(quote).joined(separator: ",") + "]" }
  static func summaryJSON(_ summary: ImageLayoutSummary) -> String {
    var fields = ["\"caption\":" + quote(summary.caption)]
    if let meta = summary.meta { fields.append("\"meta\":" + array(meta)) }
    if let canvas = summary.canvas { fields.append("\"canvas\":" + quote(canvas)) }
    if let palette = summary.palette { fields.append("\"palette\":" + array(palette)) }
    return "{" + fields.joined(separator: ",") + "}"
  }
  static func itemJSON(_ region: ImageLayoutRegion, box: String) -> String {
    var fields = ["\"label\":" + quote(region.type.rawValue), "\"caption\":" + quote(region.caption)]
    if let text = region.text { fields.append("\"text\":" + quote(text)) }
    if let tags = region.tags { fields.append("\"tags\":" + array(tags)) }
    if let palette = region.palette { fields.append("\"palette\":" + array(palette)) }
    fields.append("\"bbox\":" + box)
    return "{" + fields.joined(separator: ",") + "}"
  }
}
