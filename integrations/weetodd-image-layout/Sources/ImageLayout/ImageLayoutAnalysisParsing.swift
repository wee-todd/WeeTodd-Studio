import Foundation

public enum ImageLayoutAnalysisBoxFormat: Sendable {
  case xyxy, gemmaYXYX
}

extension ImageLayoutAnalysis {
  public static func parse(_ data: Data, width: Int, height: Int, ocr: [LayoutOCRObservation],
    visualBoxFormat: ImageLayoutAnalysisBoxFormat = .xyxy) throws -> ImageLayoutAnalysis {
    guard data.count <= 256 * 1024, ocr.count <= 384, Set(ocr.map(\.id)).count == ocr.count else {
      throw ImageLayoutError.invalid("Image analysis exceeds its response/evidence limits.")
    }
    for item in ocr {
      guard item.id >= 0, !item.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
        item.text.count <= 4096, item.confidence.isFinite, (0...1).contains(item.confidence) else {
        throw ImageLayoutError.invalid("Invalid OCR evidence.")
      }
      try ImageLayoutAnalysisGeometry.validate(item.bounds)
    }
    let evidence = Dictionary(uniqueKeysWithValues: ocr.map { ($0.id, $0) })
    let root = try LayoutJSON.parse(data)
    let separated = root["text_regions"] != nil
    try fields(root, separated ? ["caption", "palette", "warnings", "regions", "text_regions"] : ["caption", "palette", "warnings", "regions"])
    let caption = try string(root["caption"], maximum: 4000, empty: false)
    let palette = try strings(root["palette"], maximum: 12, length: 80)
    var warnings = try strings(root["warnings"], maximum: 32, length: 600)
    guard let visual = root["regions"]?.array, visual.count <= (separated ? 40 : 80) else { throw ImageLayoutError.invalid("Analysis has too many regions.") }
    let lettering: [LayoutJSON]
    if separated {
      guard let text = root["text_regions"]?.array, text.count <= 40,
        visual.allSatisfy({ ["object", "character"].contains($0["label"]?.string ?? "") }),
        text.allSatisfy({ $0["label"]?.string == "text" }) else {
        throw ImageLayoutError.invalid("Keep visual subjects and lettering in their separate analysis groups.")
      }
      lettering = text
    } else { lettering = [] }
    let items = visual + lettering
    var used = Set<Int>()
    var regions = try items.map { item -> ImageLayoutRegion in
      let styled = item["style"] != nil
      guard let label = item["label"]?.string.flatMap(ImageLayoutRegionType.init(rawValue:)) else {
        throw ImageLayoutError.invalid("Analysis region has an invalid type.")
      }
      let boxKey = visualBoxFormat == .gemmaYXYX && label != .text ? "box_2d" : "bbox"
      try fields(item, styled ? ["label", "style", boxKey, "text", "ocr_ids"] : ["label", "caption", boxKey, "text", "ocr_ids"])
      guard let box = item[boxKey]?.array, box.count == 4, box.allSatisfy({ $0.double != nil }),
        let rawIDs = item["ocr_ids"]?.array, rawIDs.allSatisfy({ $0.integer != nil }) else {
        throw ImageLayoutError.invalid("Analysis region has an invalid type, box or OCR identifiers.")
      }
      let raw = box.map { $0.double! }
      // Gemma's native box_2d is [top,left,bottom,right]. Editor and MOSS remain canonical XYXY.
      let values = boxKey == "box_2d" ? [raw[1], raw[0], raw[3], raw[2]] : raw
      guard values.allSatisfy({ $0.isFinite && (0...1000).contains($0) }),
        values[2] > values[0], values[3] > values[1] else { throw ImageLayoutError.invalid("Analysis box must be increasing normalized coordinates.") }
      var bounds = NormalizedLayoutRect(x: values[0] / 1000, y: values[1] / 1000,
        width: (values[2] - values[0]) / 1000, height: (values[3] - values[1]) / 1000)
      let description: String
      if styled {
        guard label == .text, let style = item["style"] else { throw ImageLayoutError.invalid("Typography belongs only to text regions.") }
        try fields(style, ["color", "font", "weight", "case"])
        func choice(_ field: String, _ allowed: [String]) throws -> String {
          guard let value = style[field]?.string, allowed.contains(value) else { throw ImageLayoutError.invalid("Unsupported typography style.") }
          return value
        }
        description = try choice("color", ImageLayoutTypography.colors) + ", " + choice("weight", ImageLayoutTypography.weights)
          + " " + choice("font", ImageLayoutTypography.fonts) + " " + choice("case", ImageLayoutTypography.cases) + " lettering"
      } else { description = try string(item["caption"], maximum: 1600, empty: false) }
      let modelText = try string(item["text"], maximum: 4096, empty: true)
      let ids = rawIDs.map { $0.integer! }
      guard Set(ids).count == ids.count, ids.allSatisfy({ evidence[$0] != nil && !used.contains($0) }) else {
        throw ImageLayoutError.invalid("Analysis refers to missing or repeated OCR evidence.")
      }
      var literal: String?
      if label == .text {
        if ids.isEmpty {
          guard !modelText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw ImageLayoutError.invalid("Text regions need literal text or OCR evidence.") }
          literal = modelText
          warnings.append("Review vision-only lettering: \(String(modelText.prefix(100)))")
        } else {
          let lines = ids.map { evidence[$0]! }
          literal = ImageLayoutOCR.literal(lines)
          bounds = ImageLayoutOCR.bounds(lines)
          if lines.contains(where: { $0.confidence < 0.7 }) { warnings.append("Review low-confidence OCR: \(String(literal!.prefix(100)))") }
          if !modelText.isEmpty && modelText != literal { warnings.append("Kept original OCR lettering instead of the model's alternate reading.") }
        }
      } else {
        guard ids.isEmpty, modelText.isEmpty else { throw ImageLayoutError.invalid("Only text regions can use lettering or OCR evidence.") }
      }
      used.formUnion(ids)
      return .init(type: label, caption: description, text: literal, bounds: bounds)
    }
    if separated {
      let originalCount = regions.count
      let retainedEvidence = ocr.filter { used.contains($0.id) }
      regions.removeAll { region in
        guard region.type == .object, ImageLayoutTypography.describesOnlyLettering(region.caption) else { return false }
        let box = region.bounds
        let covered = retainedEvidence.reduce(0.0) { sum, item in
          let b = item.bounds
          let w = max(0, min(box.x + box.width, b.x + b.width) - max(box.x, b.x))
          let h = max(0, min(box.y + box.height, b.y + b.height) - max(box.y, b.y))
          return sum + w * h
        }
        return covered / (box.width * box.height) >= 0.55
      }
      if regions.count < originalCount { warnings.append("Removed \(originalCount - regions.count) duplicate lettering description(s); exact OCR text remains in its text boxes.") }
    }
    if used.count < ocr.count { warnings.append("\(ocr.count - used.count) detected text line(s) were not included; review small print in the source guide.") }
    var draft = ImageLayoutDraft(width: width, height: height, caption: caption, regions: regions)
    draft.summary.palette = palette.isEmpty ? nil : palette
    _ = try MOSSLayoutCompiler.compile(draft)
    return .init(draft: draft, warnings: Array(Set(warnings)).sorted())
  }
  private static func fields(_ value: LayoutJSON, _ expected: Set<String>) throws {
    guard let object = value.object, Set(object.keys) == expected else { throw ImageLayoutError.invalid("Analysis contains missing or unsupported fields.") }
  }
  private static func string(_ value: LayoutJSON?, maximum: Int, empty: Bool) throws -> String {
    guard let text = value?.string, text.count <= maximum,
      empty || !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw ImageLayoutError.invalid("Invalid analysis text.") }
    return text
  }
  private static func strings(_ value: LayoutJSON?, maximum: Int, length: Int) throws -> [String] {
    guard let values = value?.array, values.count <= maximum else { throw ImageLayoutError.invalid("Invalid analysis text list.") }
    return try values.map { try string($0, maximum: length, empty: true) }.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
  }
}
