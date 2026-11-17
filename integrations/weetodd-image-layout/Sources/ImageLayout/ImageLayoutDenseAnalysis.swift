import Foundation

extension ImageLayoutAnalysis {
  /// Compact vision output: the model describes evidence; host code owns OCR literals/geometry.
  public static func parseDense(_ data: Data, width: Int, height: Int, ocr: [LayoutOCRObservation],
    visualBoxFormat: ImageLayoutAnalysisBoxFormat = .xyxy) throws -> ImageLayoutAnalysis {
    let strict = try LayoutJSON.parse(data)
    guard let keys = strict.object?.keys, Set(keys) == ["caption", "palette", "warnings", "regions", "text_styles", "extra_text_regions"],
      let visual = strict["regions"]?.array, (1...40).contains(visual.count),
      let styles = strict["text_styles"]?.array,
      let extras = strict["extra_text_regions"]?.array, extras.count <= 8,
      let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let rawVisual = raw["regions"] as? [[String: Any]],
      let rawStyles = raw["text_styles"] as? [[String: Any]], let rawExtras = raw["extra_text_regions"] as? [[String: Any]] else {
      throw ImageLayoutError.invalid("Invalid dense layout response.")
    }
    let groups = try ImageLayoutOCR.groups(ocr, maximum: 32)
    guard styles.count == groups.count else { throw ImageLayoutError.invalid("Typography slots must match OCR groups.") }
    let boxKey = visualBoxFormat == .gemmaYXYX ? "box_2d" : "bbox"
    var ids: [Int] = [], parents: [Int] = [], layers: [Int] = []
    for item in visual {
      guard let fields = item.object, Set(fields.keys) == ["id", "parent", "label", "caption", boxKey, "layer"],
        let id = item["id"]?.integer, (1...40).contains(id),
        let parent = item["parent"]?.integer, (-1...40).contains(parent),
        let layer = item["layer"]?.integer, (0...100).contains(layer),
        ["character", "object"].contains(item["label"]?.string ?? "") else { throw ImageLayoutError.invalid("Invalid dense region identity or depth.") }
      ids.append(id); parents.append(parent); layers.append(layer)
    }
    guard Set(ids).count == ids.count, parents.allSatisfy({ $0 == -1 || ids.contains($0) }) else {
      throw ImageLayoutError.invalid("Dense region identifiers must be unique and parents must exist.")
    }
    var items: [[String: Any]] = []
    // Some VLMs emit lettering twice despite the prompt. Admit it only through
    // text regions, so editing a word cannot leave an object caption demanding it.
    let textOnly = ["lettering", "text block", "block of text", "list of categories", "masthead", "stylized yellow title", "stylized title"]
    var admitted = Set<String>()
    let order = visual.indices.filter { i in
      let caption = (rawVisual[i]["caption"] as? String ?? "").lowercased()
      let duplicateKey = "\(parents[i])|\(caption)|\(rawVisual[i][boxKey]!)"
      let unique = admitted.insert(duplicateKey).inserted || parents.contains(ids[i])
      let physical = caption.range(of: #"\b(signboard|pole|wooden|metal|bottle|shirt|jacket|hat|book|screen|neon sign|billboard)\b"#, options: .regularExpression) != nil
      return unique && !(rawVisual[i]["label"] as? String == "object" && parents[i] == -1 && !parents.contains(ids[i]) && !physical && textOnly.contains(where: caption.contains))
    }.sorted { layers[$0] != layers[$1] ? layers[$0] < layers[$1] : $0 < $1 }
    for i in order {
      let item = rawVisual[i]
      items.append(["label": item["label"]!, "caption": item["caption"]!, boxKey: item[boxKey]!, "text": "", "ocr_ids": []])
    }
    var readings: [Int: String] = [:]
    var typography: [Int: String] = [:]
    for (i, group) in groups.enumerated() {
      var style = rawStyles[i]
      if let value = style.removeValue(forKey: "caption") {
        guard let caption = value as? String, !caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, caption.count <= 400 else { throw ImageLayoutError.invalid("Invalid typography description.") }
        typography[i] = caption
      }
      if let value = style.removeValue(forKey: "reading") {
        guard let reading = value as? String, reading.count <= 4096 else { throw ImageLayoutError.invalid("Invalid lettering reading.") }
        let literal = ImageLayoutOCR.literal(group)
        if reading != literal && admittedHeadingReading(reading, corrects: literal) { readings[i] = reading }
      }
      items.append(["label": "text", "style": style, "bbox": [0,0,1000,1000], "text": "", "ocr_ids": group.map(\.id)])
    }
    var acceptedExtras: [String] = []
    for (i, extra) in extras.enumerated() {
      guard let object = extra.object, Set(object.keys) == ["text", "caption", boxKey],
        let text = extra["text"]?.string, !text.isEmpty, text.count <= 256,
        let box = extra[boxKey]?.array, box.count == 4, box.allSatisfy({ $0.double != nil }) else {
        throw ImageLayoutError.invalid("Invalid additional lettering evidence.")
      }
      let native = box.map { $0.double! }
      let v = visualBoxFormat == .gemmaYXYX ? [native[1],native[0],native[3],native[2]] : native
      guard v.allSatisfy({ $0.isFinite && (0...1000).contains($0) }), v[2]>v[0], v[3]>v[1] else { throw ImageLayoutError.invalid("Invalid extra text box.") }
      let bounds = NormalizedLayoutRect(x: v[0]/1000, y: v[1]/1000, width: (v[2]-v[0])/1000, height: (v[3]-v[1])/1000)
      func intersection(_ b: NormalizedLayoutRect) -> Double {
        max(0,min(bounds.x+bounds.width,b.x+b.width)-max(bounds.x,b.x)) * max(0,min(bounds.y+bounds.height,b.y+b.height)-max(bounds.y,b.y))
      }
      func words(_ value: String) -> [String] {
        value.folding(options:[.caseInsensitive,.diacriticInsensitive],locale:.init(identifier:"en_US_POSIX"))
          .split{ !$0.isLetter && !$0.isNumber }.map(String.init)
      }
      let inside = ocr.filter { intersection($0.bounds)/($0.bounds.width*$0.bounds.height)>0.7 }
      let tokens = ImageLayoutOCR.readingLines(inside).flatMap{$0}.flatMap { observation in words(observation.text).map { (word:$0,observation:observation) } }
      let proposedWords = words(text)
      let alreadyRead = !proposedWords.isEmpty && proposedWords.count<=tokens.count &&
        (0...(tokens.count-proposedWords.count)).contains { start in
          let match = Array(tokens[start..<(start+proposedWords.count)])
          guard match.map(\.word) == proposedWords else { return false }
          let span = ImageLayoutOCR.bounds(match.map(\.observation))
          let overlap = intersection(span)
          return overlap/(bounds.width*bounds.height+span.width*span.height-overlap)>0.45
        }
      let covered = ocr.contains { observation in
        let b = observation.bounds
        let overlap = intersection(b)
        let union = bounds.width*bounds.height + b.width*b.height - overlap
        return overlap / union > 0.45
      }
      if covered || alreadyRead { continue } // One authority, including headings split across OCR words.
      acceptedExtras.append(text)
      items.append(["label": "text", "caption": rawExtras[i]["caption"]!, "bbox": v, "text": text, "ocr_ids": []])
    }
    let expanded: [String: Any] = ["caption": raw["caption"]!, "palette": raw["palette"]!, "warnings": raw["warnings"]!, "regions": items]
    var parsed = try parse(JSONSerialization.data(withJSONObject: expanded), width: width, height: height, ocr: ocr, visualBoxFormat: visualBoxFormat)
    var draft = parsed.draft
    var warnings = parsed.warnings
    for (slot, caption) in typography { draft.regions[order.count + slot].caption = caption }
    for (slot, reading) in readings {
      draft.regions[order.count + slot].text = reading
      warnings.append("Review visual OCR correction: \(ImageLayoutOCR.literal(groups[slot])) → \(reading)")
    }
    let mapping = Dictionary(uniqueKeysWithValues: order.enumerated().map { (ids[$0.element], draft.regions[$0.offset].id) })
    for (position, original) in order.enumerated() { if parents[original] != -1 { draft.regions[position].parentID = mapping[parents[original]] } }
    let lettering = ocr.map(\.text) + groups.map(ImageLayoutOCR.literal) + acceptedExtras + Array(readings.values)
    draft.summary.caption = normalizingSpatialLanguage(removingLiteralLettering(draft.summary.caption, literals: lettering))
    for i in draft.regions.indices {
      draft.regions[i].caption = normalizingSpatialLanguage(removingLiteralLettering(draft.regions[i].caption, literals: lettering))
    }
    try ImageLayoutEditing.validateParents(draft)
    _ = try MOSSLayoutCompiler.compile(draft)
    parsed = .init(draft: draft, warnings: warnings.sorted())
    return parsed
  }
  public static func normalizingSpatialLanguage(_ caption: String) -> String {
    var value = caption
    let patterns = [#"\b(central|centered|centrally placed)\s+"#,
      #"\b(in|at|on) the (top |bottom |upper |lower )?(left|right|center|centre)(?! (?:side |corner )?of (?:her|his|their|its|the (?:face|body|head|nose|eye|arm|hand))\b)( corner| side| of the (image|frame|picture))?\b"#]
    for pattern in patterns { value = value.replacingOccurrences(of: pattern, with: "", options: [.regularExpression,.caseInsensitive]) }
    return value.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }.joined(separator: " ")
  }
  public static func admittedHeadingReading(_ proposed: String, corrects original: String) -> Bool {
    if proposed == original { return true }
    let a = Array(original.uppercased()), b = Array(proposed.uppercased())
    guard (4...64).contains(a.count), a.count == b.count, a.allSatisfy(\.isLetter), b.allSatisfy(\.isLetter) else { return false }
    return zip(a,b).filter { $0 != $1 }.count <= min(2, a.count/4)
  }
  public static func removingLiteralLettering(_ caption: String, literals: [String]) -> String {
    var value = caption
    for literal in Set(literals).sorted(by: { $0.count > $1.count }) where !literal.isEmpty {
      let pattern = "(?<![\\p{L}\\p{N}])" + NSRegularExpression.escapedPattern(for: literal) + "(?![\\p{L}\\p{N}])"
      value = value.replacingOccurrences(of: pattern, with: "", options: literal.count >= 3 ? [.regularExpression, .caseInsensitive] : [.regularExpression])
    }
    value = value.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }.joined(separator: " ")
    value = value.replacingOccurrences(of: #"(?<!\w)-style\b"#, with: "period-style", options: .regularExpression)
      .replacingOccurrences(of: #"['‘’“”\"]\s*['‘’“”\"]"#, with: "", options: .regularExpression)
    return value.isEmpty ? "Visible design element" : value
  }
}
