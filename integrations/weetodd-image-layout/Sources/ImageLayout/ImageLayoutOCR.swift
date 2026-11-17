import Foundation

public enum ImageLayoutOCR {
  public static func merging(_ observations: [LayoutOCRObservation]) throws -> [LayoutOCRObservation] {
    var result: [LayoutOCRObservation] = []
    for item in observations.sorted(by: { $0.confidence != $1.confidence ? $0.confidence > $1.confidence : topLeft($0,$1) }) {
      try ImageLayoutAnalysisGeometry.validate(item.bounds)
      let a = item.bounds
      var remove = Set<Int>(), duplicate = false
      for (index, other) in result.enumerated() {
        let b = other.bounds
        let overlap = max(0, min(a.x+a.width,b.x+b.width)-max(a.x,b.x)) * max(0,min(a.y+a.height,b.y+b.height)-max(a.y,b.y))
        guard overlap / min(a.width*a.height,b.width*b.height) > 0.75,
          abs(a.y+a.height/2-b.y-b.height/2) < min(a.height,b.height)*0.4 else { continue }
        let left = item.text.folding(options: [.caseInsensitive,.diacriticInsensitive], locale: .init(identifier: "en_US_POSIX"))
        let right = other.text.folding(options: [.caseInsensitive,.diacriticInsensitive], locale: .init(identifier: "en_US_POSIX"))
        if left == right { duplicate = true; break }
        if left.contains(right), a.width*a.height >= b.width*b.height { remove.insert(index); continue }
        if right.contains(left), b.width*b.height >= a.width*a.height { duplicate = true; break }
        let union = a.width*a.height+b.width*b.height-overlap
        if overlap/union > 0.75 && left.count == right.count { duplicate = true; break }
      }
      if !duplicate { result = result.enumerated().filter { !remove.contains($0.offset) }.map(\.element); result.append(item) }
    }
    return readingLines(result).flatMap { $0 }.enumerated().map {
      .init(id: $0.offset, text: $0.element.text, bounds: $0.element.bounds, confidence: $0.element.confidence)
    }
  }
  private static func topLeft(_ a: LayoutOCRObservation, _ b: LayoutOCRObservation) -> Bool {
    if a.bounds.y != b.bounds.y { return a.bounds.y < b.bounds.y }
    if a.bounds.x != b.bounds.x { return a.bounds.x < b.bounds.x }
    if a.text != b.text { return a.text < b.text }
    return a.id < b.id
  }
  public static func readingLines(_ observations: [LayoutOCRObservation]) -> [[LayoutOCRObservation]] {
    var rows: [[LayoutOCRObservation]] = []
    for item in observations.sorted(by: topLeft) {
      let b = item.bounds
      let row = rows.indices.first { index in
        let a = rows[index][0].bounds
        let overlap = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
        return overlap >= min(a.height, b.height) * 0.6 && abs(a.y + a.height / 2 - b.y - b.height / 2) <= min(a.height, b.height) * 0.6
      }
      if let row { rows[row].append(item) } else { rows.append([item]) }
    }
    return rows.map { row in row.sorted { a, b in
      a.bounds.x != b.bounds.x ? a.bounds.x < b.bounds.x : topLeft(a, b)
    } }
  }
  public static func bounds(_ observations: [LayoutOCRObservation]) -> NormalizedLayoutRect {
    precondition(!observations.isEmpty)
    let x = observations.map(\.bounds.x).min()!, y = observations.map(\.bounds.y).min()!
    return .init(x: x, y: y, width: observations.map { $0.bounds.x + $0.bounds.width }.max()! - x,
      height: observations.map { $0.bounds.y + $0.bounds.height }.max()! - y)
  }
  public static func literal(_ observations: [LayoutOCRObservation]) -> String {
    readingLines(observations).map { $0.map(\.text).joined(separator: " ") }.joined(separator: "\n")
  }
  public static func groups(_ observations: [LayoutOCRObservation], maximum: Int = 20) throws -> [[LayoutOCRObservation]] {
    guard (1...32).contains(maximum), observations.count <= 384, Set(observations.map(\.id)).count == observations.count else {
      throw ImageLayoutError.invalid("Invalid OCR grouping limits or identifiers.")
    }
    for item in observations { try ImageLayoutAnalysisGeometry.validate(item.bounds) }
    var groups = observations.sorted(by: topLeft).map { [$0] }
    while groups.count > maximum {
      var best: (a: Int, b: Int, score: Double)?
      for i in groups.indices { for j in groups.indices where j > i {
        let a = bounds(groups[i]), b = bounds(groups[j])
        let xGap = max(0, max(a.x, b.x) - min(a.x + a.width, b.x + b.width))
        let yGap = max(0, max(a.y, b.y) - min(a.y + a.height, b.y + b.height))
        let overlap = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
        let sameColumn = overlap >= min(a.width, b.width) * 0.65
        let sameRow = abs(a.y + a.height / 2 - b.y - b.height / 2) <= min(a.height, b.height) * 0.5
        let compatible = sameColumn || (sameRow && xGap <= max(a.height, b.height))
        let score = (compatible ? 0 : 100) + yGap / max(0.005, min(a.height, b.height))
          + xGap / max(0.005, min(a.width, b.width)) + abs(log(a.height / b.height)) * 0.2
        if best == nil || score < best!.score { best = (i, j, score) }
      }}
      guard let best else { break }
      groups[best.a] += groups[best.b]; groups.remove(at: best.b)
    }
    return groups.map { readingLines($0).flatMap { $0 } }.sorted { topLeft($0.min(by: topLeft)!, $1.min(by: topLeft)!) }
  }
}
