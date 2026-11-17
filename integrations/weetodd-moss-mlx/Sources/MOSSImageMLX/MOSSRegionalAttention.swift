import Foundation
import ImageLayout
import MLX

struct MOSSRegionalAttentionPlan {
  struct Partition { let queries, keys: [Int] }
  let partitions: [Partition]
  let inversePermutation: [Int]
  var tokenCount: Int { inversePermutation.count }
  init(height: Int, width: Int, groups: [MOSSRegionalPrompt], textLengths: [Int]) throws {
    guard height > 0, width > 0, height*width <= 16_384, (1...62).contains(groups.count),
      textLengths.count == groups.count, textLengths.allSatisfy({ (1...4096).contains($0) }), textLengths.reduce(0,+) <= 8192,
      groups.filter({ $0.role == .background }).count == 1, groups.filter({ $0.role == .lettering }).count <= 1 else { throw MOSSError("Invalid regional attention dimensions or context budget.") }
    let rows = height*width, image = Array(0..<rows)
    var eligible = [[Int]](repeating: [], count: groups.count)
    for (index, group) in groups.enumerated() where [.subject,.detail,.object].contains(group.role) {
      guard let b = group.bounds else { throw MOSSError("A subject needs a regional box.") }
      let boxes = group.masks ?? [b]
      for box in boxes { _ = try ImageLayoutGeometry.pixelBounds(box, width: 16, height: 16) }
      var cells = Set<Int>()
      for box in boxes {
        let left = max(0,Int(floor(box.x*Double(width)))), right = min(width,Int(ceil((box.x+box.width)*Double(width))))
        let top = max(0,Int(floor(box.y*Double(height)))), bottom = min(height,Int(ceil((box.y+box.height)*Double(height))))
        for y in top..<bottom { for x in left..<right { cells.insert(y*width+x) } }
      }
      eligible[index] = cells.sorted()
      if eligible[index].isEmpty { eligible[index] = [min(height-1,Int((b.y+b.height/2)*Double(height)))*width + min(width-1,Int((b.x+b.width/2)*Double(width)))] }
    }
    let covered = groups.indices.filter { [.subject,.detail].contains(groups[$0].role) }.reduce(into:Set<Int>()) { $0.formUnion(eligible[$1]) }
    for i in groups.indices {
      if groups[i].role == .background { eligible[i] = image.filter { !covered.contains($0) } }
      if groups[i].role == .lettering {
        if let masks = groups[i].masks {
          var cells = Set<Int>()
          for b in masks {
            _ = try ImageLayoutGeometry.pixelBounds(b, width: 16, height: 16)
            let left = max(0,Int(floor(b.x*Double(width)))), right = min(width,Int(ceil((b.x+b.width)*Double(width))))
            let top = max(0,Int(floor(b.y*Double(height)))), bottom = min(height,Int(ceil((b.y+b.height)*Double(height))))
            for y in top..<bottom { for x in left..<right { cells.insert(y*width+x) } }
          }
          eligible[i] = cells.sorted()
        } else { eligible[i] = image }
      }
    }
    var text: [[Int]] = [], next = rows
    for length in textLengths { text.append(Array(next..<(next+length))); next += length }
    var memberships = [[Int]](repeating: [], count: rows)
    for group in groups.indices {
      let available = groups[group].role == .background ? image : eligible[group]
      for i in available { memberships[i].append(group) }
    }
    var buckets: [Int:[Int]] = [:]
    for i in image {
      let key = memberships[i].reduce(0) { $0 | (1 << $1) }
      buckets[key,default:[]].append(i)
    }
    var parts: [Partition] = []
    for key in buckets.keys.sorted() {
      let indices = groups.indices.filter { key & (1 << $0) != 0 }
      parts.append(.init(queries: buckets[key]!, keys: image + indices.flatMap { text[$0] }))
    }
    for i in groups.indices { parts.append(.init(queries: text[i], keys: eligible[i] + text[i])) }
    var inverse = [Int](repeating: -1, count: next), position = 0
    for part in parts { for query in part.queries {
      guard inverse[query] == -1 else { throw MOSSError("A regional attention query was assigned twice.") }
      inverse[query] = position; position += 1
    } }
    guard inverse.allSatisfy({ $0 >= 0 }) else { throw MOSSError("Regional attention omitted a query.") }
    partitions = parts; inversePermutation = inverse
  }
}

struct MOSSPreparedAttentionPlan {
  struct Partition { let queries, keys: MLXArray }
  let partitions: [Partition]
  let inverse: MLXArray
  let tokenCount: Int
  init(_ plan: MOSSRegionalAttentionPlan) {
    partitions = plan.partitions.map { .init(queries: MLXArray($0.queries.map(Int32.init)), keys: MLXArray($0.keys.map(Int32.init))) }
    inverse = MLXArray(plan.inversePermutation.map(Int32.init)); tokenCount = plan.tokenCount
  }
}

enum MOSSRegionalAttention {
  /// Disjoint queries retain Flash Attention; no quadratic dense mask or repeated model pass.
  static func apply(queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, plan: MOSSPreparedAttentionPlan) -> MLXArray {
    precondition(queries.dim(2) == plan.tokenCount && keys.dim(2) == plan.tokenCount)
    let pieces = plan.partitions.map { p in
      MLXFast.scaledDotProductAttention(queries: take(queries,p.queries,axis:2), keys: take(keys,p.keys,axis:2),
        values: take(values,p.keys,axis:2), scale: scale, mask: nil)
    }
    return take(concatenated(pieces,axis:2),plan.inverse,axis:2)
  }
}
