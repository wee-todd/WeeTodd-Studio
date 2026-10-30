import Foundation

/// Sparse selections are explicit block indices. No production token-square mask is formed.
public enum FlashVSRBlockRouting {
  public static func candidates(queryBlocks: Int, keyBlocks: Int, rows: Int, columns: Int, localRange: Int?) throws -> [[Int]] {
    guard rows > 0, columns > 0, queryBlocks > 0, keyBlocks >= queryBlocks,
      queryBlocks % (rows * columns) == 0, keyBlocks % (rows * columns) == 0,
      localRange == nil || localRange! > 0 else { throw FlashVSRError.invalid("Invalid FlashVSR block grid.") }
    let spatial = rows * columns
    return (0..<queryBlocks).map { q in
      guard let range=localRange else { return Array(0..<keyBlocks) }
      let qr=(q % spatial)/columns, qc=q % columns
      let firstRow=max(0,qr-range/2), stopRow=min(rows,qr-range/2+range)
      let firstColumn=max(0,qc-range/2), stopColumn=min(columns,qc-range/2+range)
      var indices=[Int]()
      indices.reserveCapacity((keyBlocks/spatial)*(stopRow-firstRow)*(stopColumn-firstColumn))
      for time in 0..<(keyBlocks/spatial) {
        for row in firstRow..<stopRow { for column in firstColumn..<stopColumn {
          indices.append(time*spatial+row*columns+column)
        } }
      }
      return indices
    }
  }
  /// Strict grouped threshold, including the release's first-candidate fallback on ties.
  /// Probabilities are row softmaxes; thresholding logits would change cross-row ranking.
  public static func select(probabilities: [[Float]], candidates: [[Int]], temporalGroups: Int, topK: Int) throws -> [[Int]] {
    guard !probabilities.isEmpty, probabilities.count == candidates.count, temporalGroups > 0,
      probabilities.count % temporalGroups == 0, topK >= 0, let width=probabilities.first?.count, width > 0,
      probabilities.allSatisfy({ $0.count == width }),
      candidates.allSatisfy({ !$0.isEmpty && $0.allSatisfy { $0 >= 0 && $0 < width } }) else { throw FlashVSRError.invalid("Invalid draft probabilities or candidate selection.") }
    let perGroup = probabilities.count / temporalGroups
    var result = [[Int]](repeating:[],count:probabilities.count)
    for group in 0..<temporalGroups {
      let start=group*perGroup, stop=start+perGroup
      var values=[Float](); values.reserveCapacity(perGroup * width)
      for row in start..<stop { let allowed=Set(candidates[row]); values += probabilities[row].enumerated().map { allowed.contains($0.offset) ? $0.element : -.infinity } }
      guard values.allSatisfy({ !$0.isNaN }) else { throw FlashVSRError.invalid("Nonfinite draft probabilities.") }
      let threshold=values.sorted(by:>)[min(topK,values.count-1)]
      for row in start..<stop {
        result[row]=candidates[row].filter { probabilities[row][$0] > threshold }
        if result[row].isEmpty { result[row]=[candidates[row][0]] }
      }
    }
    return result
  }
  public static func cacheIndices(keyBlocks: Int, spatialBlocks: Int, windows: Int) throws -> [Int] {
    guard spatialBlocks > 0, keyBlocks > 0, keyBlocks % spatialBlocks == 0, windows > 0 else { throw FlashVSRError.invalid("KV retention requires complete spatial grids.") }
    return Array((keyBlocks-min(windows*spatialBlocks,keyBlocks))..<keyBlocks)
  }
}
