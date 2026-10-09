import Foundation

/// Experimental contiguous Sol groups; unrelated to trained VSA spatial tiles.
/// No model, array or GPU state is created by this admission contract.
struct H3SolGeometry: Sendable, Equatable {
  let rows: Int
  let heads: Int
  let dimension: Int
  let blockSize: Int
  let queryBlockSize: Int
  let approximationRange: Range<Int>
  let localRadius: Int
  let scale: Float
  let tau: Float
  var keyBlocks: Int { (rows + blockSize - 1) / blockSize }
  var queryBlocks: Int { (rows + queryBlockSize - 1) / queryBlockSize }
  var routeWords: Int { (keyBlocks + 31) / 32 }

  init(rows: Int, heads: Int, dimension: Int = 128, blockSize: Int = 64,
    queryBlockSize: Int = 64, approximationRange: Range<Int>,
    localRadius: Int = 1, scale: Float = 1 / sqrt(Float(128)), tau: Float = 0.5) throws {
    guard (1...40_000).contains(rows), (1...56).contains(heads), dimension == 128,
      blockSize == 64, queryBlockSize == 64,
      approximationRange.lowerBound >= 0, approximationRange.upperBound <= rows,
      (1...625).contains(localRadius), scale.isFinite, abs(scale) <= 1,
      tau.isFinite, abs(tau) <= 16 else {
      throw H3CheckpointError.invalid("Invalid experimental H3 Sol geometry or numeric policy.")
    }
    self.rows = rows; self.heads = heads; self.dimension = dimension
    self.blockSize = blockSize; self.queryBlockSize = queryBlockSize
    self.approximationRange = approximationRange; self.localRadius = localRadius
    self.scale = scale; self.tau = tau
  }

  func keyCount(_ block: Int) -> Int { min(blockSize, rows - block * blockSize) }
  func queryCount(_ block: Int) -> Int { min(queryBlockSize, rows - block * queryBlockSize) }

  /// Caller must supply admitted block indices. A boundary-crossing block is
  /// protected in its entirety; no query/token mask is implied by the range.
  func requiresExact(queryBlock: Int, keyBlock: Int) -> Bool {
    let qStart = queryBlock * queryBlockSize
    let kStart = keyBlock * blockSize
    return qStart < approximationRange.lowerBound
      || qStart + queryCount(queryBlock) > approximationRange.upperBound
      || kStart < approximationRange.lowerBound
      || kStart + keyCount(keyBlock) > approximationRange.upperBound
      || abs(qStart / blockSize - keyBlock) <= localRadius
  }
}
