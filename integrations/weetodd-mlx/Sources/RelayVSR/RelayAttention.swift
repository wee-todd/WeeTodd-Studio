import Foundation
import MLX

struct RelayKV {
  let key: MLXArray
  let value: MLXArray
}

struct RelayRotary {
  let cosine: MLXArray
  let sine: MLXArray

  init(height: Int, width: Int, time: Int, axes: [Int]) {
    let half = axes.reduce(0, +) / 2
    let angles: [Float] = (0..<(height * width)).flatMap { pixel in
      zip([time, pixel / width, pixel % width], axes).flatMap { position, channels in
        (0..<(channels / 2)).map { frequency in
          Float(position) * pow(10_000, -Float(frequency * 2) / Float(channels))
        }
      }
    }
    let angle = MLXArray(angles, [height * width, 1, half])
    cosine = cos(angle)
    sine = sin(angle)
  }

  func apply(_ x: MLXArray) -> MLXArray {
    let pairs = x.reshaped([x.dim(0), x.dim(1), x.dim(2) / 2, 2]).asType(.float32)
    let even = pairs[.ellipsis, 0]
    let odd = pairs[.ellipsis, 1]
    return stacked([even * cosine - odd * sine, even * sine + odd * cosine], axis: -1)
      .reshaped(x.shape).asType(x.dtype)
  }
}

struct RelayAttentionPlan {
  private struct Tile {
    let queries: MLXArray
    let keys: MLXArray
    let mask: MLXArray
  }

  private let tiles: [Tile]
  private let restoreOrder: MLXArray?
  private let wholeGrid: Bool

  init(height: Int, width: Int, windows: [RelayWindow]) {
    let count = height * width
    wholeGrid = windows.allSatisfy { window in
      window.range(at: 0, extent: height, size: window.height).count == height &&
        window.range(at: height - 1, extent: height, size: window.height).count == height &&
        window.range(at: 0, extent: width, size: window.width).count == width &&
        window.range(at: width - 1, extent: width, size: window.width).count == width
    }
    if wholeGrid {
      tiles = []
      restoreOrder = nil
      return
    }
    var tiles: [Tile] = []
    var positions: [Int] = []
    for top in stride(from: 0, to: height, by: 8) {
      for left in stride(from: 0, to: width, by: 16) {
        let rows = top..<min(height, top + 8)
        let columns = left..<min(width, left + 16)
        let queries = rows.flatMap { row in columns.map { row * width + $0 } }
        var keys: [Int32] = []
        var masks: [MLXArray] = []
        for (plane, window) in windows.enumerated() {
          let y = queries.map { window.range(at: $0 / width, extent: height, size: window.height) }
          let x = queries.map { window.range(at: $0 % width, extent: width, size: window.width) }
          let vertical = y[0].lowerBound..<y[y.count - 1].upperBound
          let horizontal = x[0].lowerBound..<x[x.count - 1].upperBound
          var keyRows: [Int32] = []
          var keyColumns: [Int32] = []
          for row in vertical {
            for column in horizontal {
              keys.append(Int32(plane * count + row * width + column))
              keyRows.append(Int32(row))
              keyColumns.append(Int32(column))
            }
          }
          masks.append(Self.axisMask(keyRows, within: y) .&& Self.axisMask(keyColumns, within: x))
        }
        let mask = concatenated(masks, axis: 1)
        tiles.append(Tile(queries: MLXArray(queries.map(Int32.init)), keys: MLXArray(keys), mask: mask))
        positions += queries
      }
    }
    self.tiles = tiles
    var inverse = Array(repeating: Int32(0), count: count)
    for (index, pixel) in positions.enumerated() { inverse[pixel] = Int32(index) }
    restoreOrder = MLXArray(inverse)
  }

  private static func axisMask(_ coordinates: [Int32], within bounds: [Range<Int>]) -> MLXArray {
    let positions = MLXArray(coordinates).reshaped([1, -1])
    let lower = MLXArray(bounds.map { Int32($0.lowerBound) }).reshaped([-1, 1])
    let upper = MLXArray(bounds.map { Int32($0.upperBound) }).reshaped([-1, 1])
    return (positions .>= lower) .&& (positions .< upper)
  }

  func callAsFunction(_ query: MLXArray, _ planes: [RelayKV]) -> MLXArray {
    let keys = concatenated(planes.map(\.key), axis: 0)
    let values = concatenated(planes.map(\.value), axis: 0)
    if wholeGrid { return Self.attend(query, keys, values) }
    var results: [MLXArray] = []
    for (index, tile) in tiles.enumerated() {
      results.append(Self.attend(query.take(tile.queries, axis: 0), keys.take(tile.keys, axis: 0),
        values.take(tile.keys, axis: 0), mask: tile.mask))
      if index % 8 == 7 { eval(results.suffix(8)) }
    }
    return concatenated(results, axis: 0).take(restoreOrder!, axis: 0)
  }

  static func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray? = nil)
    -> MLXArray
  {
    let output = MLXFast.scaledDotProductAttention(
      queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
      keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
      values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
      scale: 1 / sqrt(Float(q.dim(-1))), mask: mask)
    return output.squeezed(axis: 0).transposed(1, 0, 2)
  }
}
