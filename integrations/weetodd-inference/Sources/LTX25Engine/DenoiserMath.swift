import Foundation

public enum DenoiserMath {
  /// Match the existing renderer's explicit BF16 input boundary, then compute in Float32.
  public static func bfloat16(_ value: Float) -> Float {
    guard value.isFinite else { return value }
    let bits = value.bitPattern
    return Float(bitPattern: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) & 0xffff0000)
  }

  public static func timestep(_ sigma: Float) throws -> [Float] {
    guard sigma.isFinite, (0...1).contains(sigma) else { throw LTXError.invalid("Sigma must be finite in 0...1.") }
    let scaled = bfloat16(bfloat16(sigma) * 1000)
    let angles = (0..<128).map { scaled * expf(-Float(log(10000.0)) * Float($0) / 128) }
    return angles.map(cosf) + angles.map(sinf)
  }

  public struct Rotary: Sendable { public let cos: [Float]; public let sin: [Float] }

  /// Validate storage without allocating grids. The default preserves standalone
  /// probe admission; renderers may supply their explicitly admitted workspace.
  public static func rotaryElementCount(axes:Int,tokens:Int,heads:Int,headWidth:Int,
    maximumBytes:Int=256*1024*1024) throws -> Int {
    guard (1...3).contains(axes),(1...131072).contains(tokens),(1...128).contains(heads),
      (2...256).contains(headWidth),headWidth % 2 == 0,
      (1...32*1024*1024*1024).contains(maximumBytes) else {
      throw LTXError.invalid("Unsupported rotary layout or workspace allowance.")
    }
    let half=heads*headWidth/2,elements=tokens*half
    guard half/axes > 1,elements <= maximumBytes/8 else {
      throw LTXError.invalid("Rotary grid needs \(elements*8) bytes for sine/cosine; admitted allowance is \(maximumBytes) bytes.")
    }
    return elements
  }

  /// Float64 logarithmic grid, Float32 position arithmetic, leading split padding.
  /// Output is token-major [tokens * heads, headWidth / 2], as the stack expects.
  public static func rotary(positions: [Float], axes: Int, tokens: Int, heads: Int,
    headWidth: Int, maximumPositions: [Float],maximumBytes:Int=256*1024*1024) throws -> Rotary {
    let elements=try rotaryElementCount(axes:axes,tokens:tokens,heads:heads,headWidth:headWidth,maximumBytes:maximumBytes)
    guard (1...3).contains(axes), (1...131072).contains(tokens), (1...128).contains(heads),
      (2...256).contains(headWidth), headWidth % 2 == 0, positions.count == tokens * axes,
      positions.allSatisfy(\.isFinite), maximumPositions.count == axes,
      maximumPositions.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw LTXError.invalid("Unsupported rotary positions or layout.")
    }
    let half = heads * headWidth / 2, count = half / axes
    let padding = half - count * axes
    let grid = (0..<count).map { Float(pow(10000, Double($0) / Double(count - 1)) * (.pi / 2)) }
    var cos = [Float](repeating: 1, count: elements), sin = [Float](repeating: 0, count: elements)
    for token in 0..<tokens {
      try Task.checkCancellation()
      for frequency in 0..<count {
        for axis in 0..<axes {
          let position = positions[token * axes + axis] / maximumPositions[axis]
          let angle = grid[frequency] * (position * 2 - 1)
          guard angle.isFinite else { throw LTXError.invalid("Rotary position arithmetic overflowed.") }
          let index = token * half + padding + frequency * axes + axis
          cos[index] = cosf(angle); sin[index] = sinf(angle)
        }
      }
    }
    return Rotary(cos: cos, sin: sin)
  }
}
