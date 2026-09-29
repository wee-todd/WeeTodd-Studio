import Foundation

/// Retain only the final interleaved layout and one normalized hidden state.
/// The normalization and index order are identical to TextMath.interleavedStates.
final class TextStateAccumulator {
  private let tokens: Int
  private let width: Int
  private let layers: Int
  private var next = 0
  private var consumed = false
  private var values: [Float]
  let admittedBytes: Int

  init(tokens: Int, width: Int, layers: Int, maximumBytes: Int = 1024*1024*1024) throws {
    guard (1...1024).contains(tokens), (1...8192).contains(width), (1...65).contains(layers),
      maximumBytes > 0 else { throw TextEncodingError.invalid("Invalid hidden-state accumulation shape or budget.") }
    admittedBytes = tokens*width*(layers+1)*4
    guard admittedBytes <= maximumBytes else { throw TextEncodingError.invalid("Hidden states exceed the interleaving memory budget.") }
    self.tokens = tokens; self.width = width; self.layers = layers
    values = [Float](repeating: 0, count: tokens*width*layers)
  }
  func append(_ state: [Float]) throws {
    guard !consumed, next < layers, state.count == tokens*width, state.allSatisfy(\.isFinite) else {
      throw TextEncodingError.invalid("Invalid or excess hidden state.")
    }
    try Task.checkCancellation()
    let normalized = TextMath.rms(state, width: width)
    for i in normalized.indices {
      if i % 4096 == 0 { try Task.checkCancellation() }
      values[i*layers+next] = normalized[i]
    }
    next += 1
  }
  func take() throws -> [Float] {
    guard !consumed, next == layers else { throw TextEncodingError.invalid("Hidden-state accumulation is incomplete or consumed.") }
    consumed = true
    let result = values; values = []
    return result
  }
}
