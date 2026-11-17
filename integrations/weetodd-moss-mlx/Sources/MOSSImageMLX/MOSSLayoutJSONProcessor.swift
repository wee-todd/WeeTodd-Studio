import Foundation
import MLX
import MLXLMCommon
import MLXGuidedGeneration

final class MOSSLayoutJSONProcessor: LogitProcessor {
  let grammar: GrammarConstraint
  let vocabularySize, eos: Int
  let whitespaceTokenIDs: Set<Int>
  private var whitespace: WhitespaceRunTracker
  private(set) var failure: String?
  private(set) var terminated = false
  init(grammar: GrammarConstraint, vocabularySize: Int, eos: Int, whitespaceTokenIDs: Set<Int> = []) {
    self.grammar = grammar; self.vocabularySize = vocabularySize; self.eos = eos
    self.whitespaceTokenIDs = whitespaceTokenIDs
    self.whitespace = WhitespaceRunTracker(threshold: 3, whitespaceTokenIDs: whitespaceTokenIDs)
  }
  func prompt(_ prompt: MLXArray) {}
  func copy() -> Self {
    do {
      let result = type(of: self).init(grammar: try grammar.clone(), vocabularySize: vocabularySize, eos: eos, whitespaceTokenIDs: whitespaceTokenIDs)
      result.whitespace = whitespace; result.failure = failure; result.terminated = terminated
      return result
    }
    catch {
      let result = type(of: self).init(grammar: grammar, vocabularySize: vocabularySize, eos: eos)
      result.failure = "Unable to copy the layout grammar."; return result
    }
  }
  func process(logits: MLXArray) -> MLXArray {
    do {
      if failure != nil || terminated {
        var bias = [Float](repeating: -.infinity, count: logits.dim(-1)); bias[eos] = 0
        return MLXArray(bias).reshaped(logits.shape)
      }
      let mask = try grammar.computeMask()
      guard mask.needsApply else { return logits }
      var bias = [Float](repeating: -.infinity, count: logits.dim(-1))
      for index in 0..<min(vocabularySize, bias.count) {
        let word = UInt32(bitPattern: mask.mask[index / 32])
        if word & (1 << (index % 32)) != 0 {
          bias[index] = whitespace.isActive && whitespaceTokenIDs.contains(index) ? -200 : 0
        }
      }
      return logits + MLXArray(bias)
    } catch {
      failure = "Layout grammar failed: \(error.localizedDescription)"
      var bias = [Float](repeating: -.infinity, count: logits.dim(-1)); bias[eos] = 0
      return MLXArray(bias).reshaped(logits.shape)
    }
  }
  func didSample(token: MLXArray) {
    guard failure == nil, !terminated else { return }
    do {
      let value = token.item(Int.self)
      terminated = try grammar.commitToken(Int32(value)).isTerminated
      _ = whitespace.record(tokenID: value)
    }
    catch { failure = "The generated layout did not match its grammar." }
  }
}
