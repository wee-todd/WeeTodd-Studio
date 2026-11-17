import Foundation
import MLX

/// A call-local graph owner. No weights, activations or plan arrays are captured.
/// The custom consumer keeps its original reduction, mask and runtime scale.
final class RelayCompiledAttentionTile {
  private let function: ([MLXArray]) -> [MLXArray] = compile { (a: [MLXArray]) in
    [RelaySkippedAttention.graph(query: a[0].take(a[3], axis: 0),
      key: a[1].take(a[4], axis: 0), value: a[2].take(a[4], axis: 0),
      mask: a[5], classes: a[6], allowSkipping: a[7], scale: a[8])]
  }
  func evaluate(query: MLXArray, key: MLXArray, value: MLXArray, queries: MLXArray,
    keys: MLXArray, mask: MLXArray, classes: MLXArray, safe: MLXArray) -> MLXArray {
    function([query,key,value,queries,keys,mask,classes,safe,
      MLXArray(1 / sqrt(Float(query.dim(2))))])[0]
  }
}
