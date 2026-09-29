import MLX

/// Comfy checkpoints group all Q, K and V heads; pruned FL2VA checkpoints
/// already store each head's Q, K and V rows together.
enum H3QKVRowOrder {
  static func forHeadMajorAttention(_ weight: MLXArray,
    heads: Int, headWidth: Int, groupedSource: Bool) -> MLXArray {
    guard groupedSource else { return weight }
    let columns = weight.shape[1]
    return weight.reshaped([3, heads, headWidth, columns])
      .transposed(1, 0, 2, 3)
      .reshaped([3 * heads * headWidth, columns])
  }
}
