import Foundation
import MLX
import TensorIO

/// Decode one bounded H3 video VAE tile through the installed affine-Q8 ViT.
/// Temporal chunking and spatial stitching are separate callers of this tile.
public enum H3VideoVAETileDecoder {
  public static func decode(checkpointURL: URL, latent: MLXArray,
    progress: (Int, Int) -> Void = { _, _ in }) throws -> MLXArray {
    try decode(checkpointURL: checkpointURL, latent: latent,
      progress: progress, observe: { _, _ in })
  }

  static func decode(checkpointURL: URL, latent: MLXArray,
    progress: (Int, Int) -> Void = { _, _ in },
    observe: (String, MLXArray) throws -> Void) throws -> MLXArray {
    guard latent.ndim == 5, (1...4).contains(latent.shape[0]),
      (1...16).contains(latent.shape[1]),
      (1...32).contains(latent.shape[2]),
      (1...32).contains(latent.shape[3]),
      latent.shape[4] == 24, latent.dtype == .float32,
      latent.shape[1] * latent.shape[2] * latent.shape[3] <= 4096 else {
      throw H3CheckpointError.invalid("H3 video VAE tile exceeds bounded latent geometry.")
    }
    _ = try H3VideoVAELayout(url: checkpointURL)
    let file = try SafeTensorFile(url: checkpointURL)
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
    }
    func read(_ name: String, shape: [Int]) throws -> MLXArray {
      guard let descriptor = file.tensors[name], descriptor.dtype == "F16",
        descriptor.shape == shape.map(UInt64.init) else {
        throw H3CheckpointError.invalid("Missing H3 video VAE tile tensor: \(name)")
      }
      let value = try file.withTensorBytes(named: name) { bytes in
        MLXArray(bytes, shape, type: Float16.self)
      }
      eval(value)
      return value
    }
    func linear(_ input: MLXArray, _ stem: String,
      rows: Int, columns: Int) throws -> MLXArray {
      let weight = try read(stem + ".weight", shape: [rows, columns])
      let bias = try read(stem + ".bias", shape: [rows])
      let result = addMM(bias, input, weight.T)
      eval(result)
      return result
    }
    let batch = latent.shape[0]
    let depth = latent.shape[1]
    let height = latent.shape[2]
    let width = latent.shape[3]
    let patches = depth * height * width
    let postWeight = try read("post_quant_conv.weight", shape: [24, 1, 1, 1, 24])
    let postBias = try read("post_quant_conv.bias", shape: [24])
    let postQuant = conv3d(latent, postWeight) + postBias
    eval(postQuant)
    try observe("postquant", postQuant)
    let embedded = try linear(postQuant.reshaped([batch, patches, 24]),
      "decoder.x_embedder", rows: 2048, columns: 24)
    try observe("embed", embedded)
    let registers = broadcast(try read("decoder.register_tokens",
      shape: [1, 4, 2048]).asType(embedded.dtype), to: [batch, 4, 2048])
    let classToken = MLXArray.zeros([batch, 1, 2048], dtype: embedded.dtype)
    var tokens = concatenated([embedded, registers, classToken], axis: 1)
    func grid(_ size: Int) -> MLXArray {
      2 * ((MLXArray((0..<size).map(Float.init)) + Float(0.5)) / Float(size)) - 1
    }
    let depthGrid = grid(depth), heightGrid = grid(height), widthGrid = grid(width)
    let tt = broadcast(depthGrid.reshaped([depth, 1, 1]), to: [depth, height, width])
    let yy = broadcast(heightGrid.reshaped([1, height, 1]), to: [depth, height, width])
    let xx = broadcast(widthGrid.reshaped([1, 1, width]), to: [depth, height, width])
    let spatial = broadcast(stacked([tt, yy, xx], axis: -1)
      .reshaped([1, patches, 3]), to: [batch, patches, 3])
    let positions = concatenated([spatial,
      MLXArray.zeros([batch, 5, 3], dtype: .float32)], axis: 1)
    eval(positions)
    try observe("positions", positions)
    for index in 0..<36 {
      tokens = try H3VideoVAEBlock.evaluate(checkpointURL: checkpointURL,
        index: index, input: tokens, positions: positions)
      if index == 0 || index == 1 || index == 35 {
        try observe("vit\(index)", tokens)
      }
      progress(index + 1, 36)
      try Task.checkCancellation()
    }
    let normWeight = try read("decoder.norm_out.weight", shape: [2048])
    let normBias = try read("decoder.norm_out.bias", shape: [2048])
    let normalized = MLXFast.layerNorm(tokens,
      weight: normWeight.asType(tokens.dtype),
      bias: normBias.asType(tokens.dtype), eps: 1e-5)
    eval(normalized)
    try observe("normout", normalized)
    let head = try linear(normalized, "decoder.proj_out",
      rows: 3072, columns: 2048)[0..<batch, 0..<patches, 0..<3072]
    eval(head)
    try observe("head", head)
    let pixels = head.reshaped([batch, depth, height, width, 3, 4, 16, 16])
      .transposed(0, 1, 5, 2, 6, 3, 7, 4)
      .reshaped([batch, depth * 4, height * 16, width * 16, 3])
    eval(pixels)
    try observe("pixels", pixels)
    try file.checkUnchanged(at: checkpointURL)
    try Task.checkCancellation()
    return pixels
  }
}
