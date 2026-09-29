import Foundation

public struct GemmaLayerConfiguration: Sendable {
  public init() {}
  public var width = 3840
  public var hidden = 15360
  public var heads = 16
  public var kvHeads = 8
  public var headWidth = 256
  public var window: Int? = 1024
  public var keyEqualsValue = false
  public var theta: Double = 10000
  public var rotaryFraction: Double = 1

  public func validate() throws {
    guard (1...8192).contains(width), (1...32768).contains(hidden), (1...128).contains(heads),
      (1...128).contains(kvHeads), heads % kvHeads == 0, (2...512).contains(headWidth), headWidth % 2 == 0,
      theta.isFinite, theta > 1, rotaryFraction.isFinite, (0...1).contains(rotaryFraction),
      window.map({ (1...1024).contains($0) }) ?? true else { throw TextEncodingError.invalid("Unsupported Gemma layer configuration.") }
  }
}

enum GemmaLayer {
  static func expectedShapes(_ c: GemmaLayerConfiguration) -> [String: [Int]] {
    var shapes: [String: [Int]] = [
      "input_layernorm.weight": [c.width], "post_attention_layernorm.weight": [c.width],
      "pre_feedforward_layernorm.weight": [c.width], "post_feedforward_layernorm.weight": [c.width],
      "layer_scalar": [1], "self_attn.q_norm.weight": [c.headWidth], "self_attn.k_norm.weight": [c.headWidth],
      "self_attn.q_proj.weight": [c.heads*c.headWidth, c.width],
      "self_attn.k_proj.weight": [c.kvHeads*c.headWidth, c.width],
      "self_attn.o_proj.weight": [c.width, c.heads*c.headWidth],
      "mlp.gate_proj.weight": [c.hidden, c.width], "mlp.up_proj.weight": [c.hidden, c.width],
      "mlp.down_proj.weight": [c.width, c.hidden]]
    if !c.keyEqualsValue { shapes["self_attn.v_proj.weight"] = [c.kvHeads*c.headWidth, c.width] }
    return shapes
  }
  static func evaluate(_ x: [Float], tokens: Int, configuration c: GemmaLayerConfiguration,
    weights w: TextWeights, gpu: TextMatrixGPU) throws -> [Float] {
    let normalized = TextMath.rms(x, width: c.width, weight: try w.vector("input_layernorm.weight", count: c.width))
    var q = try w.linear("self_attn.q_proj", normalized, tokens: tokens, width: c.width, output: c.heads*c.headWidth, gpu: gpu)
    var k = try w.linear("self_attn.k_proj", normalized, tokens: tokens, width: c.width, output: c.kvHeads*c.headWidth, gpu: gpu)
    var v = c.keyEqualsValue ? k : try w.linear("self_attn.v_proj", normalized, tokens: tokens, width: c.width, output: c.kvHeads*c.headWidth, gpu: gpu)
    q = TextMath.rms(q, width: c.headWidth, weight: try w.vector("self_attn.q_norm.weight", count: c.headWidth))
    k = TextMath.rms(k, width: c.headWidth, weight: try w.vector("self_attn.k_norm.weight", count: c.headWidth))
    v = TextMath.rms(v, width: c.headWidth)
    q = TextMath.gemmaRotary(q, tokens: tokens, heads: c.heads, width: c.headWidth, theta: c.theta, fraction: c.rotaryFraction)
    k = TextMath.gemmaRotary(k, tokens: tokens, heads: c.kvHeads, width: c.headWidth, theta: c.theta, fraction: c.rotaryFraction)
    let attention = try TextMath.attention(q: q, k: k, v: v, tokens: tokens, heads: c.heads, kvHeads: c.kvHeads,
      width: c.headWidth, scale: 1, window: c.window, gpu: gpu)
    let projected = try w.linear("self_attn.o_proj", attention, tokens: tokens, width: c.heads*c.headWidth, output: c.width, gpu: gpu)
    let post = TextMath.rms(projected, width: c.width, weight: try w.vector("post_attention_layernorm.weight", count: c.width))
    let residual = zip(x, post).map(+)
    let ffInput = TextMath.rms(residual, width: c.width, weight: try w.vector("pre_feedforward_layernorm.weight", count: c.width))
    let gate = try w.linear("mlp.gate_proj", ffInput, tokens: tokens, width: c.width, output: c.hidden, gpu: gpu)
    let up = try w.linear("mlp.up_proj", ffInput, tokens: tokens, width: c.width, output: c.hidden, gpu: gpu)
    let activated = zip(gate, up).map { TextMath.gelu($0.0) * $0.1 }
    let down = try w.linear("mlp.down_proj", activated, tokens: tokens, width: c.hidden, output: c.width, gpu: gpu)
    let ff = TextMath.rms(down, width: c.width, weight: try w.vector("post_feedforward_layernorm.weight", count: c.width))
    let scalar = try w.vector("layer_scalar", count: 1)[0]
    let result = zip(residual, ff).map { ($0.0 + $0.1) * scalar }
    guard result.allSatisfy(\.isFinite) else { throw TextEncodingError.invalid("Gemma layer produced nonfinite hidden states.") }
    return result
  }
}
