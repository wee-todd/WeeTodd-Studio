import Foundation

/// Trained LTX2.5 connector: full-width Q/K normalization, per-head gates,
/// noncausal split RoPE, GELU feedforward and repeated learned registers.
enum TextConnector {
  static func expectedShapes(width: Int, heads: Int = 32) -> [String: [Int]] {
    var result: [String: [Int]] = [:]
    for projection in ["to_q", "to_k", "to_v", "to_out.0"] {
      result["attn1.\(projection).weight"] = [width, width]
      result["attn1.\(projection).bias"] = [width]
    }
    result["attn1.q_norm.weight"] = [width]; result["attn1.k_norm.weight"] = [width]
    result["attn1.to_gate_logits.weight"] = [heads, width]; result["attn1.to_gate_logits.bias"] = [heads]
    result["ff.net.0.proj.weight"] = [4*width, width]; result["ff.net.0.proj.bias"] = [4*width]
    result["ff.net.2.weight"] = [width, 4*width]; result["ff.net.2.bias"] = [width]
    return result
  }
  static func evaluate(_ input: [Float], tokens: Int, width: Int, heads: Int = 32,
    weights w: TextWeights, gpu: TextMatrixGPU) throws -> [Float] {
    let h = width/heads, normalized = TextMath.rms(input, width: width)
    func project(_ name: String) throws -> [Float] {
      try w.linear("attn1." + name, normalized, tokens: tokens, width: width, output: width, bias: true, gpu: gpu)
    }
    var q = try project("to_q"), k = try project("to_k")
    let v = try project("to_v")
    q = TextMath.rms(q, width: width, weight: try w.vector("attn1.q_norm.weight", count: width), epsilon: 1e-5)
    k = TextMath.rms(k, width: width, weight: try w.vector("attn1.k_norm.weight", count: width), epsilon: 1e-5)
    q = TextMath.connectorRotary(q, tokens: tokens, heads: heads, width: h)
    k = TextMath.connectorRotary(k, tokens: tokens, heads: heads, width: h)
    var attention = try TextMath.attention(q: q, k: k, v: v, tokens: tokens, heads: heads, kvHeads: heads,
      width: h, scale: 1/Float(h).squareRoot(), causal: false, gpu: gpu)
    let gates = try w.linear("attn1.to_gate_logits", normalized, tokens: tokens, width: width, output: heads, bias: true, gpu: gpu)
    for i in attention.indices { attention[i] *= 2/(1+exp(-gates[i/h])) }
    let projected = try w.linear("attn1.to_out.0", attention, tokens: tokens, width: width, output: width, bias: true, gpu: gpu)
    let residual = zip(input, projected).map(+)
    let ffInput = TextMath.rms(residual, width: width)
    let expansion = try w.linear("ff.net.0.proj", ffInput, tokens: tokens, width: width, output: 4*width, bias: true, gpu: gpu)
    let feedForward = try w.linear("ff.net.2", expansion.map(TextMath.gelu), tokens: tokens, width: 4*width, output: width, bias: true, gpu: gpu)
    let result = zip(residual, feedForward).map(+)
    guard result.allSatisfy(\.isFinite) else { throw TextEncodingError.invalid("Text connector produced nonfinite output.") }
    return result
  }
}
