import Foundation
import MLX
import LTX25Text

public enum MLXTextMath {
  public static func rms(_ x:MLXArray,weight:MLXArray?=nil,epsilon:Float=1e-6) -> MLXArray {
    MLXFast.rmsNorm(x,weight:weight ?? .ones([x.shape.last!]),eps:epsilon)
  }
  static func gelu(_ x:MLXArray) -> MLXArray {
    0.5*x*(1+tanh(Float(sqrt(2/Double.pi))*(x+0.044715*x*x*x)))
  }
  public static func interleaved(_ states:[MLXArray]) throws -> MLXArray {
    guard let first=states.first, first.ndim == 2, (1...1024).contains(first.shape[0]),
      (1...8192).contains(first.shape[1]), (1...65).contains(states.count),
      first.size*states.count*4 <= 768*1024*1024,
      states.allSatisfy({ $0.shape == first.shape && $0.dtype == .float32 }) else {
      throw TextEncodingError.invalid("Invalid or oversized hidden-state stack.")
    }
    let result=stacked(states.map { rms($0) },axis:-1).reshaped([first.shape[0],first.shape[1]*states.count])
    try finish(result)
    return result
  }
  static func finish(_ x:MLXArray) throws {
    eval(x); try Task.checkCancellation()
    guard MLX.isFinite(x).all().item(Bool.self) else { throw TextEncodingError.invalid("Nonfinite text stage output.") }
  }
  static func splitRotary(_ x:MLXArray,angles:MLXArray) -> MLXArray {
    let half=x.shape.last!/2, a=x[.ellipsis,0..<half], b=x[.ellipsis,half..<(half*2)]
    return concatenated([a*cos(angles)-b*sin(angles),b*cos(angles)+a*sin(angles)],axis:-1)
  }
}

/// One layer's packed parameters. Scope ends after synchronized evaluation;
/// callers must not retain providers that cache decoded weight payloads.
private struct TextLayerWeights {
  let weights:[String:MLXWeight]
  init(shapes:[String:[Int]],provider:(String,[Int]) throws -> MLXWeight) throws {
    var result:[String:MLXWeight]=[:], bytes=0
    for name in shapes.keys.sorted() {
      try Task.checkCancellation()
      let weight=try provider(name,shapes[name]!)
      try Task.checkCancellation()
      guard weight.shape == shapes[name], weight.storageBytes <= 1024*1024*1024-bytes else {
        throw TextEncodingError.invalid("Text layer weight shape or storage budget differs.")
      }
      result[name]=weight; bytes += weight.storageBytes
    }
    try MLXWeight.materialize(Array(result.values))
    weights=result
  }
  func value(_ name:String) throws -> MLXArray { try weights[name]!.tensor().asType(.float32) }
  func linear(_ name:String,_ input:MLXArray) throws -> MLXArray {
    var output=try weights[name+".weight"]!.projected(input)
    if let bias=weights[name+".bias"] { output=output+(try bias.tensor().asType(.float32)) }
    return output
  }
}

public enum MLXGemmaLayer {
  public static func evaluate(_ input:MLXArray,configuration c:GemmaLayerConfiguration,
    weights:(String,[Int]) throws -> MLXWeight) throws -> MLXArray {
    try c.validate(); try Task.checkCancellation()
    guard Device.defaultDevice().deviceType == .gpu, input.ndim == 2,
      (1...1024).contains(input.shape[0]), input.shape[1] == c.width, input.dtype == .float32 else {
      throw TextEncodingError.invalid("Invalid Gemma input shape/device/dtype.")
    }
    let n=input.shape[0]
    let activationBytes=(n*c.width*12+n*c.hidden*5+n*c.heads*c.headWidth*8+n*c.kvHeads*c.headWidth*8+c.heads*n*n*3)*4
    guard activationBytes <= 2*1024*1024*1024 else { throw TextEncodingError.invalid("Gemma activation budget exceeded.") }
    defer { Stream.gpu.synchronize() }
    let x=input.reshaped(input.shape)
    let w=try TextLayerWeights(shapes:TextModelLayout.gemmaShapes(c),provider:weights)
    let normalized=MLXTextMath.rms(x,weight:try w.value("input_layernorm.weight"))
    var q=try w.linear("self_attn.q_proj",normalized).reshaped([n,c.heads,c.headWidth])
    var k=try w.linear("self_attn.k_proj",normalized).reshaped([n,c.kvHeads,c.headWidth])
    let rawV=c.keyEqualsValue ? k : try w.linear("self_attn.v_proj",normalized).reshaped([n,c.kvHeads,c.headWidth])
    let v=MLXTextMath.rms(rawV).transposed(1,0,2).expandedDimensions(axis:0)
    q=MLXTextMath.rms(q,weight:try w.value("self_attn.q_norm.weight"))
    k=MLXTextMath.rms(k,weight:try w.value("self_attn.k_norm.weight"))
    let pairs=Int(Double(c.headWidth)*c.rotaryFraction)/2
    let frequencies=(0..<(c.headWidth/2)).map { $0 < pairs ? Float(1/pow(c.theta,Double(2*$0)/Double(c.headWidth))) : 0 }
    let positions=MLXArray((0..<n).map(Float.init)).reshaped([n,1,1])
    let angles=positions*MLXArray(frequencies,[1,1,c.headWidth/2])
    q=MLXTextMath.splitRotary(q,angles:angles).transposed(1,0,2).expandedDimensions(axis:0)
    k=MLXTextMath.splitRotary(k,angles:angles).transposed(1,0,2).expandedDimensions(axis:0)
    let ids=MLXArray((0..<n).map(Int32.init)), rows=ids.reshaped([n,1]), cols=ids.reshaped([1,n])
    var mask=lessEqual(cols,rows)
    if let window=c.window { mask=logicalAnd(mask,greater(cols,rows-window)) }
    let attended=MLXFast.scaledDotProductAttention(queries:q,keys:k,values:v,scale:1,mask:mask)
      .transposed(0,2,1,3).reshaped([n,c.heads*c.headWidth])
    let projected=try w.linear("self_attn.o_proj",attended)
    let residual=x+MLXTextMath.rms(projected,weight:try w.value("post_attention_layernorm.weight"))
    let ffInput=MLXTextMath.rms(residual,weight:try w.value("pre_feedforward_layernorm.weight"))
    let gate=try w.linear("mlp.gate_proj",ffInput), up=try w.linear("mlp.up_proj",ffInput)
    let down=try w.linear("mlp.down_proj",MLXTextMath.gelu(gate)*up)
    let ff=MLXTextMath.rms(down,weight:try w.value("post_feedforward_layernorm.weight"))
    let output=(residual+ff)*(try w.value("layer_scalar"))
    try MLXTextMath.finish(output)
    return output
  }
}

public enum MLXTextConnector {
  public static func evaluate(_ input:MLXArray,width:Int,heads:Int=32,
    weights:(String,[Int]) throws -> MLXWeight) throws -> MLXArray {
    let shapes=try TextModelLayout.connectorShapes(width:width,heads:heads)
    guard Device.defaultDevice().deviceType == .gpu, input.ndim == 2,
      (1...1024).contains(input.shape[0]), input.shape[1] == width, input.dtype == .float32 else {
      throw TextEncodingError.invalid("Invalid connector input shape/device/dtype.")
    }
    try Task.checkCancellation()
    let n=input.shape[0], h=width/heads
    guard (n*width*48+heads*n*n*3)*4 <= 2*1024*1024*1024 else { throw TextEncodingError.invalid("Connector activation budget exceeded.") }
    defer { Stream.gpu.synchronize() }
    let x=input.reshaped(input.shape)
    let w=try TextLayerWeights(shapes:shapes,provider:weights)
    let normalized=MLXTextMath.rms(x)
    var q=try w.linear("attn1.to_q",normalized), k=try w.linear("attn1.to_k",normalized)
    let v=try w.linear("attn1.to_v",normalized).reshaped([1,n,heads,h]).transposed(0,2,1,3)
    q=MLXTextMath.rms(q,weight:try w.value("attn1.q_norm.weight"),epsilon:1e-5)
    k=MLXTextMath.rms(k,weight:try w.value("attn1.k_norm.weight"),epsilon:1e-5)
    let frequencies=(0..<(width/2)).map { Float(pow(10000,Double($0)/Double(width/2-1))*(Double.pi/2)) }
    let positions=MLXArray((0..<n).map { Float($0)/4096*2-1 }).reshaped([n,1])
    let angles=(positions*MLXArray(frequencies,[1,width/2])).reshaped([n,heads,h/2])
    q=MLXTextMath.splitRotary(q.reshaped([n,heads,h]),angles:angles).transposed(1,0,2).expandedDimensions(axis:0)
    k=MLXTextMath.splitRotary(k.reshaped([n,heads,h]),angles:angles).transposed(1,0,2).expandedDimensions(axis:0)
    let attended=MLXFast.scaledDotProductAttention(queries:q,keys:k,values:v,scale:1/Float(h).squareRoot(),mask:nil)
      .transposed(0,2,1,3)
    let gates=2*sigmoid(try w.linear("attn1.to_gate_logits",normalized)).reshaped([1,n,heads,1])
    let residual=x+(try w.linear("attn1.to_out.0",(attended*gates).reshaped([n,width])))
    let expansion=try w.linear("ff.net.0.proj",MLXTextMath.rms(residual))
    let output=residual+(try w.linear("ff.net.2",MLXTextMath.gelu(expansion)))
    try MLXTextMath.finish(output)
    return output
  }
}
