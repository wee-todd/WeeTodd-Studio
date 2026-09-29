// Independently expressed from WeeTodd's existing LTX 2.5 equations.
import NNC

final class AVBlockGraph {
  struct Binding {
    let layer: Model
    let name: String
    let shape: [Int]
    let storageShape: [Int]
    let bias: Bool
    let projection: Bool
  }
  let model: Model
  let inputNames: [String]
  let inputShapes: [String: [Int]]
  let bindings: [Binding]
  let outputNames: [String]

  init(configuration c: AVBlockConfiguration, diagnostics: Bool, sequenceAttention: Bool = false,
    precision: LTXPrecisionPolicy = .float32) {
    let vd = c.videoDimension, ad = c.audioDimension
    let nv = c.videoTokens, na = c.audioTokens, nt = c.textTokens
    let heads = c.heads, vh = c.videoHeadDimension, ah = c.audioHeadDimension
    var inputs: [Input] = [], names: [String] = [], shapes: [String: [Int]] = [:]
    var bindings: [Binding] = []
    func input(_ name: String, _ shape: [Int]) -> Input {
      let value = Input(); inputs.append(value); names.append(name); shapes[name] = shape
      return value
    }
    let video = input("video", [nv, vd]), audio = input("audio", [na, ad])
    let videoMods = input("video_modulation", [1, 9 * vd])
    let audioMods = input("audio_modulation", [1, 9 * ad])
    let videoPrompt = input("video_prompt_modulation", [1, 2 * vd])
    let audioPrompt = input("audio_prompt_modulation", [1, 2 * ad])
    let videoAV = input("video_av_modulation", [1, 4 * vd])
    let audioAV = input("audio_av_modulation", [1, 4 * ad])
    let videoGate = input("video_av_gate", [1, vd]), audioGate = input("audio_av_gate", [1, ad])
    let videoText = input("video_text", [nt, vd]), audioText = input("audio_text", [nt, ad])
    // Flatten token and head axes for split rotary slices; values may differ per head.
    let videoCos = input("video_rope_cos", [nv * heads, vh / 2])
    let videoSin = input("video_rope_sin", [nv * heads, vh / 2])
    let audioCos = input("audio_rope_cos", [na * heads, ah / 2])
    let audioSin = input("audio_rope_sin", [na * heads, ah / 2])
    let videoCrossCos = input("video_cross_rope_cos", [nv * heads, ah / 2])
    let videoCrossSin = input("video_cross_rope_sin", [nv * heads, ah / 2])
    let audioCrossCos = input("audio_cross_rope_cos", [na * heads, ah / 2])
    let audioCrossSin = input("audio_cross_rope_sin", [na * heads, ah / 2])

    func parameter(_ name: String, _ shape: [Int], storageShape: [Int]? = nil) -> Model.IO {
      let storage = storageShape ?? shape
      let layer = Parameter<Float>(.GPU(0), format: .NHWC, shape: TensorShape(storage), name: name)
      bindings.append(Binding(layer: layer, name: name, shape: shape, storageShape: storage, bias: false, projection: false))
      return layer.io
    }
    func dense(_ name: String, _ x: ModelIOConvertible, _ width: Int, _ count: Int,
      bias: Bool = true) -> Model.IO {
      let layer = Dense(count: count, noBias: !bias, name: name)
      bindings.append(Binding(layer: layer, name: name + ".weight", shape: [count, width],
        storageShape: [count, width], bias: false, projection: true))
      if bias { bindings.append(Binding(layer: layer, name: name + ".bias", shape: [count],
        storageShape: [count], bias: true, projection: true)) }
      // Projection weights and operands use the selected dtype. Return to
      // Float32 BEFORE normalization, attention, gates, GELU or residual sums.
      if precision == .float32 { return layer(x) }
      return layer(x.to(precision.projectionType)).to(.Float32)
    }
    func norm(_ x: ModelIOConvertible) -> Model.IO {
      RMSNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)(x)
    }
    func learnedNorm(_ name: String, _ x: Model.IO, _ width: Int) -> Model.IO {
      norm(x) .* parameter(name + ".weight", [width], storageShape: [1, width])
    }
    func row(_ x: ModelIOConvertible, _ index: Int, _ width: Int, _ count: Int) -> Model.IO {
      x.reshaped([1, count * width]).reshaped([1, width], offset: [0, index * width], strides: [count * width, 1])
    }
    func modulations(_ name: String, _ x: ModelIOConvertible, _ count: Int, _ width: Int) -> [Model.IO] {
      let table = parameter(name, [count, width]).reshaped([1, count * width])
      return (0..<count).map { row(x, $0, width, count) + row(table, $0, width, count) }
    }
    let vm = modulations("scale_shift_table", videoMods, 9, vd)
    let am = modulations("audio_scale_shift_table", audioMods, 9, ad)
    let vp = modulations("prompt_scale_shift_table", videoPrompt, 2, vd)
    let ap = modulations("audio_prompt_scale_shift_table", audioPrompt, 2, ad)
    let vav = parameter("scale_shift_table_a2v_ca_video", [5, vd])
    let aav = parameter("scale_shift_table_a2v_ca_audio", [5, ad])
    let vc = (0..<4).map { row(videoAV, $0, vd, 4) + row(vav, $0, vd, 5) }
    let ac = (0..<4).map { row(audioAV, $0, ad, 4) + row(aav, $0, ad, 5) }
    let vg = videoGate + row(vav, 4, vd, 5), ag = audioGate + row(aav, 4, ad, 5)

    func rotary(_ x: Model.IO, _ tokens: Int, _ width: Int, _ cos: Input, _ sin: Input) -> Model.IO {
      let flat = x.reshaped([tokens * heads, width])
      let first = flat.reshaped([tokens * heads, width / 2], offset: [0, 0], strides: [width, 1]).contiguous()
      let second = flat.reshaped([tokens * heads, width / 2], offset: [0, width / 2], strides: [width, 1]).contiguous()
      return Functional.concat(axis: 1, first .* cos - second .* sin, first .* sin + second .* cos)
        .reshaped([1, tokens, heads, width])
    }
    func attention(_ name: String, _ query: Model.IO, _ context: Model.IO,
      _ queryRows: Int, _ keyRows: Int, _ queryWidth: Int, _ keyWidth: Int, _ headWidth: Int,
      queryRoPE: (Input, Input)? = nil, keyRoPE: (Input, Input)? = nil) -> Model.IO {
      let inner = heads * headWidth
      let qp = dense(name + ".to_q", query, queryWidth, inner)
      let kp = dense(name + ".to_k", context, keyWidth, inner)
      let vp = dense(name + ".to_v", context, keyWidth, inner)
      var q = learnedNorm(name + ".q_norm", qp, inner)
      var k = learnedNorm(name + ".k_norm", kp, inner)
      let v = vp.reshaped([1, keyRows, heads, headWidth])
      if let (cos, sin) = queryRoPE { q = rotary(q, queryRows, headWidth, cos, sin) }
      else { q = q.reshaped([1, queryRows, heads, headWidth]) }
      if let (cos, sin) = keyRoPE { k = rotary(k, keyRows, headWidth, cos, sin) }
      else { k = k.reshaped([1, keyRows, heads, headWidth]) }
      if sequenceAttention {
        // Order producers while preserving the complete Q/K/V tensors and the
        // pre-cross-modal streams shared by both attention directions.
        kp.add(dependencies: [q]); vp.add(dependencies: [k])
      }
      let values = ScaledDotProductAttention(scale: 1 / Float(headWidth).squareRoot(), flags: [])(q, k, v)
      let gate = 2 * dense(name + ".to_gate_logits", query, queryWidth, heads).sigmoid()
      let gated = values .* gate.reshaped([1, queryRows, heads, 1])
      return dense(name + ".to_out", gated.reshaped([queryRows, inner]), inner, queryWidth)
    }
    func normalized(_ x: ModelIOConvertible, _ shift: Model.IO, _ scale: Model.IO) -> Model.IO {
      norm(x) .* (1 + scale) + shift
    }
    var debug: [(String, Model.IO)] = []
    func mark(_ name: String, _ x: Model.IO) -> Model.IO { debug.append((name, x)); return x }
    let vn = normalized(video, vm[0], vm[1]), an = normalized(audio, am[0], am[1])
    var vx = mark("video_self", video + attention("attn1", vn, vn, nv, nv, vd, vd, vh,
      queryRoPE: (videoCos, videoSin), keyRoPE: (videoCos, videoSin)) .* vm[2])
    var ax = mark("audio_self", audio + attention("audio_attn1", an, an, na, na, ad, ad, ah,
      queryRoPE: (audioCos, audioSin), keyRoPE: (audioCos, audioSin)) .* am[2])
    vx = mark("video_text", vx + attention("attn2", normalized(vx, vm[6], vm[7]),
      videoText .* (1 + vp[1]) + vp[0], nv, nt, vd, vd, vh) .* vm[8])
    ax = mark("audio_text", ax + attention("audio_attn2", normalized(ax, am[6], am[7]),
      audioText .* (1 + ap[1]) + ap[0], na, nt, ad, ad, ah) .* am[8])
    // Both cross-modal directions read these SAME pre-A2V normalized streams.
    let vshared = norm(vx), ashared = norm(ax)
    vx = mark("video_cross", vx + attention("audio_to_video_attn", vshared .* (1 + vc[0]) + vc[1],
      ashared .* (1 + ac[0]) + ac[1], nv, na, vd, ad, ah,
      queryRoPE: (videoCrossCos, videoCrossSin), keyRoPE: (audioCrossCos, audioCrossSin)) .* vg)
    ax = mark("audio_cross", ax + attention("video_to_audio_attn", ashared .* (1 + ac[2]) + ac[3],
      vshared .* (1 + vc[2]) + vc[3], na, nv, ad, vd, ah,
      queryRoPE: (audioCrossCos, audioCrossSin), keyRoPE: (videoCrossCos, videoCrossSin)) .* ag)
    func feedForward(_ name: String, _ x: Model.IO, _ dimension: Int, bias: Bool) -> Model.IO {
      let expanded = dense(name + ".proj_in", x, dimension, dimension * 4, bias: bias)
      return dense(name + ".proj_out", expanded.GELU(approximate: .tanh), dimension * 4, dimension, bias: bias)
    }
    vx = mark("video", vx + feedForward("ff", normalized(vx, vm[3], vm[4]), vd, bias: false) .* vm[5])
    ax = mark("audio", ax + feedForward("audio_ff", normalized(ax, am[3], am[4]), ad, bias: true) .* am[5])
    let outputs = diagnostics ? debug : [("video", vx), ("audio", ax)]
    model = Model(inputs, outputs.map(\.1))
    model.maxConcurrency = .limit(1)
    self.inputNames = names; inputShapes = shapes; self.bindings = bindings
    outputNames = outputs.map(\.0)
  }
}
