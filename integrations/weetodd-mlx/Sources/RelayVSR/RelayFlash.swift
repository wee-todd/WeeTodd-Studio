import Foundation
import MLX
import MLXNN

private let flashEpsilon: Float = 1e-5
private let flashDimension = 512

private struct FlashBlock {
  let query: RelayLinear
  let key: RelayLinear
  let value: RelayLinear
  let conditionKey: RelayLinear
  let conditionValue: RelayLinear
  let output: RelayLinear
  let attentionNorm: MLXArray
  let keyNorm: MLXArray
  let valueNorm: MLXArray
  let feedForwardNorm: MLXArray
  let gateUp: RelayLinear
  let down: RelayLinear

  init(_ weights: RelayWeights, index: Int) throws {
    let block = "backbone.blocks.\(index)"
    let path = block + ".attention"
    query = try weights.linear(path + ".query", flashDimension, flashDimension)
    key = try weights.linear(path + ".key", flashDimension, 128)
    value = try weights.linear(path + ".value", flashDimension, 128)
    conditionKey = try weights.linear(path + ".condition_key", flashDimension, 128)
    conditionValue = try weights.linear(path + ".condition_value", flashDimension, 128)
    output = try weights.linear(path + ".output", flashDimension, flashDimension)
    attentionNorm = try weights.tensor(block + ".attention_norm.weight", [flashDimension])
    keyNorm = try weights.tensor(path + ".key_norm.weight", [64])
    valueNorm = try weights.tensor(path + ".value_norm.weight", [64])
    feedForwardNorm = try weights.tensor(block + ".feed_forward_norm.weight", [flashDimension])
    gateUp = try weights.linear(block + ".feed_forward.gate_up_proj", flashDimension, 2816, bias: false)
    down = try weights.linear(block + ".feed_forward.down", 1408, flashDimension, bias: false)
  }

  func callAsFunction(_ x: MLXArray, rope: RelayRotary, pastRope: RelayRotary,
    endpoints: [RelayKV], past: RelayKV?, plan: RelayAttentionPlan) -> (MLXArray, RelayKV)
  {
    let source = MLXFast.rmsNorm(x, weight: attentionNorm, eps: flashEpsilon)
    let conditioning = endpoints.isEmpty
    let q = rope.apply(query(source).reshaped([-1, 8, 64]))
    let k = (conditioning ? conditionKey : key)(source).reshaped([-1, 2, 64])
    let v = (conditioning ? conditionValue : value)(source).reshaped([-1, 2, 64])
    let kv = RelayKV(key: MLXFast.rmsNorm(k, weight: keyNorm, eps: flashEpsilon),
      value: MLXFast.rmsNorm(v, weight: valueNorm, eps: flashEpsilon))
    let rotated = RelayKV(key: rope.apply(kv.key), value: kv.value)
    var visible = endpoints
    if let past { visible.append(RelayKV(key: pastRope.apply(past.key), value: past.value)) }
    visible.append(rotated)
    let attended = plan(q, visible).reshaped([-1, flashDimension])
    let residual = x + output(attended)
    let expanded = gateUp(MLXFast.rmsNorm(residual, weight: feedForwardNorm, eps: flashEpsilon))
    let (gate, up) = expanded.split(axis: -1)
    return (residual + down(silu(gate) * up), conditioning ? rotated : kv)
  }
}

final class RelayFlash {
  private let blocks: [FlashBlock]
  private let encoder: [RelayConv]
  private let encoderNorm: MLXArray
  private let conditionEncoder: RelayConv
  private let conditionNorm: MLXArray
  private let conditionType: MLXArray
  private let videoType: MLXArray
  private let spatialHead: RelayLinear
  private let residual: RelayLinear
  private let height: Int
  private let width: Int
  private let rope: RelayRotary
  private let pastRope: RelayRotary
  private let conditionPlan: RelayAttentionPlan
  private let videoPlans: [RelayAttentionPlan]
  private var history: [RelayKV] = []

  init(weights url: URL, geometry: RelayGeometry) throws {
    let weights = try RelayWeights(url: url, format: "relayvsr-flash-x2-v1")
    blocks = try (0..<10).map { try FlashBlock(weights, index: $0) }
    encoder = try [
      weights.convolution("lr_encoder.layers.0", 3, 64, kernel: 3, stride: 2, padding: 1),
      weights.convolution("lr_encoder.layers.2", 64, 128, kernel: 3, stride: 2, padding: 1),
      weights.convolution("lr_encoder.layers.4", 128, 128, kernel: 3, stride: 2, padding: 1),
      weights.convolution("lr_encoder.layers.6", 128, flashDimension, kernel: 1),
    ]
    encoderNorm = try weights.tensor("lr_encoder.norm.weight", [flashDimension])
    conditionEncoder = try weights.convolution("condition_encoder.projection", 48, flashDimension, kernel: 1)
    conditionNorm = try weights.tensor("condition_encoder.norm.weight", [flashDimension])
    conditionType = try weights.tensor("condition_type_embedding", [1, 1, flashDimension])
      .reshaped([flashDimension])
    videoType = try weights.tensor("video_type_embedding", [1, 1, flashDimension])
      .reshaped([flashDimension])
    spatialHead = try weights.linear("spatial_head.0", flashDimension, 1024)
    residual = try weights.linear("residual_projection", 1024, 768)
    let h = geometry.paddedHeight / 8
    let w = geometry.paddedWidth / 8
    height = h
    width = w
    rope = RelayRotary(height: h, width: w, time: 0, axes: [16, 24, 24])
    pastRope = RelayRotary(height: h, width: w, time: -1, axes: [16, 24, 24])
    let reference = RelayWindow(height: 48, width: 48, slidesAtEdge: false)
    let video = RelayWindow(height: 24, width: 24, slidesAtEdge: false)
    conditionPlan = RelayAttentionPlan(height: h, width: w, windows: [reference])
    videoPlans = [[reference, reference, video], [reference, reference, video, video]].map {
      RelayAttentionPlan(height: h, width: w, windows: $0)
    }
  }

  func prepareEndpoint(_ latent: MLXArray) throws -> [RelayKV] {
    let projected = conditionEncoder(latent).reshaped([-1, flashDimension])
    var x = MLXFast.rmsNorm(projected, weight: conditionNorm, eps: flashEpsilon) + conditionType
    var endpoint: [RelayKV] = []
    for block in blocks {
      try Task.checkCancellation()
      let (next, kv) = block(x, rope: rope, pastRope: pastRope, endpoints: [],
        past: nil, plan: conditionPlan)
      eval(next, kv.key, kv.value)
      endpoint.append(kv)
      x = next
    }
    return endpoint
  }

  func beginWindow() { history.removeAll(keepingCapacity: true) }

  func frame(_ input: MLXArray, start: [RelayKV], end: [RelayKV], detail: Float) throws -> MLXArray {
    var features = input.asType(.bfloat16)
    for convolution in encoder.dropLast() { features = silu(convolution(features)) }
    let projected = encoder[3](features).reshaped([-1, flashDimension])
    var x = MLXFast.rmsNorm(projected, weight: encoderNorm, eps: flashEpsilon) + videoType
    let hasPast = !history.isEmpty
    var nextHistory: [RelayKV] = []
    for i in blocks.indices {
      try Task.checkCancellation()
      let (next, kv) = blocks[i](x, rope: rope, pastRope: pastRope,
        endpoints: [start[i], end[i]], past: hasPast ? history[i] : nil,
        plan: videoPlans[hasPast ? 1 : 0])
      eval(next, kv.key, kv.value)
      nextHistory.append(kv)
      x = next
    }
    history = nextHistory
    // The projection stores 16×16 subpixels per channel, with channel first.
    let rgb = residual(silu(spatialHead(x))).reshaped([height, width, 3, 16, 16])
      .transposed(0, 3, 1, 4, 2).reshaped([height * 16, width * 16, 3])
    return RelayBicubic.upsample(input) + rgb.asType(.float32) * detail
  }
}

enum RelayBicubic {
  private static let kernel = MLXFast.metalKernel(name: "relay_bicubic_2x",
    inputNames: ["rgb"], outputNames: ["output"],
    source: """
      uint i = thread_position_in_grid.x;
      int h = rgb_shape[0], w = rgb_shape[1];
      if (i >= uint(h * w * 12)) return;
      int c = i % 3, x = (i / 3) % (w * 2), y = i / (w * 6);
      float sx = (float(x) + 0.5f) * 0.5f - 0.5f;
      float sy = (float(y) + 0.5f) * 0.5f - 0.5f;
      int bx = int(floor(sx)), by = int(floor(sy));
      float result = 0.0f;
      for (int dy = -1; dy <= 2; ++dy) {
        int iy = clamp(by + dy, 0, h - 1);
        float wy = relay_cubic(sy - float(by + dy));
        for (int dx = -1; dx <= 2; ++dx) {
          int ix = clamp(bx + dx, 0, w - 1);
          result += wy * relay_cubic(sx - float(bx + dx)) * rgb[(iy * w + ix) * 3 + c];
        }
      }
      output[i] = result;
      """,
    header: """
      inline float relay_cubic(float x) {
        const float a = -0.75f;
        x = abs(x);
        if (x <= 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
        if (x < 2.0f) return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
        return 0.0f;
      }
      """)

  static func upsample(_ input: MLXArray) -> MLXArray {
    kernel([input], grid: (input.size * 4, 1, 1), threadGroup: (256, 1, 1),
      outputShapes: [[input.dim(0) * 2, input.dim(1) * 2, 3]], outputDTypes: [.float32])[0]
  }
}
