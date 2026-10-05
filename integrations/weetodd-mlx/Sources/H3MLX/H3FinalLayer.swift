import Foundation
import MLX
import MLXNN
import TensorIO

/// H3's shared final modulation and separate FP32 velocity heads.
public enum H3FinalLayer {
  public struct Output {
    public let video: MLXArray
    public let audio: MLXArray
    public init(video: MLXArray, audio: MLXArray) {
      self.video = video
      self.audio = audio
    }
  }

  public static func evaluate(checkpointURL: URL, input: MLXArray,
    timeEmbeddings: MLXArray, timestepIndices: MLXArray,
    videoIndices: MLXArray, audioIndices: MLXArray) throws -> Output {
    try evaluate(checkpointURL: checkpointURL, input: input,
      timeEmbeddings: timeEmbeddings, timestepIndices: timestepIndices,
      videoIndices: videoIndices, audioIndices: audioIndices,
      observe: { _, _ in })
  }

  static func evaluate(checkpointURL: URL, input: MLXArray,
    timeEmbeddings: MLXArray, timestepIndices: MLXArray,
    videoIndices: MLXArray, audioIndices: MLXArray,
    maximumRows: Int = 40_000, lora: (any H3LoRAApplying)? = nil, loraInput: MLXArray? = nil, observe: (String, MLXArray) throws -> Void) throws -> Output {
    guard input.ndim == 3, input.shape[0] == 1,
      [40_000,64_000].contains(maximumRows), (1...maximumRows).contains(input.shape[1]), input.shape[2] == 5376,
      input.dtype == .bfloat16, timeEmbeddings.ndim == 2,
      (1...128).contains(timeEmbeddings.shape[0]),
      [64, 2688].contains(timeEmbeddings.shape[1]),
      timeEmbeddings.dtype.isFloatingPoint,
      timestepIndices.shape == [input.shape[1]],
      timestepIndices.dtype == .int32,
      videoIndices.ndim == 1, videoIndices.dtype == .int32,
      audioIndices.ndim == 1, audioIndices.dtype == .int32 else {
      throw H3CheckpointError.invalid("Invalid H3 final layer inputs.")
    }
    let timesteps = timestepIndices.asArray(Int32.self)
    let videos = videoIndices.asArray(Int32.self)
    let audios = audioIndices.asArray(Int32.self)
    guard timesteps.allSatisfy({ (0..<timeEmbeddings.shape[0]).contains(Int($0)) }),
      videos.allSatisfy({ (0..<input.shape[1]).contains(Int($0)) }),
      audios.allSatisfy({ (0..<input.shape[1]).contains(Int($0)) }) else {
      throw H3CheckpointError.invalid("H3 final output index exceeds packed rows.")
    }
    try Task.checkCancellation()
    let layout = try H3CheckpointLayout(url: checkpointURL)
    guard timeEmbeddings.shape[1] == (layout.curveRank ?? 2688) else {
      throw H3CheckpointError.invalid("H3 final AdaLN coordinates differ from the checkpoint.")
    }
    let tensorURL = try H3CheckpointSource.fileURL(checkpointURL)
    let file = try SafeTensorFile(url: tensorURL)
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
    }
    let prefix = layout.prefix + "final_layer."
    func read(_ suffix: String, shape: [Int], dtype: String) throws -> MLXArray {
      let name = prefix + suffix
      guard let descriptor = file.tensors[name], descriptor.dtype == dtype,
        descriptor.shape == shape.map(UInt64.init) else {
        throw H3CheckpointError.invalid("Missing H3 final layer tensor: \(suffix)")
      }
      let value = try file.withTensorBytes(named: name) { bytes in
        dtype == "F32"
          ? MLXArray(bytes, shape, type: Float.self)
          : MLXArray(bytes, shape, type: UInt16.self).view(dtype: .bfloat16)
      }
      eval(value)
      return value
    }
    let modWeight = try read("adaln_proj.linear.weight",
      shape: [10752, layout.curveRank ?? 2688],
      dtype: layout.curveRank == nil ? "BF16" : "F32")
    let modBias = try read("adaln_proj.linear.bias",
      shape: [10752], dtype: layout.curveRank == nil ? "BF16" : "F32")
    let activated = layout.curveRank == nil
      ? silu(timeEmbeddings.asType(.float32)).asType(.bfloat16)
      : timeEmbeddings.asType(.float32)
    let baseModulation = addMM(modBias, activated, modWeight.T)
    let modulation = try lora?.apply(base: baseModulation, input: loraInput ?? activated,
      target: "diffusion_model.final_layer.adaln_proj.linear", reorderQKV: false) ?? baseModulation
    eval(modulation)
    try observe("modulation", modulation)
    let shift = take(modulation[0..<timeEmbeddings.shape[0], 0..<5376],
      timestepIndices, axis: 0)
    let scale = take(modulation[0..<timeEmbeddings.shape[0], 5376..<10752],
      timestepIndices, axis: 0)
    let normWeight = try read("norm.weight", shape: [5376], dtype: "BF16")
    let normalized = MLXFast.rmsNorm(input, weight: normWeight, eps: 1e-5)
      * (1 + scale) + shift
    eval(normalized)
    try observe("normalized", normalized)
    func outputHead(_ suffix: String, rows: Int) throws -> MLXArray {
      let weight = try read(suffix + ".weight", shape: [rows, 5376],
        dtype: layout.curveRank == nil ? "BF16" : "F32").asType(.float32)
      let bias = try read(suffix + ".bias", shape: [rows], dtype: layout.fastVariant == nil ? "F32" : "BF16").asType(.float32)
      let projected = addMM(bias, normalized.asType(.float32), weight.T)
      eval(projected)
      return projected
    }
    let video = try outputHead("video_out", rows: 96)
    let audio = try outputHead("audio_out", rows: 32)
    let result = Output(video: take(video, videoIndices, axis: 1),
      audio: take(audio, audioIndices, axis: 1))
    eval(result.video, result.audio)
    try file.checkUnchanged(at: tensorURL)
    try H3CheckpointSource.checkUnchanged(checkpointURL)
    try Task.checkCancellation()
    return result
  }
}
