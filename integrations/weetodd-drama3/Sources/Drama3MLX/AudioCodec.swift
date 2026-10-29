import AVFoundation
import CryptoKit
import Drama3Contracts
import Foundation
import MLX
import MLXAudioCodecs
import MLXNN

public final class Drama3AudioCodec {
  private let model: FishS1DAC
  public init(at root: URL) throws {
    let file = root.appendingPathComponent("codec.safetensors")
    try Drama3CodecAdmission.inspect(file)
    // Drama 3 uses the fixed Fish S1 codec architecture, independent of the
    // speech transformer's config. Require every parameter and exact shape.
    let codec = buildFishS1DAC()
    var weights = codec.sanitize(weights: try loadArrays(url: file))
    // The SDK's causal WN wrappers expose duplicate inner convolution slots,
    // overwritten from V/G/bias on every call. Initialize these aliases from
    // verified tensors too, rather than retaining random constructor arrays.
    for (name, _) in codec.parameters().flattened() where weights[name] == nil {
      if name.hasSuffix(".conv.conv.weight") {
        let prefix = String(name.dropLast(".conv.conv.weight".count))
        if weights[prefix + ".weight_g"] != nil { weights[name] = weights[prefix + ".weight_v"] }
      } else if name.hasSuffix(".conv.conv.bias") {
        let prefix = String(name.dropLast(".conv.conv.bias".count))
        if weights[prefix + ".weight_g"] != nil { weights[name] = weights[prefix + ".bias"] }
      }
    }
    try codec.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    eval(codec.parameters())
    model = codec
  }
  func makeStream() -> Drama3CodecStream { Drama3CodecStream(model: model) }
  public func encode(_ samples: [Float]) throws -> MLXArray {
    guard !samples.isEmpty, samples.count <= 44100 * 600, samples.allSatisfy(\.isFinite) else {
      throw Drama3Error.invalid("Invalid reference audio.")
    }
    let result = model.encode(MLXArray(samples).reshaped(1, 1, samples.count))
    let codes = result.0[0, 0..., ..<result.1[0].item(Int.self)].asType(.int32)
    eval(codes)
    return codes
  }
  public func decode(_ codes: MLXArray) throws -> MLXArray {
    try Self.validate(codes)
    let result = model.decode(
      codes.expandedDimensions(axis: 0), featureLengths: MLXArray([Int32(codes.dim(1))])
    ).0.reshaped(-1)
    eval(result)
    guard MLX.all(MLX.isFinite(result)).item(Bool.self) else {
      throw Drama3Error.invalid("The audio decoder produced non-finite samples.")
    }
    return result
  }
  public static func validate(_ codes: MLXArray) throws {
    guard codes.ndim == 2, codes.dim(0) == 10, codes.dim(1) > 0, codes.dim(1) <= 262144,
      [DType.int32, .int64].contains(codes.dtype), codes.min().item(Int.self) >= 0,
      codes.max().item(Int.self) < 4096
    else {
      throw Drama3Error.invalid("Audio codes must be ten nonempty integer codebooks in 0–4095.")
    }
  }
  public static func loadCodes(_ url: URL) throws -> MLXArray {
    guard (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) <= 24 * 1024 * 1024
    else {
      throw Drama3Error.invalid("The take has oversized audio codes.")
    }
    let header = try Drama3TensorFile(url)
    guard header.tensors.count == 1, let info = header.tensors["codes"],
      info.shape.count == 2, info.shape[0] == 10, (1...262144).contains(info.shape[1]),
      ["I32", "I64"].contains(info.dtype),
      let codes = try loadArrays(url: url)["codes"]
    else { throw Drama3Error.invalid("The take has missing or oversized audio codes.") }
    try validate(codes)
    return codes.asType(.int32)
  }
  public static func saveCodes(_ codes: MLXArray, to url: URL) throws {
    try validate(codes)
    let temporary = url.deletingLastPathComponent().appendingPathComponent(
      ".\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: temporary) }
    try save(arrays: ["codes": codes.asType(.int32)], url: temporary)
    if FileManager.default.fileExists(atPath: url.path) {
      try FileManager.default.removeItem(at: url)
    }
    try FileManager.default.moveItem(at: temporary, to: url)
  }
  public static func readPCM(
    _ url: URL, start: Double = 0, duration: Double? = nil, channel: String = "mix"
  ) throws -> [Float] {
    let file = try AVAudioFile(forReading: url)
    guard file.length > 0, file.processingFormat.sampleRate == 44100,
      file.processingFormat.channelCount == 1
    else {
      throw Drama3Error.invalid("Prepare reference audio as 44.1 kHz mono before native inference.")
    }
    let first = Int64((start * 44100).rounded())
    let last = duration.map { first + Int64(($0 * 44100).rounded()) } ?? file.length
    guard first >= 0, last > first, last <= file.length else {
      throw Drama3Error.invalid("The reference range exceeds its prepared audio.")
    }
    file.framePosition = first
    var result: [Float] = []
    result.reserveCapacity(Int(last - first))
    while file.framePosition < last {
      let count = AVAudioFrameCount(min(65536, last - file.framePosition))
      let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: count)!
      try file.read(into: buffer, frameCount: count)
      guard buffer.frameLength > 0, let data = buffer.floatChannelData?[0] else {
        throw Drama3Error.invalid("Could not read prepared reference audio.")
      }
      result.append(contentsOf: UnsafeBufferPointer(start: data, count: Int(buffer.frameLength)))
    }
    return result
  }
  public static func appendPCM(_ samples: MLXArray, to writer: AVAudioFile) throws {
    let values = samples.asArray(Float.self)
    let buffer = AVAudioPCMBuffer(
      pcmFormat: writer.processingFormat, frameCapacity: AVAudioFrameCount(values.count))!
    buffer.frameLength = AVAudioFrameCount(values.count)
    values.withUnsafeBufferPointer {
      buffer.floatChannelData![0].update(from: $0.baseAddress!, count: values.count)
    }
    try writer.write(from: buffer)
  }
  public static func digest(_ url: URL) throws -> String {
    try Drama3AudioPreparation.fileDigest(url.path)
  }
}
