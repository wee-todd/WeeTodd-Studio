import MLX
import MLXAudioCodecs
import XCTest

@testable import Drama3MLX

final class StreamingCodecTests: XCTestCase {
  func testIrregularStreamMatchesWholeCausalDecodeAndNewState() throws {
    MLXRandom.seed(7)
    var c = FishS1DACBuildConfig()
    c.encoderDim = 4
    c.encoderRates = [2, 2]
    c.latentDim = 16
    c.decoderDim = 32
    c.decoderRates = [2, 2]
    c.encoderTransformerLayers = [0, 0]
    c.decoderTransformerLayers = [0, 0]
    c.semanticCodebookSize = 64
    c.codebookSize = 16
    c.codebookDim = 4
    c.quantizerTransformerDim = 16
    c.quantizerTransformerHeads = 2
    c.quantizerTransformerHeadDim = 8
    c.quantizerTransformerIntermediateSize = 32
    c.quantizerTransformerLayers = 2
    c.quantizerWindowSize = 4
    let model = buildFishS1DAC(c)
    let codes = MLXRandom.randInt(low: 0, high: 16, [1, 10, 13]).asType(.int32)
    let whole = model.decode(codes, featureLengths: MLXArray([Int32(13)])).0.reshaped(-1)
    eval(whole)
    let stream = Drama3CodecStream(model: model)
    var parts: [MLXArray] = []
    var begin = 0
    for length in [1, 3, 2, 5, 2] {
      parts.append(try stream.decode(codes[0, 0..., begin..<(begin + length)]))
      begin += length
    }
    let joined = concatenated(parts)
    XCTAssertEqual(joined.size, whole.size)
    XCTAssertLessThan(abs(joined - whole).max().item(Float.self), 0.0001)
    let fresh = try Drama3CodecStream(model: model).decode(codes[0, 0..., 0..<1])
    XCTAssertLessThan(abs(fresh - parts[0]).max().item(Float.self), 0.00001)
    XCTAssertLessThan(stream.retainedElements, 100000)
  }
}
extension StreamingCodecTests {
  func testRealCheckpointStreamingParityWhenQualified() throws {
    guard let root = ProcessInfo.processInfo.environment["DRAMA3_QUALIFICATION_ROOT"] else {
      throw XCTSkip(
        "Set DRAMA3_QUALIFICATION_ROOT for the optional real-checkpoint codec qualification.")
    }
    let directory = URL(fileURLWithPath: root)
    let codec = try Drama3AudioCodec(at: directory.appendingPathComponent("balanced"))
    let codes = try Drama3AudioCodec.loadCodes(
      directory.appendingPathComponent("clone-001/codes.safetensors"))
    let whole = try codec.decode(codes)
    let stream = codec.makeStream()
    var result: [MLXArray] = []
    var begin = 0
    for length in [1, 3, 7, 13, 31, codes.dim(1)] where begin < codes.dim(1) {
      let end = min(codes.dim(1), begin + length)
      result.append(try stream.decode(codes[0..., begin..<end]))
      begin = end
    }
    let difference = concatenated(result) - whole
    let error = abs(difference).max().item(Float.self)
    let mse = (difference * difference).mean().item(Float.self)
    let power = (whole * whole).mean().item(Float.self)
    let snr = 10 * log10(power / max(mse, 1e-20))
    print(
      "Real codec stream max absolute error: \(error); retained state elements: \(stream.retainedElements); SNR: \(snr) dB"
    )
    // Whole/chunk SDPA kernels use different reduction orders; enforce both peak and relative error.
    XCTAssertLessThan(error, 0.0003)
    XCTAssertGreaterThan(snr, 65)
  }
}
