import XCTest
@testable import LTX25Text

final class TextTests: XCTestCase {
  func testRMSAndInterleaving() throws {
    let result = try TextMath.interleavedStates([[3, 4], [0, 2]], tokens: 1, width: 2)
    let a = Float(12.5 + 1e-6).squareRoot()
    XCTAssertEqual(result[0], 3/a, accuracy: 1e-6)
    XCTAssertEqual(result[1], 0)
    XCTAssertEqual(result[2], 4/a, accuracy: 1e-6)
    XCTAssertEqual(result[3], Float(2)/Float(2 + 1e-6).squareRoot(), accuracy: 1e-6)
  }
  func testCausalGroupedAttention() throws {
    let y = try TextMath.attention(q: [1, 2, 3, 4], k: [1, 2], v: [10, 20], tokens: 2, heads: 2, kvHeads: 1, width: 1, scale: 1, window: 1)
    XCTAssertEqual(y, [10, 10, 20, 20])
  }
  func testProportionalRotaryUsesFullHeadDenominator() {
    let x = [Float](repeating: 1, count: 16)
    let y = TextMath.gemmaRotary(x, tokens: 2, heads: 1, width: 8, theta: 10000, fraction: 0.5)
    XCTAssertEqual(y[8], cos(1)-sin(1), accuracy: 1e-6)
    XCTAssertEqual(y[9], cos(0.1)-sin(0.1), accuracy: 1e-6)
    XCTAssertEqual(y[10], 1)
    XCTAssertEqual(y[14], 1)
  }
  func testRejectInvalidShape() {
    XCTAssertThrowsError(try TextMath.interleavedStates([[1]], tokens: 2, width: 3))
  }
}

import TensorIO

extension TextTests {
  private func fixture(_ name: String) throws -> SafeTensorFile {
    let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "safetensors", subdirectory: "Fixtures"))
    return try SafeTensorFile(url: url)
  }
  private func assertNear(_ result: [Float], _ expected: [Float], tolerance: Float = 1e-4) {
    XCTAssertEqual(result.count, expected.count)
    let maxError = zip(result, expected).map { abs($0.0-$0.1) }.max() ?? 0
    XCTAssertLessThan(maxError, tolerance, "maximum absolute error \(maxError)")
  }
  func testGemmaSlidingAndGlobalAgainstMLX() throws {
    for full in [false,true] {
      let file = try fixture(full ? "gemma-full" : "gemma-sliding")
      var c = GemmaLayerConfiguration()
      c.width = 8; c.hidden = 12; c.heads = 4; c.kvHeads = full ? 1 : 2; c.headWidth = 4
      c.keyEqualsValue = full; c.rotaryFraction = full ? 0.5 : 1; c.theta = full ? 1e6 : 1e4; c.window = full ? nil : 2
      let output = try GemmaLayer.evaluate(file.readFloat32(named: "input"), tokens: 3, configuration: c,
        weights: TextWeights(file: file, prefix: ""), gpu: TextMatrixGPU())
      assertNear(output, try file.readFloat32(named: "expected"))
    }
  }
  func testConnectorAgainstMLX() throws {
    let file = try fixture("connector")
    let output = try TextConnector.evaluate(file.readFloat32(named: "input"), tokens: 3, width: 8, heads: 2,
      weights: TextWeights(file: file, prefix: ""), gpu: TextMatrixGPU())
    assertNear(output, try file.readFloat32(named: "expected"))
  }
  func testInstalledTokenizerAgainstHuggingFace() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_TEST_GEMMA_ROOT"] else {
      throw XCTSkip("Set WEETODD_TEST_GEMMA_ROOT to qualify embedded installed tokenizer.")
    }
    let file = try SafeTensorFile(url: URL(fileURLWithPath: path).appendingPathComponent("pages/fixed.safetensors"))
    let tokenizer = try GemmaTokenizer(fixed: file)
    struct Reference: Decodable { let prompt: String; let ids: [Int] }
    let url = try XCTUnwrap(Bundle.module.url(forResource: "tokenizer-reference", withExtension: "json", subdirectory: "Fixtures"))
    for reference in try JSONDecoder().decode([Reference].self, from: Data(contentsOf: url)) {
      XCTAssertEqual(try tokenizer.encode(reference.prompt), reference.ids, reference.prompt)
      XCTAssertEqual(try tokenizer.encode(reference.prompt,maxLength: 2), Array(reference.ids.prefix(2)))
    }
    XCTAssertThrowsError(try tokenizer.encode("a",maxLength: 0))
  }
  func testInstalledFullPrompt() throws {
    guard ProcessInfo.processInfo.environment["WEETODD_TEST_FULL_TEXT"] == "1",
      let root = ProcessInfo.processInfo.environment["WEETODD_TEST_GEMMA_ROOT"],
      let connector = ProcessInfo.processInfo.environment["WEETODD_TEST_CONNECTOR"] else {
      throw XCTSkip("Full installed text qualification is opt-in.")
    }
    let encoder = try LTX25TextEncoder(gemmaRoot: URL(fileURLWithPath: root), connectorURL: URL(fileURLWithPath: connector))
    let prompt: String
    if let file = ProcessInfo.processInfo.environment["WEETODD_TEST_TEXT_PROMPT_FILE"] {
      prompt = try String(contentsOfFile:file,encoding:.utf8)
    } else { prompt = ProcessInfo.processInfo.environment["WEETODD_TEST_TEXT_PROMPT"] ?? "A red fox." }
    var checkedReentry = false
    var denied = TextEncodingConfiguration(); denied.maximumOwnedBufferBytes = 1
    var admissionProgress = 0
    XCTAssertThrowsError(try encoder.encode(prompt: prompt, configuration: denied) { _ in admissionProgress += 1 })
    XCTAssertEqual(admissionProgress, 0)
    let output = try encoder.encode(prompt: prompt) { event in
      if !checkedReentry {
        checkedReentry = true
        XCTAssertThrowsError(try encoder.encode(prompt: "This nested encode must be rejected."))
      }
      print("TEXT_PROGRESS \(event.stage) \(event.completed)/\(event.total)")
    }
    XCTAssertEqual(output.videoShape,[1,1024,4096]); XCTAssertEqual(output.audioShape,[1,1024,2048])
    XCTAssertTrue(output.video.allSatisfy(\.isFinite)); XCTAssertTrue(output.audio.allSatisfy(\.isFinite))
    if let reference = ProcessInfo.processInfo.environment["WEETODD_TEST_TEXT_REFERENCE"] {
      let oracle = try SafeTensorFile(url:URL(fileURLWithPath:reference))
      // Frozen before the long boxing-prompt holdout: accumulated 48-layer +
      // 8-connector FP32 execution has a separate final-context budget.
      for (name,actual) in [("video",output.video),("audio",output.audio)] {
        let expected = try oracle.readFloat32(named:name)
        assertNear(actual,expected,tolerance:1e-3)
        XCTAssertLessThan(relativeL2(actual,expected),1e-4,name)
      }
      let expectedIDs = try oracle.withTensorBytes(named:"token_ids") { bytes in
        stride(from:0,to:bytes.count,by:4).map { Int(bytes.loadUnaligned(fromByteOffset:$0,as:Int32.self).littleEndian) }
      }
      XCTAssertEqual(output.tokenIDs,expectedIDs)
    }
    if let outputPath = ProcessInfo.processInfo.environment["WEETODD_TEST_TEXT_OUTPUT"] {
      for (name,values) in [("video",output.video),("audio",output.audio)] {
        let data = values.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: outputPath + "-" + name + ".f32"))
      }
    }
  }
}

extension TextTests {
  func testEmbeddedBPEMergesSpecialsByteFallbackAndTruncation() throws {
    var vocab = ["<pad>": 0, "<bos>": 1, "a": 2, "b": 3, "▁": 4, "ab": 5, "▁ab": 6]
    for byte in 0...255 { vocab[String(format: "<0x%02X>",byte)] = byte+10 }
    let json: [String: Any] = [
      "model": ["type": "BPE", "byte_fallback": true, "ignore_merges": false, "vocab": vocab,
        "merges": [["a","b"],["▁","ab"]]],
      "normalizer": ["type": "Replace", "pattern": ["String": " "], "content": "▁"],
      "pre_tokenizer": ["type": "Split", "pattern": ["String": " "], "behavior": "MergedWithPrevious", "invert": false],
      "post_processor": ["type": "TemplateProcessing", "single": [["Sequence": ["id": "A"]]]],
      "added_tokens": [["content": "<bos>", "id": 1, "normalized": false, "single_word": false, "lstrip": false, "rstrip": false]]]
    let tokenizer = try GemmaTokenizer(tokenizerJSON: JSONSerialization.data(withJSONObject: json),
      configurationJSON: JSONSerialization.data(withJSONObject: ["bos_token": "<bos>","pad_token": "<pad>"]))
    XCTAssertEqual(try tokenizer.encode(" ab ab "),[1,5,6])
    XCTAssertEqual(try tokenizer.encode("<bos>ab"),[1,5])
    XCTAssertEqual(try tokenizer.encode("ab ab", maxLength: 2),[1,5])
    XCTAssertEqual(try tokenizer.encode("🧪"),[1]+Array("🧪".utf8).map { Int($0)+10 })
    XCTAssertEqual(try tokenizer.encode(""),[1])
  }
}

extension TextTests {
  func testVocabularyPreservesBOMAndCanonicallyEquivalentUTF8Keys() throws {
    let json = Data(#"{"model":{"vocab":{"é":1,"e\u0301":2,"\uFEFF":3,"\uFEFF\uFEFF":4}}}"#.utf8)
    let vocabulary = try ExactJSONVocabulary.decode(json)
    XCTAssertEqual(vocabulary.count,4)
    XCTAssertEqual(vocabulary[Data("é".utf8)],1)
    XCTAssertEqual(vocabulary[Data("e\u{0301}".utf8)],2)
    XCTAssertEqual(vocabulary[Data("\u{FEFF}".utf8)],3)
    XCTAssertEqual(vocabulary[Data("\u{FEFF}\u{FEFF}".utf8)],4)
    XCTAssertNil(vocabulary[Data()])
  }
}

extension TextTests {
  func testInstalledComponentsAgainstExactMLXInputs() throws {
    guard let reference = ProcessInfo.processInfo.environment["WEETODD_TEST_TEXT_COMPONENTS"],
      let root = ProcessInfo.processInfo.environment["WEETODD_TEST_GEMMA_ROOT"],
      let connector = ProcessInfo.processInfo.environment["WEETODD_TEST_CONNECTOR"] else {
      throw XCTSkip("Set installed Gemma, connector and exact-input component reference paths.")
    }
    let oracle = try SafeTensorFile(url: URL(fileURLWithPath: reference))
    let gpu = try TextMatrixGPU()
    for index in [0,5,47] {
      let file = try SafeTensorFile(url: URL(fileURLWithPath: root).appendingPathComponent(String(format:"pages/layer-%03d.safetensors",index)))
      var config = GemmaLayerConfiguration()
      if index % 6 == 5 {
        config.headWidth = 512; config.kvHeads = 1; config.keyEqualsValue = true
        config.window = nil; config.theta = 1e6; config.rotaryFraction = 0.25
      }
      let input = try oracle.readFloat32(named:"gemma_input_\(index)")
      let result = try GemmaLayer.evaluate(input,tokens:input.count/3840,configuration:config,
        weights:TextWeights(file:file,prefix:"model.layers.\(index)."),gpu:gpu)
      let expected = try oracle.readFloat32(named:"gemma_expected_\(index)")
      print("COMPONENT_GEMMA_\(index)_MAXABS \(zip(result,expected).map { abs($0.0-$0.1) }.max() ?? 0)")
      assertNear(result,expected)
    }
    let file = try SafeTensorFile(url:URL(fileURLWithPath:connector))
    for (name,width) in [("video",4096),("audio",2048)] {
      let result = try TextConnector.evaluate(oracle.readFloat32(named:name+"_connector_input"),tokens:1024,width:width,
        weights:TextWeights(file:file,prefix:"model.diffusion_model.\(name)_embeddings_connector.transformer_1d_blocks.0."),gpu:gpu)
      if let base = ProcessInfo.processInfo.environment["WEETODD_TEST_TEXT_COMPONENT_OUTPUT"] {
        try result.withUnsafeBytes { try Data($0).write(to:URL(fileURLWithPath:base+"-"+name+".f32")) }
      }
      let expected = try oracle.readFloat32(named:name+"_connector_expected")
      print("COMPONENT_\(name)_CONNECTOR_MAXABS \(zip(result,expected).map { abs($0.0-$0.1) }.max() ?? 0)")
      // Raw video activations reach 444; the original absolute 1e-4 gate
      // failed even between MLX CPU/GPU. Preserve that historical result in
      // qualification.json; the scaled gate retains absolute 1e-4 near zero.
      let scaled = zip(result,expected).map { abs(Double($0.0)-Double($0.1))/max(abs(Double($0.1)),1) }.max() ?? 0
      XCTAssertLessThanOrEqual(scaled,1e-4,name)
      XCTAssertLessThanOrEqual(relativeL2(result,expected),1e-4,name)
      assertNear(TextMath.rms(result,width:width),TextMath.rms(expected,width:width))
    }
  }
  private func relativeL2(_ actual: [Float], _ expected: [Float]) -> Double {
    let error = zip(actual,expected).reduce(0.0) { $0 + pow(Double($1.0)-Double($1.1),2) }
    let reference = expected.reduce(0.0) { $0 + Double($1)*Double($1) }
    return sqrt(error/max(reference,1e-30))
  }
}


extension TextTests {
  func testReentrantEncodingIsRejectedAndFailureReleasesGate() throws {
    let gate = TextExecutionGate()
    try gate.run {
      XCTAssertThrowsError(try gate.run { XCTFail("Reentrant weighted stage must never start.") })
    }
    enum SyntheticFailure: Error { case expected }
    XCTAssertThrowsError(try gate.run { throw SyntheticFailure.expected })
    XCTAssertEqual(try gate.run { 7 },7)
  }
}
