import Foundation
import ImageIO
import ImageLayout
import CoreGraphics
import MLX
import MLXGuidedGeneration
import XCTest
@testable import MOSSImageMLX

final class LayoutAnalysisTests: XCTestCase {
  func testTextRectangleOvershootIsClippedWithoutAdmittingEmptyOrNonfiniteBoxes() throws {
    let bounds = try XCTUnwrap(MOSSLayoutVision.clippedTextBounds(CGRect(x:-0.02,y:0.15,width:0.99,height:0.92)))
    XCTAssertEqual(bounds.x,0);XCTAssertEqual(bounds.y,0,accuracy:1e-9)
    XCTAssertEqual(bounds.width,0.97,accuracy:1e-9);XCTAssertEqual(bounds.height,0.85,accuracy:1e-9)
    XCTAssertNil(MOSSLayoutVision.clippedTextBounds(CGRect(x:-2,y:0,width:0.1,height:0.1)))
    XCTAssertNil(MOSSLayoutVision.clippedTextBounds(CGRect(x:0,y:0,width:Double.nan,height:1)))
  }
  func testHeadingCropRecoversOrnateTitleFromLowResolutionCoverWhenProvided() throws {
    guard let path = ProcessInfo.processInfo.environment["MOSS_LAYOUT_HEADING_FIXTURE"] else { throw XCTSkip("Optional private source-heading fixture") }
    let source = try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath:path) as CFURL,nil))
    let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source,0,nil))
    let detected = try MOSSLayoutVision.recognize(image)
    let title = try XCTUnwrap(detected.first{$0.text == "CLASSIC"},"Detected: \(detected)")
    XCTAssertGreaterThan(title.bounds.width,0.8);XCTAssertLessThan(title.bounds.y,0.1)
  }
  func testGemmaPromptGrammarAndParserUseNativeYFirstContract() async throws {
    XCTAssertTrue(MOSSLayoutPrompt.system(for: .gemma4_12b).contains("box_2d is [top,left,bottom,right]"))
    XCTAssertTrue(MOSSLayoutPrompt.system(for: .qwen35_4b).contains("bbox is [left,top,right,bottom]"))
    let schema = MOSSLayoutPrompt.schema(groups: [[0]], family: .gemma4_12b)
    let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(schema.utf8)) as? [String: Any])
    let properties = try XCTUnwrap(root["properties"] as? [String: [String: Any]])
    let visual = try XCTUnwrap(properties["regions"]?["items"] as? [String: Any])
    XCTAssertTrue((visual["required"] as? [String])?.contains("box_2d") == true)
    XCTAssertFalse((visual["required"] as? [String])?.contains("bbox") == true)
    let response = #"{"caption":"Scene","palette":[],"warnings":[],"regions":[{"label":"character","caption":"Left figure","box_2d":[215,0,980,330],"text":"","ocr_ids":[]}]}"#
    let result = try await MOSSLayoutAnalyzer.validatedAnalysis(width: 1872, height: 1056, ocr: [], family: .gemma4_12b) { _, _ in response }
    XCTAssertEqual(result.analysis.draft.regions[0].bounds, .init(x: 0, y: 0.215, width: 0.33, height: 0.765))
  }
  func testInstalledGemmaHeaderAdmissionWhenProvided() throws {
    guard let path = ProcessInfo.processInfo.environment["MOSS_LAYOUT_GEMMA_MODEL"] else { throw XCTSkip("Optional installed vision checkpoint header check") }
    let model = try MOSSLayoutModelIdentity(directory: URL(fileURLWithPath: path))
    XCTAssertEqual(model.family, .gemma4_12b)
    XCTAssertEqual(model.index.descriptors["vision_embedder.patch_dense.weight"]?.dtype, "U32")
    XCTAssertNoThrow(try model.checkUnchanged())
  }
  func testGemmaUsesChatEndRatherThanTextEOSAndNeverItsVocabularyForMOSSCounting() throws {
    let tokens = ["<bos>": 2, "<eos>": 1, "<|turn>": 105, "<turn|>": 106,
      "<|image>": 255999, "<|image|>": 258880, "<image|>": 258882]
    XCTAssertEqual(try MOSSLayoutTokenizer.chatStop(family: .gemma4_12b, tokens: tokens, declaredEOS: 1), 106)
    var wrong = tokens; wrong["<turn|>"] = 50
    XCTAssertThrowsError(try MOSSLayoutTokenizer.chatStop(family: .gemma4_12b, tokens: wrong, declaredEOS: 1))
    XCTAssertThrowsError(try MOSSLayoutTokenBudget.directory(provided: nil, model: URL(fileURLWithPath: "/gemma"), family: .gemma4_12b))
    XCTAssertEqual(try MOSSLayoutTokenBudget.directory(provided: nil, model: URL(fileURLWithPath: "/qwen"), family: .qwen35_4b).path, "/qwen")
    XCTAssertThrowsError(try MOSSLayoutTokenBudget.validateVocabulary(tokens))
    var request = MOSSLayoutAnalysisRequest(requestID: UUID().uuidString, sourcePath: "/image.png",
      sourceSHA256: String(repeating: "a", count: 64), modelDirectory: "/gemma")
    request.mossTokenizerDirectory = "relative/tokenizer"
    XCTAssertThrowsError(try request.validate())
  }
  func testGrammarAdmitsLiteralOCRTextVariant() throws {
    let vocab = (32...126).map { String(UnicodeScalar($0)!) } + ["<eos>"]
    let tokenizer = try GrammarTokenizer(vocab: vocab, vocabType: .raw, eosTokenId: 95)
    let grammar = try GrammarConstraint(tokenizer: tokenizer, jsonSchema: MOSSLayoutPrompt.schema(groups: [[2]]))
    let json = #"{"caption":"A","palette":[],"regions":[{"label":"object","caption":"B","bbox":[1,2,3,4],"text":"","ocr_ids":[]}],"text_regions":[{"label":"text","style":{"color":"gold","font":"serif","weight":"bold","case":"uppercase"},"bbox":[0,0,1000,1000],"text":"","ocr_ids":[2]}],"warnings":[]}"#
    for char in json.unicodeScalars { _ = try grammar.commitToken(Int32(char.value - 32)) }
    XCTAssertTrue(try grammar.commitToken(95).isTerminated)
  }
  func testGrammarSuppressesPathologicalWhitespaceButPreservesValidJSONSpaces() throws {
    let vocab = [" ", "\"", "a", "<eos>"]
    let tokenizer = try GrammarTokenizer(vocab: vocab, vocabType: .raw, eosTokenId: 3)
    let grammar = try GrammarConstraint(tokenizer: tokenizer, jsonSchema: #"{"type":"string","maxLength":16}"#)
    let processor = MOSSLayoutJSONProcessor(grammar: grammar, vocabularySize: 4, eos: 3, whitespaceTokenIDs: [0])
    let logits = MLXArray([Float(10), 0, 0, 0])
    processor.didSample(token: MLXArray(1))
    XCTAssertGreaterThan(processor.process(logits: logits)[0].item(Float.self), 0)
    for _ in 0..<3 { processor.didSample(token: MLXArray(0)) }
    XCTAssertLessThan(processor.process(logits: logits)[0].item(Float.self), -100)
    processor.didSample(token: MLXArray(2))
    XCTAssertLessThan(processor.process(logits: logits)[0].item(Float.self), -100)
    XCTAssertNil(processor.failure)
  }
  func testAnalysisGrammarBoundsRepetitionAndSeparatesTextEvidenceFromObjects() throws {
    let schema = MOSSLayoutPrompt.schema(groups: [[2], [8]])
    let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(schema.utf8)) as? [String: Any])
    let properties = try XCTUnwrap(root["properties"] as? [String: Any])
    XCTAssertEqual((properties["caption"] as? [String: Any])?["maxLength"] as? Int, 1100)
    let regions = try XCTUnwrap(properties["regions"] as? [String: Any])
    XCTAssertEqual(regions["maxItems"] as? Int, 12)
    let physical = try XCTUnwrap((regions["items"] as? [String: Any])?["properties"] as? [String: [String: Any]])
    XCTAssertEqual(physical["ocr_ids"]?["maxItems"] as? Int, 0)
    let textRegions = try XCTUnwrap(properties["text_regions"] as? [String: Any])
    XCTAssertEqual(textRegions["maxItems"] as? Int, 2)
    let slots = try XCTUnwrap(textRegions["prefixItems"] as? [[String: Any]])
    let text = try XCTUnwrap(slots.first?["properties"] as? [String: [String: Any]])
    XCTAssertEqual(text["ocr_ids"]?["const"] as? [Int], [2])
    let style = try XCTUnwrap(text["style"]?["properties"] as? [String: [String: Any]])
    XCTAssertEqual(style["color"]?["enum"] as? [String], ImageLayoutTypography.colors)
    XCTAssertTrue(schema.range(of: "\"label\"")!.lowerBound < schema.range(of: "\"bbox\"")!.lowerBound)
  }
  func testOneStructuralRepairIsValidatedAndNeverReturnsPartialOrLoops() async throws {
    let valid = #"{"caption":"Scene","palette":[],"warnings":[],"regions":[{"label":"object","caption":"cup","bbox":[100,100,600,700],"text":"","ocr_ids":[]}]}"#
    let result = try await MOSSLayoutAnalyzer.validatedAnalysis(width: 1024, height: 1024, ocr: []) { attempt, feedback in
      if attempt == 0 { return valid.replacingOccurrences(of: "[100,100,600,700]", with: "[600,100,100,700]") }
      XCTAssertNotNil(feedback); return valid
    }
    XCTAssertEqual(result.attempts, 2); XCTAssertEqual(result.analysis.draft.regions.count, 1)
    do {
      _ = try await MOSSLayoutAnalyzer.validatedAnalysis(width: 1024, height: 1024, ocr: []) { _, _ in "{}" }
      XCTFail("Two invalid proposals must fail")
    } catch { XCTAssertTrue(error.localizedDescription.contains("repair")) }
  }
  func testAnalyzerAdmitsThePublishedChatStopAndVisionTokenIDs() throws {
    XCTAssertNoThrow(try MOSSLayoutTokenizer.validateSpecialTokens([
      "<|im_start|>": 248045, "<|im_end|>": 248046, "<|vision_start|>": 248053,
      "<|vision_end|>": 248054, "<|image_pad|>": 248056], eos: 248046))
    XCTAssertThrowsError(try MOSSLayoutTokenizer.validateSpecialTokens([
      "<|im_start|>": 248045, "<|im_end|>": 248046, "<|vision_start|>": 248053,
      "<|vision_end|>": 248054, "<|image_pad|>": 248056], eos: 248044))
  }
  func testAnalysisRequestRequiresBoundLocalIdentityAndSmallBudget() throws {
    var request = MOSSLayoutAnalysisRequest(requestID: UUID().uuidString, sourcePath: "/images/source.jpg",
      sourceSHA256: String(repeating: "a", count: 64), modelDirectory: "/models/qwen")
    XCTAssertNoThrow(try request.validate())
    request.sourcePath = "https://example.com/image.jpg"; XCTAssertThrowsError(try request.validate())
    request.sourcePath = "/images/source.jpg"; request.maxTokens = 100_000; XCTAssertThrowsError(try request.validate())
    request.maxTokens = 4096; request.targetPixels = 8_000_000; XCTAssertThrowsError(try request.validate())
  }
  func testOrientedSnapshotPreservesWholeImageAndRejectsChangedSource() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let source = folder.appendingPathComponent("rotated.jpg"), output = folder.appendingPathComponent("snapshot.png")
    let bytes = Data([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
      255, 0, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255])
    let image = try XCTUnwrap(CGImage(width: 4, height: 2, bitsPerComponent: 8, bitsPerPixel: 32,
      bytesPerRow: 16, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: .init(rawValue: CGImageAlphaInfo.last.rawValue),
      provider: CGDataProvider(data: bytes as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(source as CFURL, "public.jpeg" as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: 6] as CFDictionary)
    XCTAssertTrue(CGImageDestinationFinalize(destination))
    let hash = try MOSSLayoutImage.sha256(source)
    let snapshot = try MOSSLayoutImage.prepare(source: source, expectedSHA256: hash, output: output)
    XCTAssertEqual(snapshot.width, 2); XCTAssertEqual(snapshot.height, 4)
    XCTAssertEqual(snapshot.sourceSHA256, hash)
    XCTAssertTrue(FileManager.default.fileExists(atPath: output.path))
    XCTAssertThrowsError(try MOSSLayoutImage.prepare(source: source, expectedSHA256: String(repeating: "b", count: 64),
      output: folder.appendingPathComponent("bad.png")))
    XCTAssertFalse(FileManager.default.fileExists(atPath: folder.appendingPathComponent("bad.png").path))
  }
  func testAnalyzerRejectsTextOnlyOrUnrelatedModelBeforeWeights() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    try Data(#"{"model_type":"qwen3_5","text_config":{"hidden_size":2560}}"#.utf8).write(to: folder.appendingPathComponent("config.json"))
    XCTAssertThrowsError(try MOSSLayoutModelIdentity(directory: folder))
  }
}
