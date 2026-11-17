import Foundation
import ImageLayout
import MLXGuidedGeneration
import XCTest
@testable import MOSSImageMLX

final class DenseAnalysisTests: XCTestCase {
  func testFaceCropUsesDetectorEvidenceInsideTheSubjectAndPadsWithinCanvas() {
    let subject = NormalizedLayoutRect(x: 0.3, y: 0.2, width: 0.5, height: 0.8)
    let face = NormalizedLayoutRect(x: 0.4, y: 0.3, width: 0.12, height: 0.18)
    let unrelated = NormalizedLayoutRect(x: 0, y: 0, width: 0.25, height: 0.25)
    XCTAssertEqual(MOSSLayoutDetailGeometry.face(in: subject, candidates: [unrelated, face]), face)
    XCTAssertNil(MOSSLayoutDetailGeometry.face(in: subject, candidates: [unrelated]))
    let second = NormalizedLayoutRect(x: 0.6, y: 0.5, width: 0.15, height: 0.2)
    XCTAssertEqual(MOSSLayoutDetailGeometry.face(in: subject, candidates: [second, face], preferred: face), face)
    let padded = MOSSLayoutDetailGeometry.padded(unrelated, fraction: 0.35)
    XCTAssertEqual(padded.x, 0); XCTAssertEqual(padded.y, 0)
    XCTAssertEqual(padded.width, 0.3375, accuracy: 1e-9)
  }
  func testDenseGrammarSupportsOwnedDetailsAndMissingLetteringWithoutOCRBoilerplate() throws {
    let schema = MOSSLayoutDensePrompt.schema(groupCount: 1, family: .qwen35_4b)
    XCTAssertFalse(schema.contains("ocr_ids"))
    let vocab = (32...126).map { String(UnicodeScalar($0)!) } + ["<eos>"]
    let tokenizer = try GrammarTokenizer(vocab: vocab, vocabType: .raw, eosTokenId: 95)
    let grammar = try GrammarConstraint(tokenizer: tokenizer, jsonSchema: schema)
    let json = #"{"caption":"A cover","palette":[],"warnings":[],"regions":[{"id":1,"parent":-1,"label":"character","caption":"A woman with dark curled hair","bbox":[100,200,900,980],"layer":1},{"id":2,"parent":1,"label":"object","caption":"Her raised hand holds a cigarette","bbox":[100,400,300,700],"layer":2}],"text_styles":[{"reading":"TITLE","caption":"Bold white letters","color":"white","font":"serif","weight":"bold","case":"uppercase"}],"extra_text_regions":[{"text":"CLASSIC","caption":"Cream ornamental serif title","bbox":[0,0,1000,150]}]}"#
    for char in json.unicodeScalars { _ = try grammar.commitToken(Int32(char.value - 32)) }
    XCTAssertTrue(try grammar.commitToken(95).isTerminated)
    XCTAssertTrue(MOSSLayoutDensePrompt.system(for: .gemma4_12b).contains("box_2d is [top,left,bottom,right]"))
    let gemma = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(MOSSLayoutDensePrompt.schema(groupCount:0,family:.gemma4_12b).utf8)) as? [String:Any])
    let properties = try XCTUnwrap(gemma["properties"] as? [String:[String:Any]])
    let extra = try XCTUnwrap(properties["extra_text_regions"]?["items"] as? [String:Any])
    XCTAssertEqual(Set(extra["required"] as! [String]),Set(["text","caption","box_2d"]))
  }
  func testDenseParentCycleGetsOneStructuralRepair() async throws {
    let valid = #"{"caption":"Portrait","palette":[],"warnings":[],"regions":[{"id":1,"parent":-1,"label":"character","caption":"Man","bbox":[100,0,900,1000],"layer":1}],"text_styles":[],"extra_text_regions":[]}"#
    let result = try await MOSSLayoutAnalyzer.validatedAnalysis(width: 1872, height: 1056, ocr: [], dense: true) { attempt, feedback in
      if attempt == 0 { return valid.replacingOccurrences(of: "\"parent\":-1", with: "\"parent\":1") }
      XCTAssertNotNil(feedback); return valid
    }
    XCTAssertEqual(result.attempts, 2)
    XCTAssertNil(result.analysis.draft.sourceReconstruction)
  }
}
