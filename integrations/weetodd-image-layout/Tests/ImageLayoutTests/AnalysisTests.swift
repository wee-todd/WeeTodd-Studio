import Foundation
import XCTest
@testable import ImageLayout

final class AnalysisTests: XCTestCase {
  func testGemmaNativeBoxesConvertOnceWhileOCRAndDefaultXYXYStayUnchanged() throws {
    let evidence = [LayoutOCRObservation(id: 0, text: "TITLE", bounds: .init(x: 0.1, y: 0.2, width: 0.6, height: 0.1), confidence: 1)]
    let response = #"{"caption":"Scene","palette":[],"warnings":[],"regions":[{"label":"character","caption":"Left figure","box_2d":[215,0,980,330],"text":"","ocr_ids":[]}],"text_regions":[{"label":"text","caption":"white lettering","bbox":[0,0,1000,1000],"text":"","ocr_ids":[0]}]}"#
    let result = try ImageLayoutAnalysis.parse(Data(response.utf8), width: 1872, height: 1056, ocr: evidence, visualBoxFormat: .gemmaYXYX)
    let bounds = result.draft.regions[0].bounds
    XCTAssertEqual(bounds.x, 0); XCTAssertEqual(bounds.y, 0.215)
    XCTAssertEqual(bounds.width, 0.33); XCTAssertEqual(bounds.height, 0.765)
    let textBounds = result.draft.regions[1].bounds
    XCTAssertEqual(textBounds.x, evidence[0].bounds.x); XCTAssertEqual(textBounds.y, evidence[0].bounds.y)
    XCTAssertEqual(textBounds.width, evidence[0].bounds.width, accuracy: 1e-12)
    XCTAssertEqual(textBounds.height, evidence[0].bounds.height, accuracy: 1e-12)
    XCTAssertEqual(result.draft.regions[1].text, "TITLE")
    XCTAssertThrowsError(try ImageLayoutAnalysis.parse(Data(response.utf8), width: 1872, height: 1056, ocr: evidence))
    let qwen = response.replacingOccurrences(of: "\"box_2d\":[215,0,980,330]", with: "\"bbox\":[0,215,330,980]")
    XCTAssertEqual(try ImageLayoutAnalysis.parse(Data(qwen.utf8), width: 1872, height: 1056, ocr: evidence).draft.regions[0].bounds, bounds)
    XCTAssertThrowsError(try ImageLayoutAnalysis.parse(Data(qwen.utf8), width: 1872, height: 1056, ocr: evidence, visualBoxFormat: .gemmaYXYX))
    let duplicate = response.replacingOccurrences(of: "\"box_2d\":[215,0,980,330]", with: "\"box_2d\":[215,0,980,330],\"box_2d\":[0,0,1,1]")
    XCTAssertThrowsError(try ImageLayoutAnalysis.parse(Data(duplicate.utf8), width: 1872, height: 1056, ocr: evidence, visualBoxFormat: .gemmaYXYX))
  }
  func testStructuredTypographyCannotInjectAlternateWordsAndDuplicateTextObjectsAreRemoved() throws {
    let ocr = [LayoutOCRObservation(id: 0, text: "TITLE", bounds: .init(x: 0.1, y: 0.1, width: 0.8, height: 0.1), confidence: 1)]
    let response = #"{"caption":"A cover","palette":[],"warnings":[" "],"regions":[{"label":"object","caption":"Large gold title lettering","bbox":[100,100,900,200],"text":"","ocr_ids":[]},{"label":"character","caption":"A woman","bbox":[200,200,800,1000],"text":"","ocr_ids":[]}],"text_regions":[{"label":"text","style":{"color":"gold","font":"serif","weight":"bold","case":"uppercase"},"bbox":[0,0,1000,1000],"text":"","ocr_ids":[0]}]}"#
    let result = try ImageLayoutAnalysis.parse(Data(response.utf8), width: 1248, height: 1616, ocr: ocr)
    XCTAssertEqual(result.draft.regions.map(\.type), [.character,.text])
    XCTAssertEqual(result.draft.regions.last?.caption, "gold, bold serif uppercase lettering")
    XCTAssertEqual(result.draft.regions.last?.text, "TITLE")
    XCTAssertTrue(result.warnings.contains { $0.contains("duplicate lettering") })
    let physical = response.replacingOccurrences(of: "Large gold title lettering", with: "A gold bottle with printed letters")
    XCTAssertEqual(try ImageLayoutAnalysis.parse(Data(physical.utf8), width: 1248, height: 1616, ocr: ocr).draft.regions.count, 3)
    let plaque = response.replacingOccurrences(of: "Large gold title lettering", with: "A brass plaque with engraved lettering")
    XCTAssertEqual(try ImageLayoutAnalysis.parse(Data(plaque.utf8), width: 1248, height: 1616, ocr: ocr).draft.regions.count, 3)
    XCTAssertThrowsError(try ImageLayoutAnalysis.parse(Data(response.replacingOccurrences(of: "\"color\":\"gold\"", with: "\"color\":\"OTHER WORDS\"").utf8), width: 1248, height: 1616, ocr: ocr))
  }
  func testOCRRowsAndDenseGroupsAreStableAndPreserveEveryLiteralOnce() throws {
    let evidence = [
      LayoutOCRObservation(id: 0, text: "A", bounds: .init(x: 0.8, y: 0.10, width: 0.1, height: 0.1), confidence: 1),
      LayoutOCRObservation(id: 1, text: "B", bounds: .init(x: 0.5, y: 0.13, width: 0.1, height: 0.1), confidence: 1),
      LayoutOCRObservation(id: 2, text: "C", bounds: .init(x: 0.2, y: 0.16, width: 0.1, height: 0.1), confidence: 1)]
    let expected = ImageLayoutOCR.readingLines(evidence).map { $0.map(\.id) }
    for order in [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]] {
      let shuffled = order.map { evidence[$0] }
      XCTAssertEqual(ImageLayoutOCR.readingLines(shuffled).map { $0.map(\.id) }, expected)
      let groups = try ImageLayoutOCR.groups(shuffled, maximum: 2)
      XCTAssertEqual(groups.count, 2)
      XCTAssertEqual(groups.flatMap { $0.map(\.id) }.sorted(), [0,1,2])
    }
  }
  func testSeparateVisualAndTextGroupsCannotCrowdOutSubjects() throws {
    let evidence = [LayoutOCRObservation(id: 0, text: "TITLE", bounds: .init(x: 0.1, y: 0.1, width: 0.8, height: 0.1), confidence: 1)]
    let response = #"{"caption":"A cover","palette":[],"warnings":[],"regions":[{"label":"character","caption":"woman","bbox":[200,200,800,1000],"text":"","ocr_ids":[]}],"text_regions":[{"label":"text","caption":"gold serif","bbox":[100,100,900,200],"text":"","ocr_ids":[0]}]}"#
    let parsed = try ImageLayoutAnalysis.parse(Data(response.utf8), width: 1248, height: 1616, ocr: evidence)
    XCTAssertEqual(parsed.draft.regions.map(\.type), [.character, .text])
    XCTAssertEqual(parsed.draft.regions.last?.text, "TITLE")
    XCTAssertThrowsError(try ImageLayoutAnalysis.parse(Data(response.replacingOccurrences(of: "\"label\":\"character\"", with: "\"label\":\"text\"").utf8), width: 1248, height: 1616, ocr: evidence))
  }
  func testTwoMegapixelTargetsMinimizeRatioErrorWithinAreaTolerance() throws {
    for (w, h) in [(825, 433), (735, 952), (504, 680), (800, 453), (236, 295), (501, 700),
      (780, 439), (1200, 675), (2000, 1333), (320, 180), (524, 740), (640, 480)] {
      let size = try ImageLayoutAnalysisGeometry.targetSize(sourceWidth: w, sourceHeight: h)
      XCTAssertEqual(size.width % 16, 0); XCTAssertEqual(size.height % 16, 0)
      XCTAssertTrue((1_960_000...2_040_000).contains(size.width * size.height))
      XCTAssertLessThan(abs(Double(size.width) / Double(size.height) / (Double(w) / Double(h)) - 1), 0.0045)
    }
    let landscape = try ImageLayoutAnalysisGeometry.targetSize(sourceWidth: 1200, sourceHeight: 675)
    XCTAssertEqual(landscape.width, 1872); XCTAssertEqual(landscape.height, 1056)
    let portrait = try ImageLayoutAnalysisGeometry.targetSize(sourceWidth: 735, sourceHeight: 952)
    XCTAssertEqual(portrait.width, 1248); XCTAssertEqual(portrait.height, 1616)
    XCTAssertThrowsError(try ImageLayoutAnalysisGeometry.targetSize(sourceWidth: 0, sourceHeight: 10))
    XCTAssertThrowsError(try ImageLayoutAnalysisGeometry.targetSize(sourceWidth: Int.max, sourceHeight: 1))
  }
  func testSideLimitedPanoramasPreserveRatioBeforeArea() throws {
    for (w, h) in [(10_000, 100), (100, 10_000)] {
      let size = try ImageLayoutAnalysisGeometry.targetSize(sourceWidth: w, sourceHeight: h)
      XCTAssertEqual(Double(size.width) / Double(size.height), Double(w) / Double(h), accuracy: 1e-12)
      XCTAssertLessThanOrEqual(max(size.width, size.height), 4096)
      XCTAssertEqual(size.width * size.height, 102_400)
    }
  }
  func testVisionLowerLeftConvertsOnceToTopLeft() throws {
    let box = try ImageLayoutAnalysisGeometry.visionBounds(x: 0.1, y: 0.7, width: 0.5, height: 0.2)
    XCTAssertEqual(box.x, 0.1); XCTAssertEqual(box.y, 0.1, accuracy: 1e-12)
    XCTAssertEqual(box.width, 0.5); XCTAssertEqual(box.height, 0.2)
    XCTAssertThrowsError(try ImageLayoutAnalysisGeometry.visionBounds(x: 0.5, y: 0.7, width: 0.6, height: 0.2))
  }
  func testVisionBoundaryRoundoffProducesCompilerSafeBounds() throws {
    let bounds = try ImageLayoutAnalysisGeometry.visionBounds(x: 0, y: 0.9, width: 1, height: 0.1)
    XCTAssertEqual(bounds.y, 0)
    let draft = ImageLayoutDraft(width: 1024, height: 1024, caption: "A heading",
      regions: [.init(type: .text, caption: "black lettering", text: "TITLE", bounds: bounds)])
    XCTAssertNoThrow(try MOSSLayoutCompiler.compile(draft))
  }
  func testOCREvidenceIsLiteralAndItsGeometryWinsOverModelGuess() throws {
    let evidence = [
      LayoutOCRObservation(id: 0, text: "Café • 茶", bounds: .init(x: 0.1, y: 0.1, width: 0.4, height: 0.04), confidence: 0.95),
      LayoutOCRObservation(id: 1, text: "TEA & LIGHT", bounds: .init(x: 0.1, y: 0.15, width: 0.5, height: 0.04), confidence: 0.6),
    ]
    let data = Data(#"{"caption":"A cream tea poster","palette":["cream","brown"],"warnings":[],"regions":[{"label":"text","caption":"dark serif lettering","bbox":[0,0,500,300],"text":"WRONG MODEL WORDS","ocr_ids":[1,0]}]}"#.utf8)
    let result = try ImageLayoutAnalysis.parse(data, width: 1248, height: 1616, ocr: evidence)
    XCTAssertEqual(result.draft.regions[0].text, "Café • 茶\nTEA & LIGHT")
    XCTAssertEqual(result.draft.regions[0].bounds.x, 0.1)
    XCTAssertEqual(result.draft.regions[0].bounds.y, 0.1)
    XCTAssertEqual(result.draft.regions[0].bounds.width, 0.5, accuracy: 1e-12)
    XCTAssertEqual(result.draft.regions[0].bounds.height, 0.09, accuracy: 1e-12)
    XCTAssertFalse(result.warnings.isEmpty)
    XCTAssertTrue(try MOSSLayoutCompiler.compile(result.draft).effectivePrompt.contains("Café"))
  }
  func testMalformedAndInventedEvidenceCannotBecomePartialLayout() throws {
    let valid = #"{"caption":"Scene","palette":[],"warnings":[],"regions":[{"label":"object","caption":"cup","bbox":[100,100,600,700],"text":"","ocr_ids":[]}]}"#
    let invalid = [
      valid.replacingOccurrences(of: #""caption":"Scene""#, with: #""caption":"Scene","caption":"Other""#),
      valid.replacingOccurrences(of: #""palette":[]"#, with: #""palette":[],"tools":[] "#),
      valid.replacingOccurrences(of: "[100,100,600,700]", with: "[600,100,100,700]"),
      valid.replacingOccurrences(of: "[100,100,600,700]", with: "[-1,100,600,700]"),
      valid.replacingOccurrences(of: #""ocr_ids":[]"#, with: #""ocr_ids":[42]"#),
      String(valid.dropLast()),
    ]
    for text in invalid { XCTAssertThrowsError(try ImageLayoutAnalysis.parse(Data(text.utf8), width: 1024, height: 1024, ocr: [])) }
  }
  func testAnalysisProvenanceRoundtripsButDoesNotAlterCompiledPromptAndCanUndo() throws {
    let before = ImageLayoutDraft(width: 1024, height: 1024, caption: "Original")
    var after = before
    after.analysis = .init(requestID: UUID().uuidString, sourceSHA256: String(repeating: "a", count: 64),
      sourceWidth: 800, sourceHeight: 453, sourceName: "source.jpg", modelIdentity: "model", recipe: "v1", warnings: [])
    XCTAssertEqual(try MOSSLayoutCompiler.compile(before), try MOSSLayoutCompiler.compile(after))
    XCTAssertEqual(try JSONDecoder().decode(ImageLayoutDraft.self, from: JSONEncoder().encode(after)), after)
    var history = ImageLayoutHistory(); history.record(before: before, after: after)
    XCTAssertEqual(history.undo(current: after), before)
    XCTAssertEqual(history.redo(current: before), after)
    XCTAssertNil(try JSONDecoder().decode(ImageLayoutDraft.self, from: JSONEncoder().encode(before)).analysis)
  }
  func testExternalAnalysisApplicationIsOneUndoAndDoesNotBreakRedoWhenObservedAgain() throws {
    let original = ImageLayoutDraft(width: 1024, height: 1024, caption: "Keep me")
    var replacement = original; replacement.width = 1872; replacement.height = 1056
    replacement.summary.caption = "Extracted scene"; replacement.guideImagePath = "/cache/source.png"
    replacement.analysis = .init(requestID: "analysis", sourceSHA256: String(repeating: "a", count: 64),
      sourceWidth: 800, sourceHeight: 453, sourceName: "source.jpg", modelIdentity: "model", recipe: "v1", warnings: [])
    var history = ImageLayoutHistory()
    history.observeAnalysisReplacement(from: original, to: replacement)
    let undone = try XCTUnwrap(history.undo(current: replacement))
    XCTAssertEqual(undone, original)
    history.observeAnalysisReplacement(from: replacement, to: undone)
    let redone = try XCTUnwrap(history.redo(current: undone))
    XCTAssertEqual(redone, replacement)
    history.observeAnalysisReplacement(from: undone, to: redone)
    XCTAssertEqual(history.undo(current: redone), original)
    XCTAssertFalse(history.canUndo)
  }
}
