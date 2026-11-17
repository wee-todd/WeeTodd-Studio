import Foundation
import XCTest
@testable import ImageLayout

final class DenseLayoutTests: XCTestCase {
  func testTinySubtitleContainingTitleWordDoesNotReplaceTheLargeMasthead() throws {
    let ocr = [LayoutOCRObservation(id:0,text:"LIVE THE GOOD LIFE",bounds:.init(x:0.4,y:0.05,width:0.2,height:0.02),confidence:1)]
    let root: [String:Any] = ["caption":"Magazine cover", "palette":[], "warnings":[],
      "regions":[["id":1,"parent":-1,"label":"object","caption":"Wall","bbox":[0,300,1000,1000],"layer":0]],
      "text_styles":[["caption":"Small subtitle","color":"black","font":"serif","weight":"regular","case":"uppercase"]],
      "extra_text_regions":[["text":"LIFE","caption":"Large red masthead","bbox":[50,20,950,220]]]]
    let draft = try ImageLayoutAnalysis.parseDense(JSONSerialization.data(withJSONObject:root),width:1216,height:1648,ocr:ocr).draft
    XCTAssertTrue(draft.regions.contains{$0.text == "LIFE"})
  }
  func testCombinedExtraHeadingDoesNotDuplicateSplitOCRWords() throws {
    let ocr = [LayoutOCRObservation(id:0,text:"NEW",bounds:.init(x:0.1,y:0.02,width:0.3,height:0.15),confidence:1),
      .init(id:1,text:"YORK",bounds:.init(x:0.6,y:0.02,width:0.3,height:0.15),confidence:1)]
    let style: [String:Any] = ["caption":"Bold serif capitals","color":"black","font":"serif","weight":"bold","case":"uppercase"]
    let root: [String:Any] = ["caption":"Magazine cover", "palette":[], "warnings":[],
      "regions":[["id":1,"parent":-1,"label":"object","caption":"Dark wall","bbox":[0,200,1000,1000],"layer":0]],
      "text_styles":[style,style], "extra_text_regions":[["text":"NEW YORK","caption":"Bold title","bbox":[100,20,900,170]]]]
    var draft = try ImageLayoutAnalysis.parseDense(JSONSerialization.data(withJSONObject:root),width:1216,height:1648,ocr:ocr).draft
    XCTAssertEqual(draft.regions.filter{$0.type == .text}.count,2)
    let word = try XCTUnwrap(draft.regions.firstIndex{$0.text == "NEW"});draft.regions[word].text="OLD"
    XCTAssertFalse(try MOSSLayoutCompiler.compile(draft).effectivePrompt.contains("NEW"))
  }
  func testGemmaAdditionalHeadingUsesTheSameYFirstContractAsVisualRegions() throws {
    let root: [String:Any] = ["caption":"Illustrated magazine cover", "palette":[], "warnings":[],
      "regions":[["id":1,"parent":-1,"label":"object","caption":"Dark textured wall","box_2d":[150,0,950,410],"layer":0]],
      "text_styles":[], "extra_text_regions":[["text":"CLASSIC","caption":"Outlined cream serif title","box_2d":[15,55,175,985]]]]
    let draft = try ImageLayoutAnalysis.parseDense(JSONSerialization.data(withJSONObject:root),width:1216,height:1648,ocr:[],visualBoxFormat:.gemmaYXYX).draft
    let title = try XCTUnwrap(draft.regions.first{$0.text == "CLASSIC"})
    XCTAssertEqual(title.bounds.x,0.055,accuracy:1e-9);XCTAssertEqual(title.bounds.y,0.015,accuracy:1e-9)
    XCTAssertEqual(title.bounds.width,0.93,accuracy:1e-9);XCTAssertEqual(title.bounds.height,0.16,accuracy:1e-9)
  }
  func testSmallSubtitleInsideAnOrnateHeadingDoesNotDiscardTheDistinctHeading() throws {
    let ocr = [LayoutOCRObservation(id:0,text:"MOTION PICTURE",bounds:.init(x:0.3,y:0.05,width:0.25,height:0.025),confidence:1)]
    let root: [String:Any] = ["caption":"Magazine cover", "palette":[], "warnings":[],
      "regions":[["id":1,"parent":-1,"label":"object","caption":"Dark wall","bbox":[0,200,1000,1000],"layer":0]],
      "text_styles":[["reading":"MOTION PICTURE","caption":"Small serif subtitle","color":"black","font":"serif","weight":"regular","case":"uppercase"]],
      "extra_text_regions":[["text":"CLASSIC","caption":"Ornate large serif title","bbox":[30,40,980,180]]]]
    let draft = try ImageLayoutAnalysis.parseDense(JSONSerialization.data(withJSONObject:root),width:1216,height:1648,ocr:ocr).draft
    XCTAssertTrue(draft.regions.contains{$0.text == "CLASSIC"});XCTAssertTrue(draft.regions.contains{$0.text == "MOTION PICTURE"})
  }
  func testTypographyDoesNotKeepOldGlyphNamesAfterATitleEdit() {
    let style = ImageLayoutTypography.editableStyle("Muted gold capitals on red. The 'V' is wide and the ‘I’ is a vertical bar.")
    XCTAssertFalse(style.contains("'V'")); XCTAssertFalse(style.contains("‘I’"))
    XCTAssertTrue(style.contains("Muted gold capitals on red"))
  }
  func testCleanupPreservesAnatomicalDetailsPhysicalSignsAndCompleteOCRLines() throws {
    let caption = "A woman with a scar on the left side of her face and a tattoo on the right side of her body"
    XCTAssertEqual(ImageLayoutAnalysis.normalizingSpatialLanguage(caption), caption)
    XCTAssertFalse(ImageLayoutAnalysis.removingLiteralLettering("A poster titled IT", literals: ["IT"]).contains("IT"))
    let whole = LayoutOCRObservation(id: 0, text: "THE GREAT ESCAPE", bounds: .init(x: 0.1, y: 0.1, width: 0.8, height: 0.1), confidence: 0.9)
    let part = LayoutOCRObservation(id: 1, text: "GREAT", bounds: .init(x: 0.35, y: 0.1, width: 0.3, height: 0.1), confidence: 1)
    XCTAssertEqual(try ImageLayoutOCR.merging([part,whole]).map(\.text), ["THE GREAT ESCAPE"])
    let root: [String: Any] = ["caption": "Street", "palette": [], "warnings": [], "regions": [
      ["id": 1, "parent": -1, "label": "object", "caption": "A weathered wooden signboard on a thick metal pole with painted STOP lettering", "bbox": [100,100,900,900], "layer": 1]],
      "text_styles": [], "extra_text_regions": [["text": "STOP", "caption": "White block capitals", "bbox": [200,200,800,400]]]]
    let draft = try ImageLayoutAnalysis.parseDense(JSONSerialization.data(withJSONObject: root), width: 1024, height: 1024, ocr: []).draft
    XCTAssertEqual(draft.regions.count, 2)
    XCTAssertTrue(draft.regions[0].caption.contains("wooden signboard"))
  }
  func testDenseCompilationUpdatesSubjectLocationWordsWhenItsBoxMoves() throws {
    var subject = ImageLayoutRegion(type: .character, caption: "A central woman with her right hand raised", bounds: .init(x: 0.05, y: 0.3, width: 0.3, height: 0.65))
    var draft = ImageLayoutDraft(width: 1200, height: 1696, caption: "A central monochrome woman on paper", regions: [subject])
    draft.analysis = .init(requestID: UUID().uuidString, sourceSHA256: String(repeating: "a", count: 64), sourceWidth: 600, sourceHeight: 848, sourceName: "source.png", modelIdentity: String(repeating: "b", count: 64), recipe: "moss-image-layout-dense-v2", warnings: [])
    XCTAssertTrue(try MOSSLayoutCompiler.compile(draft).effectivePrompt.contains("left side"))
    subject.bounds.x = 0.65; draft = try ImageLayoutEditing.replacing(subject, in: draft)
    let prompt = try MOSSLayoutCompiler.compile(draft).effectivePrompt
    XCTAssertTrue(prompt.contains("right side")); XCTAssertFalse(prompt.contains("central woman")); XCTAssertFalse(prompt.contains("central monochrome"))
    XCTAssertTrue(prompt.contains("right hand raised"))
  }
  func testSpatialProseCannotPinAnEditableSubjectAndAnatomicalHandsRemain() {
    let text = ImageLayoutAnalysis.normalizingSpatialLanguage("A central monochrome woman in the bottom right corner, with her right hand raised.")
    XCTAssertFalse(text.contains("central")); XCTAssertFalse(text.contains("bottom right corner"))
    XCTAssertTrue(text.contains("right hand raised"))
  }
  func testConservativeHeadingCorrectionAndDuplicateDetails() throws {
    XCTAssertTrue(ImageLayoutAnalysis.admittedHeadingReading("RETRO", corrects: "REIRO"))
    XCTAssertFalse(ImageLayoutAnalysis.admittedHeadingReading("HORIZON", corrects: "VINTAGE"))
    XCTAssertFalse(ImageLayoutAnalysis.admittedHeadingReading("Invented body copy", corrects: "Actual body copy"))
    let ocr = [LayoutOCRObservation(id: 0, text: "REIRO", bounds: .init(x: 0.1, y: 0.1, width: 0.8, height: 0.1), confidence: 0.9)]
    let root: [String: Any] = ["caption": "A central retro-style cover", "palette": [], "warnings": [],
      "regions": [["id": 1, "parent": -1, "label": "character", "caption": "A central woman", "bbox": [100,300,900,1000], "layer": 1],
        ["id": 2, "parent": 1, "label": "object", "caption": "Her right hand", "bbox": [200,500,300,700], "layer": 2],
        ["id": 3, "parent": 1, "label": "object", "caption": "Her right hand", "bbox": [200,500,300,700], "layer": 2]],
      "text_styles": [["reading": "RETRO", "caption": "Cutout paper lettering in orange, green and black with cream outlines", "color": "orange", "font": "serif", "weight": "bold", "case": "uppercase"]], "extra_text_regions": []]
    let result = try ImageLayoutAnalysis.parseDense(JSONSerialization.data(withJSONObject: root), width: 1024, height: 1024, ocr: ocr)
    XCTAssertEqual(result.draft.regions.count, 3); XCTAssertEqual(result.draft.regions.last?.text, "RETRO")
    XCTAssertTrue(result.draft.regions.last!.caption.contains("Cutout paper"))
    XCTAssertFalse(result.draft.summary.caption.contains("central"))
    XCTAssertTrue(result.warnings.contains(where: { $0.contains("REIRO → RETRO") }))
  }
  func testTextOnlyArtworkDoesNotBecomeACompetingObjectBox() throws {
    let root: [String: Any] = ["caption": "A cover", "palette": [], "warnings": [],
      "regions": [["id": 1, "parent": -1, "label": "object", "caption": "The textured paper background", "bbox": [0,0,1000,1000], "layer": 0],
        ["id": 2, "parent": -1, "label": "object", "caption": "Stylized 3D block lettering for the word RETRO", "bbox": [50,50,950,250], "layer": 2]],
      "text_styles": [], "extra_text_regions": [["text": "RETRO", "caption": "Orange dimensional serif lettering", "bbox": [50,50,950,250]]]]
    let draft = try ImageLayoutAnalysis.parseDense(JSONSerialization.data(withJSONObject: root), width: 1024, height: 1024, ocr: []).draft
    XCTAssertEqual(draft.regions.count, 2)
    XCTAssertEqual(draft.regions.filter { $0.type == .object }.count, 1)
  }
  func testMovingAndResizingACharacterTransformsItsDetailsAndUndoTogether() throws {
    var parent = ImageLayoutRegion(type: .character, caption: "Woman", bounds: .init(x: 0.1, y: 0.1, width: 0.4, height: 0.8))
    var head = ImageLayoutRegion(type: .object, caption: "Her face", bounds: .init(x: 0.2, y: 0.2, width: 0.1, height: 0.1))
    head.parentID = parent.id
    var eye = ImageLayoutRegion(type: .object, caption: "Her eyes", bounds: .init(x: 0.22, y: 0.22, width: 0.04, height: 0.02))
    eye.parentID = head.id
    var draft = ImageLayoutDraft(width: 1024, height: 1024, caption: "Scene", regions: [parent, head, eye])
    let before = draft
    parent.bounds.x = 0.5
    draft = try ImageLayoutEditing.replacing(parent, in: draft)
    XCTAssertEqual(draft.regions[1].bounds.x, 0.6, accuracy: 1e-10)
    XCTAssertEqual(draft.regions[2].bounds.x, 0.62, accuracy: 1e-10)
    var history = ImageLayoutHistory(); history.record(before: before, after: draft)
    XCTAssertEqual(history.undo(current: draft), before)
    XCTAssertEqual(history.redo(current: before), draft)
    parent.bounds.width = 0.2
    draft = try ImageLayoutEditing.replacing(parent, in: draft)
    XCTAssertEqual(draft.regions[1].bounds.x, 0.55, accuracy: 1e-10)
    XCTAssertEqual(draft.regions[1].bounds.width, 0.05, accuracy: 1e-10)
    XCTAssertEqual(draft.regions[2].bounds.width, 0.02, accuracy: 1e-10)
    XCTAssertNoThrow(try MOSSLayoutCompiler.compile(draft))
    draft.regions[0].parentID = eye.id
    XCTAssertThrowsError(try MOSSLayoutCompiler.compile(draft))
  }
  func testDenseAnalysisRecoversMissingTitleAndKeepsLiteralTextOutOfOtherCaptions() throws {
    let ocr = [LayoutOCRObservation(id: 0, text: "MOTION PICTURE", bounds: .init(x: 0.2, y: 0.01, width: 0.6, height: 0.03), confidence: 1)]
    var regions: [[String: Any]] = (1...14).map { i in
      ["id": i, "parent": -1, "label": i == 1 ? "character" : "object", "caption": i == 1 ? "Woman in a CLASSIC magazine portrait" : "Visible prop \(i)",
        "bbox": [100, 250, 900, 950], "layer": i]
    }
    regions[1]["parent"] = 1; regions[1]["caption"] = "Her face with arched brows and a small closed mouth"
    let root: [String: Any] = ["caption": "CLASSIC cover with MOTION PICTURE lettering", "palette": [], "warnings": [], "regions": regions,
      "text_styles": [["color": "white", "font": "serif", "weight": "bold", "case": "uppercase"]],
      "extra_text_regions": [["text": "CLASSIC", "caption": "large CLASSIC ornate cream serif lettering", "bbox": [30, 40, 980, 180]]]]
    let result = try ImageLayoutAnalysis.parseDense(JSONSerialization.data(withJSONObject: root), width: 1216, height: 1648, ocr: ocr)
    XCTAssertEqual(result.draft.regions.count, 16)
    XCTAssertEqual(result.draft.regions[1].parentID, result.draft.regions[0].id)
    XCTAssertEqual(result.draft.regions.suffix(2).compactMap(\.text), ["MOTION PICTURE", "CLASSIC"])
    XCTAssertFalse(result.draft.summary.caption.contains("CLASSIC"))
    XCTAssertFalse(result.draft.regions[0].caption.contains("CLASSIC"))
    var changed = result.draft
    changed.regions[15].text = "HORIZON"
    let prompt = try MOSSLayoutCompiler.compile(changed).effectivePrompt
    XCTAssertTrue(prompt.contains("HORIZON")); XCTAssertFalse(prompt.contains("CLASSIC"))
    XCTAssertFalse(prompt.contains("parentID"))
    var invalid = root; var bad = regions; bad[0]["parent"] = 2; invalid["regions"] = bad
    XCTAssertThrowsError(try ImageLayoutAnalysis.parseDense(JSONSerialization.data(withJSONObject: invalid), width: 1216, height: 1648, ocr: ocr))
  }
  func testOCRCropDuplicatesMergeWithoutDuplicatingLettering() throws {
    let box = NormalizedLayoutRect(x: 0.1, y: 0.1, width: 0.8, height: 0.1)
    let observations = [LayoutOCRObservation(id: 0, text: "TITLE", bounds: box, confidence: 0.8),
      .init(id: 7, text: "TITLE", bounds: box, confidence: 0.99),
      .init(id: 8, text: "Other words", bounds: .init(x: 0.1, y: 0.3, width: 0.5, height: 0.1), confidence: 0.9)]
    let merged = try ImageLayoutOCR.merging(observations)
    XCTAssertEqual(merged.map(\.text), ["TITLE", "Other words"])
    XCTAssertEqual(merged.map(\.id), [0, 1]); XCTAssertEqual(merged[0].confidence, 0.99)
  }
}
