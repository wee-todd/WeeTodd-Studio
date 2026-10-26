import XCTest
import MLX
@testable import H3MLX

final class H3QwenBudgetTests: XCTestCase {
  private func tokenizer() throws -> H3QwenTokenizer {
    var vocabulary = Dictionary(uniqueKeysWithValues: (33...126).map {
      (String(UnicodeScalar($0)!), Int32($0))
    })
    vocabulary["Ġ"] = 32
    return try H3QwenTokenizer(vocabulary:vocabulary, merges:[], regex:".", specialIDs:[
      "<|vision_start|>":151652, "<|vision_end|>":151653,
      "<|image_pad|>":151655, "<|video_pad|>":151656])
  }

  func testFull1024ReferenceKeepsEveryVisualPadAndTheCompletePrompt() throws {
    let tokenizer = try tokenizer(), prompt = "Replace the cup with a red teapot."
    let grid = H3QwenRequest.Grid(temporal:1,height:64,width:64)
    let request = try H3QwenRequest.references(prompt:prompt,
      references:[.image(grid:grid)], tokenizer:tokenizer)
    XCTAssertEqual(request.visualRanges.count,1)
    XCTAssertEqual(request.visualRanges[0].count,1026)
    XCTAssertEqual(request.tokenIDs.filter { $0 == 151655 }.count,1024)
    let promptIDs = try tokenizer.encode(prompt)
    XCTAssertEqual(Array(request.tokenIDs.suffix(promptIDs.count)),promptIDs)
    XCTAssertGreaterThan(request.tokenIDs.count,1024)
    let positions = try H3QwenMRoPE.positions(request:request,grids:[grid])
    XCTAssertEqual(positions.map(\.count),Array(repeating:request.tokenIDs.count,count:3))
    let first = request.visualRanges[0].lowerBound + 1
    XCTAssertEqual(positions[0][first+1023],positions[0][first])
    XCTAssertEqual(positions[1][first+1023],positions[1][first]+31)
    XCTAssertEqual(positions[2][first+1023],positions[2][first]+31)
    XCTAssertEqual(positions[0].last,positions[1].last)
    XCTAssertEqual(positions[1].last,positions[2].last)
  }

  func testFullVisualBudgetCanCoexistWithTheTextBudgetForRefAndKeyframeRequests() throws {
    let tokenizer = try tokenizer()
    let grid = H3QwenRequest.Grid(temporal:1,height:64,width:64)
    let labelCount = try tokenizer.encode("<Picture 1>: ").count
    let prompt = String(repeating:"x",count:1024-labelCount-2)
    for reference in [true,false] {
      let request = try reference
        ? H3QwenRequest.references(prompt:prompt,references:[.image(grid:grid)],tokenizer:tokenizer)
        : H3QwenRequest.keyframes(prompt:prompt,grids:[grid],tokenizer:tokenizer)
      XCTAssertEqual(request.tokenIDs.count,2048)
      XCTAssertNoThrow(try H3QwenMRoPE.positions(request:request,grids:[grid]))
      if reference {
        XCTAssertThrowsError(try H3QwenRequest.references(prompt:prompt+"x",
          references:[.image(grid:grid)],tokenizer:tokenizer))
      } else {
        XCTAssertThrowsError(try H3QwenRequest.keyframes(prompt:prompt+"x",grids:[grid],tokenizer:tokenizer))
      }
    }
  }

  func testTextOnlyAndAggregateVisualBoundsRemainInForce() throws {
    let tokenizer = try tokenizer()
    XCTAssertEqual(try H3QwenRequest.text(String(repeating:"x",count:1024),tokenizer:tokenizer).tokenIDs.count,1024)
    XCTAssertThrowsError(try H3QwenRequest.text(String(repeating:"x",count:1025),tokenizer:tokenizer))
    let full = H3QwenRequest.Grid(temporal:1,height:64,width:64)
    let extra = H3QwenRequest.Grid(temporal:1,height:2,width:2)
    XCTAssertThrowsError(try H3QwenRequest.references(prompt:"x",
      references:[.image(grid:full),.image(grid:extra)],tokenizer:tokenizer))
    XCTAssertThrowsError(try H3QwenRequest.keyframes(prompt:"x",grids:[full,extra],tokenizer:tokenizer))
    let oversized = H3QwenRequest(tokenIDs:Array(repeating:1,count:2049),
      tags:Array(repeating:1,count:2049),visualRanges:[])
    XCTAssertThrowsError(try H3QwenMRoPE.positions(request:oversized,grids:[]))
    let invalid = try H3QwenTokenizer(vocabulary:["x":151936],merges:[],regex:".")
    XCTAssertThrowsError(try H3QwenRequest.text("x",tokenizer:invalid))
  }

  func testInstalledFull1024RecipeKeepsVisionAndVAEReferenceGrids() throws {
    guard let file = ProcessInfo.processInfo.environment["WEETODD_H3_FULL_REFERENCE_RECIPE"] else {
      throw XCTSkip("Set the saved 1024-reference image recipe for full preparation admission.")
    }
    try Device.withDefaultDevice(.cpu) {
      let prepared = try H3ImageRecipe.compile(data:Data(contentsOf:URL(fileURLWithPath:file))) { source,digest,geometry in
        let loaded = try H3StillReferenceMedia.load(path:source,outputGeometry:geometry,pixelBudgetPercent:100)
        XCTAssertEqual(loaded.sourceSHA256,digest)
        XCTAssertEqual(loaded.sourceWidth,1024); XCTAssertEqual(loaded.sourceHeight,1024)
        XCTAssertEqual(loaded.reference.width,1024); XCTAssertEqual(loaded.reference.height,1024)
        return loaded.reference
      }
      let request = try XCTUnwrap(prepared.reference)
      let reference = try H3VideoReferencePreparation.prepare(prompt:request.prompt,
        geometry:request.geometry,references:request.references,tokenizerURL:request.tokenizer)
      XCTAssertEqual(reference.qwenGrids.count,1)
      XCTAssertEqual(reference.qwenGrids[0].height,64); XCTAssertEqual(reference.qwenGrids[0].width,64)
      XCTAssertEqual(reference.qwenPixels.shape,[4096,1536])
      XCTAssertEqual(reference.layout.conditionVideoIndices.count,1024)
      XCTAssertGreaterThan(reference.qwenRequest.tokenIDs.count,1024)
      let admission = try H3Ref2VAStillRunner.preflight(request)
      XCTAssertEqual(admission.textRows,reference.qwenRequest.tokenIDs.count)
      XCTAssertEqual(admission.packedRows,reference.layout.tags.count)
      print("Full1024 reference: Qwen rows=\(admission.textRows), visual pads=1024, VAE condition rows=1024, packed rows=\(admission.packedRows)")
    }
  }
}
