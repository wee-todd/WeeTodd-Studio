import XCTest
@testable import StudioCore

final class DrawThingsHiresImportTests: XCTestCase {
  func testPlainI2VHighResRecipePreservesExactSettings() throws {
    let data = Data(#"{"model":"ltx_2.3_22b_dev_q8p.ckpt","width":2048,"height":1152,"sampler":17,"steps":8,"guidanceScale":1,"shift":6.5,"loras":[{"file":"ltx_2.3_22b_distilled_lora_f16.ckpt","weight":0.9}],"hiresFix":true,"hiresFixWidth":1024,"hiresFixHeight":576,"hiresFixStrength":0.5}"#.utf8)
    let value = try DrawThingsConfigImport.parse(data, operation: "video")[0]
    XCTAssertTrue(value.warnings.isEmpty)
    XCTAssertEqual(value.configuration["hiresFix"], .boolean(true))
    XCTAssertEqual(value.configuration["hiresFixWidth"], .integer(1024))
    XCTAssertEqual(value.configuration["hiresFixHeight"], .integer(576))
    XCTAssertEqual(value.configuration["hiresFixStrength"], .number(0.5))
    XCTAssertEqual(value.configuration["sampler"], .integer(17))
    XCTAssertEqual(value.loras?.first?.weight, 0.9)
  }

  func testHighResValuesAreTypedAndDimensionsUsePixelGrid() throws {
    for json in [#"{"hiresFix":1}"#, #"{"hiresFix":"true"}"#, #"{"hiresFixWidth":1025}"#,
                 #"{"hiresFixHeight":true}"#, #"{"hiresFixStrength":1.1}"#] {
      XCTAssertThrowsError(try DrawThingsConfigImport.parse(Data(json.utf8), operation: "video"))
    }
    let value = try DrawThingsConfigImport.parse(Data(#"{"hiresFix":false}"#.utf8), operation: "video")[0]
    XCTAssertEqual(value.configuration["hiresFix"], .boolean(false))
  }

  func testImageImportAndStageTwoOmissionsAreVisible() throws {
    let value = try DrawThingsConfigImport.parse(Data(#"{"steps":8,"hiresFix":true,"hiresFixWidth":1024,"stage2Steps":10,"stage2Cfg":1,"stage2Shift":1}"#.utf8), operation: "image")[0]
    XCTAssertNil(value.configuration["hiresFix"])
    XCTAssertNil(value.configuration["hiresFixWidth"])
    XCTAssertEqual(value.warnings.count, 5)
    XCTAssertTrue(value.warnings.contains { $0.contains("Wurstchen") && $0.contains("LTX") })
  }
}
