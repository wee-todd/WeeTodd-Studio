import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3SharpCurveTests: XCTestCase {
  func testEndpointsAndReleaseTimesInterpolateIndependentEightDimensionalBasis() throws {
    let values = (0..<1025).flatMap { row in (0..<8).map { Float(row) + Float($0)*10000 } }
    let times: [Float] = [0,1,0.090625,1.0/7,1.0/3,0.3,1-0.49411764705882355]
    let result = try H3SharpCurve.coordinates(table:MLXArray(values,[1025,8]),modelTimes:times).asArray(Float.self)
    for (i,t) in times.enumerated() {
      XCTAssertEqual(result[i*8],t*1024,accuracy:0.0001)
      XCTAssertEqual(result[i*8+7],70000+t*1024,accuracy:0.01)
    }
    XCTAssertThrowsError(try H3SharpCurve.coordinates(table:.zeros([1001,64]),modelTimes:[0.5]))
    let invalid: [[Float]] = [[-.infinity],[.nan],[-0.01],[1.01],[]]
    for times in invalid {
      XCTAssertThrowsError(try H3SharpCurve.coordinates(table:.zeros([1025,8]),modelTimes:times))
    }
  }
  func testAllFiftySharpBranchesRequiredAndNoExtraTargets() throws {
    var tensors: [String:H3RegenerateTensor] = [:]
    for block in 0..<50 {
      let prefix="transformer.transformer_blocks.\(block).adaln_proj.linear"
      tensors[prefix+".curve_A.weight"] = .init(dtype:"F32",shape:[64,8])
      tensors[prefix+".curve_B.weight"] = .init(dtype:"F32",shape:[96768,64])
    }
    let metadata=["format":"lms-sharp-curve-adaln","strength":"1.0 folded"]
    XCTAssertNoThrow(try H3SharpCurve.validate(metadata:metadata,tensors:tensors))
    let name="transformer.transformer_blocks.49.adaln_proj.linear.curve_A.weight"
    tensors[name] = .init(dtype:"F32",shape:[64,64])
    XCTAssertThrowsError(try H3SharpCurve.validate(metadata:metadata,tensors:tensors))
    tensors[name] = nil
    XCTAssertThrowsError(try H3SharpCurve.validate(metadata:metadata,tensors:tensors))
  }
  func testModulationDeltaRetainsTimeAndModalityRowsBeforeIndexSelection() throws {
    let coords=MLXArray([Float(1),2, 3,4],[2,2])
    let a=MLXArray([Float(2),1],[1,2])
    let b=MLXArray((1...18).map(Float.init),[18,1])
    let delta=try H3SharpCurve.project(coordinates:coords,a:a,b:b)
    let base=MLXArray.zeros([2,18],dtype:.bfloat16)
    let added=try H3SharpCurve.add(delta:delta,to:base)
    let rows=added.reshaped([6,6])
    XCTAssertEqual(rows[0,0].item(Float.self),4)
    XCTAssertEqual(rows[2,0].item(Float.self),52)
    XCTAssertEqual(rows[3,0].item(Float.self),10)
    XCTAssertEqual(rows[5,0].item(Float.self),130)
    XCTAssertThrowsError(try H3SharpCurve.add(delta:delta,to:.zeros([1,18])))
  }
  func testWrongCurveIdentityRejectsEvenWhenByteCountMatches() throws {
    let url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:url) }
    try Data(repeating:0,count:1025*8*4).write(to:url)
    XCTAssertThrowsError(try H3SharpCurve.readTable(url:url))
  }
}
