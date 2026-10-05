import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3VDNInputGridTests: XCTestCase {
  func testSourceGridInterpolatesInFP32WithItsOwnRowCountAndNoSecondSiLU() throws {
    let grid=MLXArray([Float(1),2,5,6,9,10],[3,2]).asType(.bfloat16)
    let actual=try H3VDNInputGrid.interpolate(grid:grid,timesteps:[0,0.25,0.5,0.75,1])
    XCTAssertEqual(actual.dtype,.float32)
    XCTAssertEqual(actual.asArray(Float.self),[1,2,3,4,5,6,7,8,9,10])
    XCTAssertThrowsError(try H3VDNInputGrid.interpolate(grid:grid,timesteps:[.nan]))
    XCTAssertThrowsError(try H3VDNInputGrid.interpolate(grid:grid,timesteps:[1.1]))
  }

  func testInstalledSourceGridProducesBoundedFiniteCoordinatesForExactSamplingTable() throws {
    guard let path=ProcessInfo.processInfo.environment["WEETODD_H3_VDN_GRID"] else {
      throw XCTSkip("Opt-in installed original-width AdaLN grid")
    }
    let source=try H3VDNInputGrid(url:URL(fileURLWithPath:path))
    let geometry=try H3Geometry(width:672,height:384,durationSeconds:5)
    let layout=try H3PackedLayout(geometry:geometry,textTags:[1,1,1],anchors:[])
    let rows=try H3RowSchedule(layout:layout,video:H3Schedule(requestedSteps:9,shift:12),audio:H3Schedule(requestedSteps:9,shift:3))
    let coordinates=try source.evaluate(timesteps:rows.table)
    XCTAssertEqual(coordinates.shape,[rows.table.count,2688])
    XCTAssertTrue(MLX.isFinite(coordinates).all().item(Bool.self))
    XCTAssertLessThan(coordinates.nbytes,2*1024*1024)
  }
  func testInstalledPrunedModulationAddsOriginalCoordinateDeltaBeforeBF16Rounding() throws {
    let env=ProcessInfo.processInfo.environment
    guard let fixtures=env["WEETODD_H3_VDN_ORACLE"],let stage=env["WEETODD_H3_VDN_STAGE"],
      let pages=env["WEETODD_H3_VDN_PAGED"] else { throw XCTSkip("Opt-in actual pruned AdaLN/VDN adapter witness") }
    let values=try loadArrays(url:URL(fileURLWithPath:fixtures).appendingPathComponent("grid-adaln-oracle.safetensors"))
    let adapter=try H3VDNLoRAFile(directory:URL(fileURLWithPath:stage).appendingPathComponent("adapters/turbo"),kind:.turbo)
    let output=try H3AdaLNProjection.evaluate(checkpointURL:URL(fileURLWithPath:pages),blockIndex:49,
      timeEmbeddings:values["curve"]!,lora:adapter,loraInput:values["coordinates"]!)
    XCTAssertEqual(output.asArray(Float.self),values["expected"]!.asArray(Float.self))
    XCTAssertGreaterThan(max(abs(output.asType(.float32)-values["base"]!.asType(.float32))).item(Float.self),0)
  }

}
