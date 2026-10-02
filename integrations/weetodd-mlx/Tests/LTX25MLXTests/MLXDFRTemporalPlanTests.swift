import XCTest
@testable import LTX25MLX

final class MLXDFRTemporalPlanTests: XCTestCase {
  func testOneRoundOwnsEveryLatentOnceAcrossSeams() throws {
    let tiles = try MLXDFRTemporalPlan.tiles(seams: [48, 96], frames: 97, maximumTiles: 2)
    XCTAssertEqual(tiles.count, 2)
    XCTAssertEqual(tiles[0].pixelStart, 0)
    XCTAssertEqual(tiles[0].pixelEnd, 48)
    XCTAssertEqual(tiles[0].anchorFrames, [48])
    XCTAssertEqual(tiles[0].slotFrames, [24])
    XCTAssertEqual(tiles[1].pixelStart, 24)
    XCTAssertEqual(tiles[1].pixelEnd, 96)
    XCTAssertEqual(tiles[1].anchorFrames, [96])
    XCTAssertEqual(tiles[1].slotFrames, [72])
    XCTAssertEqual(tiles[1].latentStart, 4)
    XCTAssertEqual(tiles[1].dropLatentPrefix, 4)
    XCTAssertEqual(tiles[1].frames, 73)
    let owned = tiles.map { $0.latentFrames - $0.dropLatentPrefix }.reduce(0, +)
    XCTAssertEqual(owned, 13)
  }

  func testTwoRoundsScaleClockButClampConditioning() throws {
    XCTAssertEqual(try MLXDFRTemporalPlan.outputFrames(inputFrames: 49, rounds: 2), 193)
    let tiles=try MLXDFRTemporalPlan.tiles(seams:[48,96,144,192],frames:193,maximumTiles:4)
    XCTAssertEqual(tiles.map(\.slotFrames),[[24],[72],[120],[168]])
    XCTAssertEqual(tiles.map(\.anchorFrames),[[48],[96],[144],[192]])
    XCTAssertEqual(tiles.map { $0.latentFrames-$0.dropLatentPrefix }.reduce(0,+),25)
    XCTAssertEqual(tiles.map(\.pixelStart),[0,24,72,120])
    XCTAssertEqual(try MLXDFRTemporalPlan.conditioningFPS(24), 24)
    XCTAssertEqual(try MLXDFRTemporalPlan.conditioningFPS(48), 60)
    XCTAssertEqual(try MLXDFRTemporalPlan.conditioningFPS(96), 60)
    XCTAssertThrowsError(try MLXDFRTemporalPlan.tiles(seams: [48, 96], frames: 98, maximumTiles: 2))
    XCTAssertThrowsError(try MLXDFRTemporalPlan.outputFrames(inputFrames: 49, rounds: 3))
  }
}
