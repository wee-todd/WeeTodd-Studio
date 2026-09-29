import XCTest
@testable import H3MLX

final class H3RowScheduleTests: XCTestCase {
  func testOneGlobalTableAndPerRowIndicesForFirstFrameConditioning() throws {
    let geometry = try H3Geometry(width: 64, height: 32, durationSeconds: 2.5)
    let layout = try H3PackedLayout(geometry: geometry, textTags: [1, 1], anchors: [.first])
    let plan = try H3RowSchedule(layout: layout,
      video: H3Schedule(requestedSteps: 5, shift: 12),
      audio: H3Schedule(requestedSteps: 5, shift: 3))
    XCTAssertEqual(plan.table,
      [0, 0.0270270109, 0.0769230723, 0.1000000238,
        0.1999999881, 0.25, 0.5, 0.999] as [Float])
    XCTAssertEqual(plan.indicesByStep.count, 4)
    XCTAssertEqual(plan.indicesByStep[0].count, layout.tags.count)
    XCTAssertEqual(plan.indicesByStep[0][0], 0) // text inherits video time
    XCTAssertEqual(plan.indicesByStep[0][2], 7) // conditioned video remains near x0
    XCTAssertEqual(plan.indicesByStep[0][layout.audioStart], 0)
    XCTAssertEqual(plan.indicesByStep[1][0], 1)
    XCTAssertEqual(plan.indicesByStep[1][2], 7)
    XCTAssertEqual(plan.indicesByStep[1][layout.audioStart], 3)
    XCTAssertEqual(plan.indicesByStep[3][layout.videoStart], 4)
    XCTAssertEqual(plan.indicesByStep[3][layout.audioStart], 6)
  }
}
