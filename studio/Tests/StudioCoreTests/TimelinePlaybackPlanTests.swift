import StudioCore
import XCTest

final class TimelinePlaybackPlanTests: XCTestCase {
  func testOverlappingClipsKeepTimelineTimingAndIncomingShotOwnsCut() {
    var project = StudioProject()
    var a = Clip(name: "A"); a.duration = 5; a.sourceIn = 11
    var b = Clip(name: "B"); b.duration = 4; b.sourceIn = 20
    b.transition = "dissolve"; b.transitionDuration = 1
    project.clips = [a, b]
    let plan = TimelinePlaybackPlan(project: project)
    XCTAssertEqual(plan.duration, 8)
    XCTAssertEqual(plan.spans[0].duration, 4)
    XCTAssertEqual(plan.spans[1].start, 4)
    XCTAssertEqual(plan.span(at: 3.999)?.clipID, a.id)
    XCTAssertEqual(plan.span(at: 4)?.clipID, b.id)
    XCTAssertEqual(plan.span(at: 4)?.sourceIn, 20)
    XCTAssertEqual(plan.span(at: 8)?.clipID, b.id)
  }

  func testRulerIgnoresBlankAreaAndHandleClampsWithoutAcceptingNonfiniteTime() {
    var project = StudioProject(); project.clips = [Clip()]
    let plan = TimelinePlaybackPlan(project: project)
    XCTAssertNil(plan.seekPosition(-0.1, clamped: false))
    XCTAssertNil(plan.seekPosition(5.01, clamped: false))
    XCTAssertEqual(plan.seekPosition(2.75, clamped: false), 2.75)
    XCTAssertEqual(plan.seekPosition(-1, clamped: true), 0)
    XCTAssertEqual(plan.seekPosition(8, clamped: true), 5)
    for invalid in [Double.nan, Double.infinity, -Double.infinity] {
      XCTAssertNil(plan.seekPosition(invalid, clamped: true))
    }
    XCTAssertNil(TimelinePlaybackPlan(project: StudioProject()).seekPosition(0, clamped: true))
  }

  func testUnrenderedAndStillClipsKeepTheirFullTimeAndPreviewSizeIsBounded() {
    var project = StudioProject()
    var a = Clip(); a.duration = 2
    var b = Clip(); b.sourcePath = "/tmp/still.PNG"; b.duration = 3
    project.clips = [a, b]; project.settings.width = 3840; project.settings.height = 2160
    let plan = TimelinePlaybackPlan(project: project)
    XCTAssertEqual(plan.duration, 5)
    XCTAssertEqual(plan.span(at: 2.1)?.clipID, b.id)
    XCTAssertTrue(plan.spans[1].isStill)
    XCTAssertEqual(plan.width, 1280); XCTAssertEqual(plan.height, 720)
  }
}
