import Foundation
import XCTest
@testable import StudioCore
final class NativeH3ReferenceOptionsTests:XCTestCase {
  func testResettingPlacementRetainsBudgetAndMovieDensityOverrides() {
    XCTAssertTrue(H3ReferencePlacement().isEmpty)
    var image=H3ReferencePlacement(frame:.last,imagePixelBudgetPercent:200)
    image.frame=nil;XCTAssertFalse(image.isEmpty)
    var movie=H3ReferencePlacement(frame:.last,videoTemporalDensity:.quarter)
    movie.frame=nil;XCTAssertFalse(movie.isEmpty)
    XCTAssertFalse(H3ReferencePlacement(videoSizePolicy:.nativeH3).isEmpty)
    XCTAssertFalse(H3ReferencePlacement(soundtrackPath:"/audio.wav").isEmpty)
  }
  func testLegacyPlacementOmitsNewFieldsAndExplicitMovieDefaultsAreFrozen() throws {
    let old=H3ReferencePlacement()
    XCTAssertTrue(try old.mediaOptions(kind:"video",task:"ref2va").isEmpty)
    let encoded=try JSONSerialization.jsonObject(with:JSONEncoder().encode(old)) as! [String:Any]
    XCTAssertNil(encoded["imagePixelBudgetPercent"]);XCTAssertNil(encoded["videoSizePolicy"]);XCTAssertNil(encoded["videoTemporalDensity"])
    let fields=try H3ReferencePlacement(videoTemporalDensity:.quarter).mediaOptions(kind:"video",task:"ref2va")
    XCTAssertEqual(fields["temporal_density"] as? String,"quarter");XCTAssertEqual(fields["size_policy"] as? String,"match_output")
    let native=try H3ReferencePlacement(videoSizePolicy:.nativeH3).mediaOptions(kind:"video",task:"ref2va")
    XCTAssertEqual(native["temporal_density"] as? String,"full");XCTAssertEqual(native["size_policy"] as? String,"native_h3")
  }
  func testImageBudgetAcceptsBoundedRefAndA2VAndRejectsIgnoredKindsAndTasks() throws {
    for task in ["ref2va","a2v"] {
      let fields=try H3ReferencePlacement(imagePixelBudgetPercent:400).mediaOptions(kind:"image",task:task)
      XCTAssertEqual(fields["image_pixel_budget_percent"] as? Int,400)
    }
    for task in ["fflf","i2v","extension","motion_fidelity"] {
      XCTAssertThrowsError(try H3ReferencePlacement(imagePixelBudgetPercent:100).mediaOptions(kind:"image",task:task))
    }
    XCTAssertThrowsError(try H3ReferencePlacement(imagePixelBudgetPercent:49).mediaOptions(kind:"image",task:"ref2va"))
    XCTAssertThrowsError(try H3ReferencePlacement(imagePixelBudgetPercent:100).mediaOptions(kind:"audio",task:"ref2va"))
    XCTAssertThrowsError(try H3ReferencePlacement(videoTemporalDensity:.half).mediaOptions(kind:"image",task:"ref2va"))
    XCTAssertThrowsError(try H3ReferencePlacement(frame:.index(5),imagePixelBudgetPercent:100).mediaOptions(kind:"image",task:"a2v"))
  }
}
