import Foundation
import XCTest
@testable import StudioCore

final class H3AttentionPolicySelectionTests:XCTestCase {
  func testOldSelectionRemainsDenseWithoutNewSavedKeyAndResetClearsOnlyOverride() throws {
    var selection=GenerationSelection(task:"ref2va")
    let old=try JSONSerialization.jsonObject(with:JSONEncoder().encode(selection)) as! [String:Any]
    XCTAssertNil(old["h3AttentionPolicy"])
    XCTAssertEqual(NativeH3AttentionPolicy.resolved(selection:nil,recipeValue:nil),.dense)
    selection.h3AttentionPolicy = .solExperimental
    XCTAssertTrue(selection.isModified)
    let restored=try JSONDecoder().decode(GenerationSelection.self,from:JSONEncoder().encode(selection))
    XCTAssertEqual(restored.h3AttentionPolicy,.solExperimental)
    XCTAssertEqual(NativeH3AttentionPolicy.resolved(selection:.dense,recipeValue:"sol_experimental"),.dense)
    selection.resetOverrides();XCTAssertNil(selection.h3AttentionPolicy)
    var invalid=old;invalid["h3AttentionPolicy"]="sol"
    XCTAssertThrowsError(try JSONDecoder().decode(GenerationSelection.self,from:JSONSerialization.data(withJSONObject:invalid)))
  }
}
