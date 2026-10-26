import Foundation
import XCTest
@testable import H3MLX

final class H3ImageIdentityTests: XCTestCase {
  func testPreflightBindingChangesWithComponentAndIgnoresOnlyItsExpectedReceipt() throws {
    let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:folder) }
    let file=folder.appendingPathComponent("model.bin")
    try Data([1]).write(to:file)
    var recipe:[String:Any]=["prompt":"A robot", "still_image":["request_id":UUID().uuidString]]
    let first=try H3ImageIdentity.fingerprint(recipe:recipe,components:[folder])
    recipe["still_image"]=(recipe["still_image"] as! [String:Any]).merging(["expected_fingerprint":first]) {_,new in new}
    XCTAssertEqual(try H3ImageIdentity.fingerprint(recipe:recipe,components:[folder]),first)
    try Data([1,2]).write(to:file)
    XCTAssertNotEqual(try H3ImageIdentity.fingerprint(recipe:recipe,components:[folder]),first)
    recipe["prompt"]="A fox"
    XCTAssertNotEqual(try H3ImageIdentity.fingerprint(recipe:recipe,components:[folder]),first)
  }
}
