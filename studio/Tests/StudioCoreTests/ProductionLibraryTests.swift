import XCTest
@testable import StudioCore

final class ProductionLibraryTests: XCTestCase {
  func testPublishedObjectCarriesRefModMetadataIntoAnotherMovie() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    var object = PlanningSubject(name: "Character", kind: .character)
    object.details = "Reviewed identity"
    var mod = MediaAsset(name: "Character embedding", kind: .refmod,
      path: "/linked/character.safetensors")
    mod.refmodKind = "image"
    mod.refmodTokenCount = 512
    mod.refmodSHA256 = String(repeating: "b", count: 64)
    object.referenceAssetIDs = [mod.id]
    var plan = ProjectPlanning(); plan.subjects = [object]
    let library = try ProductionLibrary(url: folder.appendingPathComponent("library.sqlite"))
    let package = try library.publish(rootID: object.id, planning: plan, assets: [mod])
    XCTAssertEqual(package.assets.first, mod)
    var movie = StudioProject()
    try movie.importLibraryPackage(package)
    XCTAssertEqual(movie.assets.first?.refmodSHA256, mod.refmodSHA256)
    XCTAssertEqual(movie.planning?.subjects.first?.referenceAssetIDs, [mod.id])
    var conflicting = package
    conflicting.assets[0].refmodSHA256 = String(repeating: "c", count: 64)
    let before = movie
    XCTAssertThrowsError(try movie.importLibraryPackage(conflicting))
    XCTAssertEqual(movie, before)
  }
  func testPublishReopenSearchAndPinnedImportWithoutCopyingMedia() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    let url = folder.appendingPathComponent("library.sqlite")
    var environment = PlanningSubject(name: "Penthouse", kind: .environment)
    environment.details = "Dark concrete"; environment.tags = ["cyberpunk", "luxury"]
    var set = PlanningSubject(name: "Gallery", kind: .set); set.environmentID = environment.id
    var prop = PlanningSubject(name: "Bronze statue", kind: .prop)
    let image = MediaAsset(name: "Statue", kind: .image, path: "/unavailable/reference.jpg")
    prop.referenceAssetIDs = [image.id]
    set.relationships = [ObjectRelationship(targetID: prop.id, role: .contains)]
    var plan = ProjectPlanning(); plan.subjects = [environment, set, prop]
    let library = try ProductionLibrary(url: url)
    let first = try library.publish(rootID: environment.id, planning: plan, assets: [image])
    XCTAssertEqual(first.version, 1)
    XCTAssertEqual(try library.publish(rootID: environment.id, planning: plan, assets: [image]).version, 1)
    let reopened = try ProductionLibrary(url: url)
    XCTAssertEqual(try reopened.search("cyberpunk").count, 1)
    XCTAssertEqual(try reopened.search("' OR 1=1 --").count, 0)
    var movie = StudioProject()
    try movie.importLibraryPackage(first)
    XCTAssertEqual(movie.planning?.subjects.count, 3)
    XCTAssertEqual(movie.assets.first?.path, image.path)
    plan.subjects[2].details = "New definition"
    let second = try library.publish(rootID: environment.id, planning: plan, assets: [image])
    XCTAssertEqual(second.version, 2)
    XCTAssertEqual(try reopened.package(rootID: environment.id, version: 1), first)
    XCTAssertEqual(movie.planning?.subjects.first(where: { $0.id == prop.id })?.details, "")
    // Publishing a single set must not pull its sibling into the package.
    var sibling = PlanningSubject(name: "Pool", kind: .set); sibling.environmentID = environment.id
    plan.subjects.append(sibling)
    let setPackage = try library.publish(rootID: set.id, planning: plan, assets: [image])
    XCTAssertFalse(setPackage.subjects.contains { $0.id == sibling.id })
    let before = movie
    XCTAssertThrowsError(try movie.importLibraryPackage(second))
    XCTAssertEqual(movie, before)
    XCTAssertEqual(first.missingMediaPaths, [image.path])
  }
  func testInvalidGraphCannotPublishPartialPackage() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: folder) }
    let library = try ProductionLibrary(url: folder.appendingPathComponent("library.sqlite"))
    var set = PlanningSubject(name: "Broken", kind: .set); set.environmentID = UUID()
    var plan = ProjectPlanning(); plan.subjects = [set]
    XCTAssertThrowsError(try library.publish(rootID: set.id, planning: plan, assets: []))
    XCTAssertTrue(try library.search("").isEmpty)
  }
}
