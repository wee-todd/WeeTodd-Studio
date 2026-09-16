import XCTest
import StudioCore
@testable import WeeToddStudio

final class DrawThingsCatalogDiscoveryTests: XCTestCase {
  let catalog: [String: Any] = ["models": [["id": "klein", "name": "Klein"]],
    "capabilities": ["klein": ["operations": ["image": [:]]]]]

  @MainActor func testLoadOnceAndKeepCatalogAfterFailedRefresh() async {
    let discovery = DrawThingsCatalogDiscovery()
    let connection = DrawThingsConnection(id: "local")
    var calls = 0
    await discovery.load(connection) { calls += 1; return self.catalog }
    await discovery.load(connection) { calls += 1; return [:] }
    XCTAssertEqual(calls, 1)
    await discovery.load(connection, force: true) { throw StudioError.invalid("Server unavailable") }
    XCTAssertNotNil(discovery.catalogs[connection.id])
    XCTAssertTrue(discovery.errors[connection.id]?.contains("Server unavailable") == true)
    XCTAssertFalse(discovery.loading.contains(connection.id))
    await discovery.load(connection, force: true) { self.catalog }
    XCTAssertNil(discovery.errors[connection.id])
  }

  @MainActor func testConcurrentRequestsJoin() async {
    let discovery = DrawThingsCatalogDiscovery()
    let connection = DrawThingsConnection(id: "local")
    let started = expectation(description: "Discovery started")
    var finish: CheckedContinuation<[String: Any], Error>?
    let first = Task { await discovery.load(connection) {
      try await withCheckedThrowingContinuation { finish = $0; started.fulfill() }
    } }
    await fulfillment(of: [started], timeout: 2)
    let joined = Task { await discovery.load(connection) { XCTFail("Duplicated discovery"); return [:] } }
    await Task.yield()
    finish?.resume(returning: catalog)
    await first.value; await joined.value
    XCTAssertNotNil(discovery.catalogs[connection.id])
  }

  @MainActor func testInvalidatedReplyCannotOverwriteNewConnection() async {
    let discovery = DrawThingsCatalogDiscovery()
    var connection = DrawThingsConnection(id: "same-id")
    let old = connection
    let started = expectation(description: "First request started")
    var finish: CheckedContinuation<[String: Any], Error>?
    let first = Task { await discovery.load(old) {
      try await withCheckedThrowingContinuation { finish = $0; started.fulfill() }
    } }
    await fulfillment(of: [started], timeout: 2)
    discovery.invalidate(old.id)
    connection.port = 7860
    await discovery.load(connection) { ["models": [], "capabilities": [:], "marker": "new"] }
    finish?.resume(returning: catalog)
    await first.value
    XCTAssertEqual(discovery.catalogs[connection.id]?["marker"] as? String, "new")
    XCTAssertFalse(discovery.loading.contains(connection.id))
  }

  @MainActor func testFirstFailureEmptyCatalogAndMalformedResponseAreDistinct() async {
    let discovery = DrawThingsCatalogDiscovery()
    let connection = DrawThingsConnection(id: "local")
    await discovery.load(connection) { throw StudioError.invalid("Offline") }
    XCTAssertNil(discovery.catalogs[connection.id])
    XCTAssertNotNil(discovery.errors[connection.id])
    await discovery.load(connection, force: true) { ["models": [], "capabilities": [:]] }
    XCTAssertNotNil(discovery.catalogs[connection.id])
    XCTAssertNil(discovery.errors[connection.id])
    await discovery.load(connection, force: true) { [:] }
    XCTAssertNotNil(discovery.errors[connection.id])
    XCTAssertNotNil(discovery.catalogs[connection.id])
  }
}
