import XCTest
@testable import StudioCore

final class SessionCredentialStoreTests: XCTestCase {
  enum Failure: Error { case denied }

  func testRepeatedAndConcurrentReadsShareOneLookup() throws {
    var lookups = 0
    let store = SessionCredentialStore(read: { _ in
      lookups += 1
      return "fixture"
    }, save: { _, _ in }, remove: { _ in })
    DispatchQueue.concurrentPerform(iterations: 20) { _ in
      XCTAssertEqual(try? store.read("one"), "fixture")
    }
    XCTAssertEqual(lookups, 1)
    XCTAssertEqual(try store.read("two"), "fixture")
    XCTAssertEqual(lookups, 2)
  }

  func testSaveAndRemoveInvalidateCachedAccessEvenWhenTheyFail() throws {
    var stored: String? = "old"
    var fail = false
    let store = SessionCredentialStore(read: { _ in stored }, save: { value, _ in
      if fail { throw Failure.denied }
      stored = value
    }, remove: { _ in
      if fail { throw Failure.denied }
      stored = nil
    })
    XCTAssertEqual(try store.read("one"), "old")
    try store.save("new", reference: "one")
    XCTAssertEqual(try store.read("one"), "new")
    fail = true
    XCTAssertThrowsError(try store.save("denied", reference: "one"))
    stored = "external change"
    XCTAssertEqual(try store.read("one"), "external change")
    XCTAssertThrowsError(try store.remove("one"))
    stored = "external change again"
    XCTAssertEqual(try store.read("one"), "external change again")
    fail = false
    try store.remove("one")
    XCTAssertNil(try store.read("one"))
  }

  func testMissingAndDeniedLookupsAreNotRememberedAndClearStartsFresh() throws {
    var attempts = 0
    let store = SessionCredentialStore(read: { _ in
      attempts += 1
      if attempts == 1 { throw Failure.denied }
      if attempts == 2 { return nil }
      return "fixture"
    }, save: { _, _ in }, remove: { _ in })
    XCTAssertThrowsError(try store.read("one"))
    XCTAssertNil(try store.read("one"))
    XCTAssertEqual(try store.read("one"), "fixture")
    XCTAssertEqual(try store.read("one"), "fixture")
    XCTAssertEqual(attempts, 3)
    store.clear()
    XCTAssertEqual(try store.read("one"), "fixture")
    XCTAssertEqual(attempts, 4)
  }
}
