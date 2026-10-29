import CryptoKit
import Drama3Contracts
import XCTest

final class Drama3DigestTests: XCTestCase {
  private func fixture() throws -> (URL, Data) {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let data = Data(repeating: 0x5a, count: 3 * 1024 * 1024 + 17)
    try data.write(to: file)
    addTeardownBlock { try? FileManager.default.removeItem(at: file) }
    return (file, data)
  }

  func testMultichunkSpeechDigestMatchesWholePayloadSHA256() throws {
    let (file, data) = try fixture()
    let expected = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    XCTAssertEqual(try Drama3AudioPreparation.fileDigest(file.path), expected)
  }

  func testCancelledSpeechDigestStopsBeforeReadingThePayload() async throws {
    let (file, _) = try fixture()
    let task = Task { () -> Bool in
      withUnsafeCurrentTask { $0?.cancel() }
      do { _ = try Drama3AudioPreparation.fileDigest(file.path); return false }
      catch is CancellationError { return true }
      catch { XCTFail("Unexpected digest error: \(error)"); return false }
    }
    let cancelled = await task.value
    XCTAssertTrue(cancelled, "Native setup checksum reading must observe cancellation.")
  }
}
