import Foundation
import XCTest
@testable import StudioCore

extension NativeModelDownloadsTests {
  func testReusePrefersExactBasenameWithDeterministicFallbackOrdering() {
    let urls=["/z/renamed.safetensors","/z/layer-014.safetensors","/a/layer-014.safetensors","/a/other.safetensors"].map{URL(fileURLWithPath:$0)}
    XCTAssertEqual(NativeModelDownloads.orderedReuseCandidates(urls,target:"pages/layer-014.safetensors").map(\.path),
      ["/a/layer-014.safetensors","/z/layer-014.safetensors","/a/other.safetensors","/z/renamed.safetensors"])
    XCTAssertEqual(NativeModelDownloads.orderedReuseCandidates(Array(urls.reversed()),target:"pages/layer-014.safetensors"),
      NativeModelDownloads.orderedReuseCandidates(urls,target:"pages/layer-014.safetensors"))
  }
  func testNameHintNeverReplacesHashAndRenamedCompatibleFileStillReuses() async throws {
    let root=try directory();defer{try?FileManager.default.removeItem(at:root)}
    let (catalog,_)=try fixture(root)
    try Data(repeating:0,count:bytes.count).write(to:root.appendingPathComponent("model.safetensors"))
    let compatible=root.appendingPathComponent("renamed-compatible.safetensors");try bytes.write(to:compatible)
    let output=try await NativeModelDownloads.prepare(id:"test-model",catalog:catalog,destination:root.appendingPathComponent("library"),
      existingRoots:[root.path],token:nil,progress:{_,_ in},transfer:{_,_,_,_ in XCTFail("Renamed compatible content must reuse after rejecting wrong same-name bytes");throw CancellationError()})
    XCTAssertEqual(try Data(contentsOf:output.appendingPathComponent("pages/model.safetensors")),bytes)
    let original=try FileManager.default.attributesOfItem(atPath:compatible.path),installed=try FileManager.default.attributesOfItem(atPath:output.appendingPathComponent("pages/model.safetensors").path)
    XCTAssertEqual(original[.systemFileNumber] as? NSNumber,installed[.systemFileNumber] as? NSNumber)
  }
}

extension NativeModelDownloadsTests {
  func testH3PackageForwardsExplicitTokenWithoutPersistingIt() async throws {
    let root=try directory();defer{try?FileManager.default.removeItem(at:root)}
    let (catalog,_)=try fixture(root),payload=bytes
    let output=try await NativeUpscalerSetup.ensureH3Package("test-model",family:"H3",catalog:catalog,library:root,
      token:"hf_synthetic_test_token",progress:{_,_ in},transfer:{_,partial,token,_ in
        XCTAssertEqual(token,"hf_synthetic_test_token")
        try payload.write(to:partial)
      })
    XCTAssertEqual(try Data(contentsOf:output.appendingPathComponent("pages/model.safetensors")),payload)
    let provenance=try String(contentsOf:output.appendingPathComponent("setup_provenance.json"),encoding:.utf8)
    XCTAssertFalse(provenance.contains("hf_synthetic_test_token"))
  }
}
