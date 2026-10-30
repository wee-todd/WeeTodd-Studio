import Foundation
import XCTest
@testable import InferenceContracts
final class NativeFlashVSRAssetsTests:XCTestCase {
  func testNativeRootAndLegacyLocalPoliciesRemainDistinct() throws {
    let root=try fixture();defer { try? FileManager.default.removeItem(at:root) }
    for name in ["posi_prompt.pth","TCDecoder.ckpt"] { try write(root.appendingPathComponent(name));try write(root.appendingPathComponent("turbo/"+name)) }
    let native=try NativeFlashVSRAssets.resolve(root:root,variant:.turbo,decoder:.tiny)
    let legacy=try NativeFlashVSRAssets.resolve(root:root,variant:.turbo,decoder:.tiny,policy:.legacyVpipe)
    XCTAssertEqual(native.context,root.appendingPathComponent("posi_prompt.pth").resolvingSymlinksInPath())
    XCTAssertEqual(legacy.context,root.appendingPathComponent("turbo/posi_prompt.pth").resolvingSymlinksInPath())
    // A root payload still wins even if its later hash check will fail. Never
    // retry a different file to hide a selected component checksum mismatch.
    try Data("corrupt selected payload".utf8).write(to:native.decoder)
    XCTAssertEqual(try NativeFlashVSRAssets.resolve(root:root,variant:.turbo,decoder:.tiny).decoder,native.decoder)
  }
  func testNativeCanReuseLocalOnlySharedAssetsWithoutDecoderSubstitution() throws {
    let root=try fixture();defer { try? FileManager.default.removeItem(at:root) }
    for name in ["posi_prompt.pth","TCDecoder.ckpt"] { try write(root.appendingPathComponent("turbo/"+name)) }
    let result=try NativeFlashVSRAssets.resolve(root:root,variant:.turbo,decoder:.tiny)
    XCTAssertEqual(result.context,root.appendingPathComponent("turbo/posi_prompt.pth").resolvingSymlinksInPath())
    XCTAssertThrowsError(try NativeFlashVSRAssets.resolve(root:root,variant:.turbo,decoder:.wan))
    try FileManager.default.createDirectory(at:root.appendingPathComponent("TCDecoder.ckpt"),withIntermediateDirectories:false)
    XCTAssertThrowsError(try NativeFlashVSRAssets.resolve(root:root,variant:.turbo,decoder:.tiny))
  }
  func testDanglingPreferredLinkCannotSelectAnUnrelatedFallback() throws {
    let root=try fixture();defer { try? FileManager.default.removeItem(at:root) }
    try write(root.appendingPathComponent("posi_prompt.pth"));try write(root.appendingPathComponent("turbo/TCDecoder.ckpt"))
    try FileManager.default.createSymbolicLink(at:root.appendingPathComponent("TCDecoder.ckpt"),withDestinationURL:root.appendingPathComponent("missing"))
    XCTAssertThrowsError(try NativeFlashVSRAssets.resolve(root:root,variant:.turbo,decoder:.tiny))
  }
  private func fixture()throws->URL {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try write(root.appendingPathComponent("turbo/dit.safetensors"));try write(root.appendingPathComponent("turbo/lq_proj_in.safetensors"));return root
  }
  private func write(_ url:URL)throws {
    try FileManager.default.createDirectory(at:url.deletingLastPathComponent(),withIntermediateDirectories:true)
    try Data("fixture".utf8).write(to:url)
  }
}
