import Foundation
import XCTest
@testable import StudioCore

final class NativeH3PreparationSetupTests:XCTestCase {
  func testCLIRequiresExactProfilesAndSourceCounts() throws {
    let all=try NativeUpscalerSetupArguments(["prepare-h3","--model-library","/models","--include-fast","true"])
    XCTAssertEqual(all.action,.prepareH3)
    let encoder=try NativeUpscalerSetupArguments(["prepare-h3","--profile","publisher-encoder","--model-library","/models","--source","/two","--source","/three"])
    XCTAssertEqual(encoder.sources,["/two","/three"])
    for arguments in [
      ["prepare-h3","--model-library","/models","--profile","unknown"],
      ["prepare-h3","--model-library","/models","--profile","publisher-encoder","--source","/one"],
      ["prepare-h3","--model-library","/models","--source","/ignored"],
      ["prepare-h3","--model-library","/models","--include-fast","yes"],
      ["prepare-h3","--model-library","/models","--profile","ref2va-base","--catalog","/ignored"]] {
      XCTAssertThrowsError(try NativeUpscalerSetupArguments(arguments))
    }
  }
  private func fixture() throws -> (URL,URL,NativeUpscalerSetup.H3PreparationPlan) {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    let source=root.appendingPathComponent("source.bin");try Data("source".utf8).write(to:source)
    let script=root.appendingPathComponent("convert.sh")
    let outputHash=NativeHeadlessJob.hash(Data("prepared".utf8))
    let body="""
    #!/bin/sh
    printf prepared > "$4"
    printf '{"status":"success","operation":"%s","format":"fixture-v1","path":"%s","sha256":"\(outputHash)"}\\n' "$0" "$4"
    """
    try Data(body.utf8).write(to:script)
    let plan=NativeUpscalerSetup.H3PreparationPlan(profile:"fixture",version:"fixture-v1",command:script.path,
      directory:"WeeTodd/H3/fixture",filename:"prepared.bin",
      sources:[(NativeHeadlessJob.hash(Data("source".utf8)),6,"--source")],outputs:[(outputHash,8,"prepared.bin")])
    return(root,source,plan)
  }
  func testNativeWorkerBridgeVerifiesOutputAndReusesWithoutSourceOrWorker() throws {
    let (root,source,plan)=try fixture();defer{try? FileManager.default.removeItem(at:root)}
    let receipt=try NativeUpscalerSetup.executeH3Preparation(plan,sources:[source],modelLibrary:root,
      worker:URL(fileURLWithPath:"/bin/sh"),existingFiles:[],cancellation:.init(),progress:{_,_ in})
    XCTAssertEqual(receipt.outputSHA256,NativeHeadlessJob.hash(Data("prepared".utf8)))
    XCTAssertTrue(receipt.outputPath.hasPrefix(root.path+"/WeeTodd/H3/"))
    let reused=try NativeUpscalerSetup.executeH3Preparation(plan,sources:[],modelLibrary:root,
      worker:nil,existingFiles:[],cancellation:.init(),progress:{_,_ in})
    XCTAssertEqual(receipt,reused)
    try Data("tampered".utf8).write(to:URL(fileURLWithPath:receipt.outputPath))
    XCTAssertThrowsError(try NativeUpscalerSetup.executeH3Preparation(plan,sources:[source],modelLibrary:root,
      worker:URL(fileURLWithPath:"/bin/sh"),existingFiles:[],cancellation:.init(),progress:{_,_ in}))
    XCTAssertEqual(try Data(contentsOf:URL(fileURLWithPath:receipt.outputPath)),Data("tampered".utf8))
  }
  func testCancellationTerminatesNativeConverterChild() async throws {
    let (root,source,plan)=try fixture();defer{try? FileManager.default.removeItem(at:root)}
    try Data("#!/bin/sh\nprintf partial > \"$4\"\nexec /bin/sleep 30\n".utf8).write(to:URL(fileURLWithPath:plan.command))
    let cancellation=NativeHeadlessCancellation(),start=Date()
    let task=Task.detached {try NativeUpscalerSetup.executeH3Preparation(plan,sources:[source],modelLibrary:root,
      worker:URL(fileURLWithPath:"/bin/sh"),existingFiles:[],cancellation:cancellation,progress:{_,_ in})}
    try await Task.sleep(for:.milliseconds(100));cancellation.cancel()
    do {_ = try await task.value;XCTFail("Cancelled converter completed")}
    catch is CancellationError {}
    XCTAssertLessThan(Date().timeIntervalSince(start),3)
    XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent(plan.directory+"/"+plan.filename).path))
  }
  func testPinnedRawBaseCatalogAndEquivalentPreparedIdentities() throws {
    XCTAssertEqual(NativeUpscalerSetup.h3BasePlan.outputs.count,2)
    XCTAssertEqual(NativeUpscalerSetup.h3BasePlan.sources.first?.sha,"9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9")
    let catalog=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("src/wee_todd_mlx/native_model_download_catalog.json")
    let package=try XCTUnwrap(NativeModelDownloads.catalog(at:catalog).first{$0.descriptor.id=="h3-regenerate-base-source"})
    XCTAssertEqual(package.kind,"h3-ref2va-source")
    XCTAssertEqual(package.files.first?.sha256,NativeUpscalerSetup.h3BasePlan.sources.first?.sha)
    XCTAssertEqual(package.files.first?.size,34_038_894_550)
  }
}
