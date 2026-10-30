import Foundation
import XCTest
@testable import StudioCore

final class NativeH3RawWorkerBoundaryTests:XCTestCase {
  func testActualSubprocessRequiresCanonicalRawVideoPathAndAcceptsRGBReceipt() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
    defer{try?FileManager.default.removeItem(at:root)}
    let recipe=root.appendingPathComponent("job.json")
    try Data("{\"engine\":\"h3\",\"task\":\"h3_regenerate\"}".utf8).write(to:recipe)
    for legacy in [true,false] {
      let executable=root.appendingPathComponent(legacy ? "legacy-worker":"current-worker")
      let script="""
      #!/usr/bin/swift
      import Foundation
      let envelope=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[3]))) as! [String:Any]
      let directory=URL(fileURLWithPath:CommandLine.arguments[5])
      try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
      let rgb=directory.appendingPathComponent("video.rgb24")
      try Data([0,1,2,3,4,5]).write(to:rgb)
      var result:[String:Any]=["jobID":envelope["jobID"]!,"nativeRuntime":"swift-mlx","outputRGBPath":rgb.path,"pixelFormat":"rgb24"]
      if \(!legacy) {result["video"]=rgb.path}
      let data=try JSONSerialization.data(withJSONObject:["status":"success","result":result])
      FileHandle.standardOutput.write(data+Data([10]))
      """
      try Data(script.utf8).write(to:executable)
      try FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:executable.path)
      let output=root.appendingPathComponent(legacy ? "legacy-output":"current-output")
      if legacy {
        XCTAssertThrowsError(try NativeHeadlessExecutor.worker(executable.path,recipe:recipe,output:output,mode:"render",cancellation:.init(),emit:{_ in})) {
          XCTAssertTrue($0.localizedDescription.contains("no movie"))
        }
      } else {
        let result=try NativeHeadlessExecutor.worker(executable.path,recipe:recipe,output:output,mode:"render",cancellation:.init(),emit:{_ in})
        XCTAssertEqual(result["video"] as? String,output.appendingPathComponent("video.rgb24").path)
        XCTAssertEqual(result["pixelFormat"] as? String,"rgb24")
      }
    }
  }
}
