import CryptoKit
import Foundation
import XCTest

final class H3ImageCLITests: XCTestCase {
  func testPreflightRoutesImageRecipeBeforeVideoReferenceAdmission() throws {
    guard let executable=ProcessInfo.processInfo.environment["WEETODD_H3_IMAGE_WORKER"] else {
      throw XCTSkip("Set the built worker path for CLI dispatch qualification.")
    }
    let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:directory) }
    let recipe:[String:Any]=["format":"weetodd-headless-v2","engine":"h3","prompt":"A robot",
      "components":["task":"ref2va","transformer":"/tmp/checkpoint","text_encoder":"/tmp/qwen",
        "tokenizer":"/tmp/tokenizer","video_vae":"/tmp/vae","audio_vae":"/tmp/audio"],
      "conditioning":["version":1,"task":"ref2va","inputs":[],"audio_policy":"generated"],
      "config":["width":32,"height":32,"steps":5,"seed":42,"duration_seconds":5.0],
      "still_image":["version":1,"mode":"single_latent","frame_selection":"middle"]]
    let bytes=try JSONSerialization.data(withJSONObject:recipe)
    let recipeFile=directory.appendingPathComponent("recipe.json")
    try bytes.write(to:recipeFile)
    let output=directory.appendingPathComponent("output")
    let envelope:[String:Any]=["version":1,"jobID":UUID().uuidString,"engine":"h3",
      "recipePath":recipeFile.path,"recipeSHA256":SHA256.hash(data:bytes).map {String(format:"%02x",$0)}.joined(),
      "outputDirectory":output.path]
    let input=directory.appendingPathComponent("envelope.json")
    try JSONSerialization.data(withJSONObject:envelope).write(to:input)
    let process=Process(),pipe=Pipe()
    process.executableURL=URL(fileURLWithPath:executable)
    process.arguments=["preflight","--request",input.path,"--output",output.path]
    process.standardOutput=pipe;process.standardError=FileHandle.nullDevice
    try process.run()
    let result=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
    XCTAssertNotEqual(process.terminationStatus,0)
    let response=try JSONSerialization.jsonObject(with:result) as! [String:Any]
    XCTAssertTrue((response["error"] as? String ?? "").contains("single-latent"),String(decoding:result,as:UTF8.self))
    XCTAssertFalse(FileManager.default.fileExists(atPath:output.path))
  }
}
