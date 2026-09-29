import XCTest
import Foundation
import MLX
import TensorIO
@testable import LTX25MLX

final class MLXTextNativeWeightsTests:XCTestCase {
  func fixtureURL() throws -> URL {
    try XCTUnwrap(Bundle.module.url(forResource:"text-connector",withExtension:"safetensors",subdirectory:"Fixtures"))
  }
  func testNativeConnectorWeightsMatchIndependentFixtureOnRepeatedUse() throws {
    let url=try fixtureURL(),file=try SafeTensorFile(url:url)
    let source=MLXNativeWeightSource(file:file,url:url)
    for _ in 0..<2 {
      source.clear()
      let actual=try MLXTextConnector.evaluate(MLXWeight.read(file,"input"),width:8,heads:2,weights:source.read)
      let expected=try file.readFloat32(named:"expected")
      XCTAssertLessThan(zip(actual.asArray(Float.self),expected).map { abs($0-$1) }.max()!,0.0001)
    }
  }
  func testAtomicReplacementBeforeMaterializationRejectsDeferredTextWeights() throws {
    let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:directory) }
    let url=directory.appendingPathComponent("text.safetensors")
    try FileManager.default.copyItem(at:fixtureURL(),to:url)
    let file=try SafeTensorFile(url:url),source=MLXNativeWeightSource(file:file,url:url)
    let weight=try source.read("attn1.q_norm.weight",[8])
    let replacement=directory.appendingPathComponent("replacement.safetensors")
    try FileManager.default.copyItem(at:fixtureURL(),to:replacement)
    try FileManager.default.removeItem(at:url)
    try FileManager.default.moveItem(at:replacement,to:url)
    XCTAssertThrowsError(try MLXWeight.materialize([weight]))
    XCTAssertThrowsError(try source.read("attn1.k_norm.weight",[8]))
  }
}
