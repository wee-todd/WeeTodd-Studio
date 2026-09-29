import Foundation
import XCTest
import MLX
import TensorIO
import LTX25Engine
@testable import LTX25MLX

final class MLXFixedTimingTests:XCTestCase {
  func testInstalledCompletedFixedReads() throws {
    guard let request=ProcessInfo.processInfo.environment["WEETODD_MLX_FIXED_TIMING_REQUEST"] else { throw XCTSkip("Installed fixed-read timing is opt-in.") }
    let json=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:request))) as! [String:Any]
    let root=URL(fileURLWithPath:json["transformer_root"] as! String)
    let manifest=try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("paged_manifest.json"))) as! [String:Any]
    let url=root.appendingPathComponent((manifest["fixed"] as! [String:Any])["file"] as! String)
    let c=try AVBlockConfiguration(videoTokens:3276,audioTokens:93,textTokens:1024)
    let shapes=DenoiserLayout.weightShapes(c),file=try SafeTensorFile(url:url)
    let mapped=try MLXFixedSource(url:url,configuration:c,nativeLoading:false)
    let native=MLXNativeWeightSource(file:file,url:url)
    var names:[String:String]=[:]
    for original in file.tensors.keys {
      if let name=LTXAdapterCompatibility.normalize(original),shapes[name] != nil { names[name]=original }
    }
    let old=Memory.cacheLimit;Memory.cacheLimit=128*1024*1024
    defer { native.clear();Memory.clearCache();Memory.cacheLimit=old }
    var expected:[Float]?
    for run in 0..<3 {
      for useNative in [false,true] {
        native.clear();Memory.clearCache();Memory.peakMemory=0
        var seconds=0.0,signature:[Float]=[],bytes=0
        for name in shapes.keys.sorted() {
          try autoreleasepool {
            let start=Date()
            let weight=try useNative ? native.read(names[name]!,shapes[name]!) : mapped.read(name,shape:shapes[name]!)
            try MLXWeight.materialize([weight])
            seconds += Date().timeIntervalSince(start);bytes += weight.storageBytes
            let flat=try weight.tensor().reshaped([-1])
            signature += flat[0..<min(16,flat.size)].asType(.float32).asArray(Float.self)
          }
        }
        if let expected { XCTAssertEqual(signature,expected) } else { expected=signature }
        print("FIXED_READ_TIMING run=\(run) native=\(useNative) seconds=\(seconds) bytes=\(bytes) mlx_peak=\(Memory.peakMemory)")
      }
    }
  }
}
