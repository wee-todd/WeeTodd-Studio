import XCTest
import MLX
import LTX25MLX
import TensorIO
import InferenceTestSupport

final class MLXWeightTests: XCTestCase {
  func testBufferedWeightReadsPackedCompanionsAndRejectsChangedCheckpoint() throws {
    let words=[UInt32](repeating:0x03020100,count:32)
    let scales:[Float]=[0.25,0.5], offsets:[Float]=[-0.5,0.25]
    try withTensorFile(tensors:[("p.weight",[2,16],"U32"),("p.scales",[2,1],"F32"),("p.biases",[2,1],"F32")],payloads:[
      "p.weight":words.withUnsafeBytes { Data($0) },"p.scales":scales.withUnsafeBytes { Data($0) },"p.biases":offsets.withUnsafeBytes { Data($0) }]) { url in
      let file=try SafeTensorFile(url:url)
      let weight=try MLXWeight(file:file,name:"p.weight",shape:[2,64],access:.buffered)
      XCTAssertEqual(weight.storageBytes,144)
      XCTAssertEqual(try weight.projected(.ones([1,64])).asArray(Float.self),[-8,64])
      let handle=try FileHandle(forWritingTo:url)
      try handle.seekToEnd();try handle.write(contentsOf:Data([0]));try handle.close()
      XCTAssertThrowsError(try MLXWeight(file:file,name:"p.weight",shape:[2,64],access:.buffered))
    }
    let values:[UInt16]=[0x3f80,0x4000,0x4040,0x4080]
    try withTensorFile(tensors:[("w",[2,2],"BF16")],payloads:["w":values.withUnsafeBytes { Data($0) }]) { url in
      let weight=try MLXWeight(file:SafeTensorFile(url:url),name:"w",shape:[2,2],access:.buffered)
      XCTAssertEqual(weight.storageBytes,8)
      XCTAssertEqual(try weight.projected(MLXArray([Float(2),3],[1,2])).asArray(Float.self),[8,18])
    }
  }
  func testPackedRowSlicesRetainOnlySelectedRowsAndMatchDense() throws {
    let words=[UInt32](repeating:0x03020100,count:64)
    let scales:[Float]=[0.25,0.5,1,2], offsets:[Float]=[-0.5,0.25,0,1]
    try withTensorFile(tensors:[("p.weight",[4,16],"U32"),("p.scales",[4,1],"F32"),("p.biases",[4,1],"F32")],payloads:[
      "p.weight":words.withUnsafeBytes { Data($0) },"p.scales":scales.withUnsafeBytes { Data($0) },"p.biases":offsets.withUnsafeBytes { Data($0) }]) { url in
      let file=try SafeTensorFile(url:url)
      let slice=try MLXWeight(file:file,name:"p.weight",shape:[4,64],rows:1..<3)
      XCTAssertEqual(slice.shape,[2,64]); XCTAssertEqual(slice.storageBytes,144)
      XCTAssertEqual(try slice.projected(.ones([1,64])).asArray(Float.self),[64,96])
      XCTAssertThrowsError(try MLXWeight(file:file,name:"p.weight",shape:[4,64],rows:0..<4,maximumBytes:4))
      XCTAssertThrowsError(try MLXWeight(file:file,name:"p.weight",shape:[4,64],rows:4..<5))
    }
    let data:[UInt16]=[0x3f80,0x4000,0x4040,0x4080,0x40a0,0x40c0]
    try withTensorFile(tensors:[("embedding",[3,2],"BF16")],payloads:["embedding":data.withUnsafeBytes { Data($0) }]) { url in
      let row=try MLXWeight(file:SafeTensorFile(url:url),name:"embedding",shape:[3,2],rows:2..<3)
      XCTAssertEqual(row.storageBytes,4)
      XCTAssertEqual(try row.tensor().asType(.float32).asArray(Float.self),[5,6])
    }
  }
  func testPackedQ8ProjectionAndMultipleLowRankUpdates() throws {
    let words = [UInt32](repeating: 0x03020100, count: 32)
    let scales: [Float] = [0.25, 0.5], offsets: [Float] = [-0.5, 0.25]
    try withTensorFile(tensors: [("p.weight", [2,16], "U32"),
      ("p.scales", [2,1], "F32"), ("p.biases", [2,1], "F32")], payloads: [
        "p.weight": words.withUnsafeBytes { Data($0) },
        "p.scales": scales.withUnsafeBytes { Data($0) },
        "p.biases": offsets.withUnsafeBytes { Data($0) }]) { url in
      let weight = try MLXWeight(file: SafeTensorFile(url: url), name: "p.weight", shape: [2,64])
      XCTAssertEqual(weight.storageBytes, 144, "Packed weights must not become a dense Float32 matrix")
      let x = MLXArray.ones([3,64])
      let base = try weight.projected(x).asArray(Float.self)
      XCTAssertEqual(base, [-8, 64, -8, 64, -8, 64])
      let a = try MLXLoRA(down: .ones([1,64]), up: MLXArray([Float(1),2],[2,1]), strength: 0.5)
      let b = try MLXLoRA(down: .ones([2,64]), up: .ones([2,2]), strength: -0.25, alpha: 4)
      let output = try weight.projected(x, adapters: [a,b]).asArray(Float.self)
      XCTAssertEqual(output, [-40,64,-40,64,-40,64])
      XCTAssertThrowsError(try weight.projected(.ones([3,32])))
      XCTAssertThrowsError(try weight.projected(x, adapters: [
        MLXLoRA(down: .ones([1,32]), up: .ones([2,1]), strength: 1)]))
    }
  }

  func testDenseBF16AndShapeAdmission() throws {
    let values: [UInt16] = [0x3f80,0x4000,0x4040,0x4080]
    try withTensorFile(tensors: [("w", [2,2], "BF16")], payloads: ["w":values.withUnsafeBytes { Data($0) }]) { url in
      let file = try SafeTensorFile(url: url)
      let weight = try MLXWeight(file:file,name:"w",shape:[2,2])
      XCTAssertEqual(weight.storageBytes,8)
      XCTAssertEqual(try weight.projected(MLXArray([Float(2),3],[1,2])).asArray(Float.self),[8,18])
      XCTAssertThrowsError(try MLXWeight(file:file,name:"w",shape:[4,1]))
      XCTAssertThrowsError(try MLXWeight(file:file,name:"missing",shape:[2,2]))
    }
    XCTAssertThrowsError(try MLXLoRA(down:.ones([2,64]),up:.ones([3,1]),strength:1))
    XCTAssertThrowsError(try MLXLoRA(down:.ones([1,64]),up:.ones([3,1]),strength:.nan))
  }
  func testBufferedPackedReadsCrossFourMiBWindowsWithoutChangingValues() throws {
    let count=3*1024*1024
    let words=(0..<count).map { UInt16($0%2 == 0 ? 0x3f80 : 0x4000) }
    try withTensorFile(tensors:[("factor",[count],"BF16")],payloads:["factor":words.withUnsafeBytes { Data($0) }]) { url in
      let file=try SafeTensorFile(url:url)
      let buffered=try MLXWeight.read(file,"factor",access:.buffered)
      XCTAssertEqual(buffered.dtype,.bfloat16)
      let actual=buffered.asType(.float32).asArray(Float.self)
      XCTAssertEqual(actual.count,count)
      XCTAssertTrue(actual.enumerated().allSatisfy { $0.element == ($0.offset%2 == 0 ? 1 : 2) })
    }
  }
}
