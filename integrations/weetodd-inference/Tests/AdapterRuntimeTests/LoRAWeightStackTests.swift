import XCTest
import TensorIO
import InferenceTestSupport
@testable import AdapterRuntime

final class LoRAWeightStackTests: XCTestCase {
  func fixture<T>(_ a: [Float], _ b: [Float], metadata: [String:String] = [:],
    _ body: (URL) throws -> T) throws -> T {
    try withTensorFile(metadata: metadata, tensors: [("layer.lora_A.weight",[2,3],"F32"),
      ("layer.lora_B.weight",[2,2],"F32")], payloads: [
        "layer.lora_A.weight":a.withUnsafeBytes { Data($0) },
        "layer.lora_B.weight":b.withUnsafeBytes { Data($0) }],body)
  }
  func stack(_ adapters: [LoRAAdapter], budget: UInt64 = 64*1024*1024) throws -> LoRAWeightStack {
    try LoRAWeightStack(adapters: adapters,maximumWorkspaceBytes: budget,rowsPerTile: 1) { file,strength in
      try LoRAPlan(file: file,strength: strength,targetShapes: ["layer":[2,3]])
    }
  }
  func testAppliesAlphaAndSignedStrengthToBaseWithoutChangingUnmatchedWeights() throws {
    try fixture([1,2,3,4,5,6],[1,2,3,4],metadata: ["lora_alpha":"1"]) { url in
      let value = try stack([LoRAAdapter(path: url.path,strength: -0.5)])
      let result = try value.read("layer.weight",shape: [2,3]) { [10,10,10,10,10,10] }
      XCTAssertEqual(result,[7.75,7,6.25,5.25,3.5,1.75])
      XCTAssertEqual(try value.read("layer.bias",shape: [2]) { [1,2] },[1,2])
      XCTAssertLessThanOrEqual(value.admittedWorkspaceBytes,64*1024*1024)
    }
  }
  func testOrderIsPreservedAndZeroDisabledAreExactNoOps() throws {
    try fixture([1,0,0,0,0,0],[1,0,0,0]) { url in
      let ordered = try stack([LoRAAdapter(path: url.path,strength: -16777216),LoRAAdapter(path: url.path,strength: 1)])
      let reversed = try stack([LoRAAdapter(path: url.path,strength: 1),LoRAAdapter(path: url.path,strength: -16777216)])
      XCTAssertEqual(try ordered.read("layer.weight",shape: [2,3]) { [16777216,0,0,0,0,0] }[0],1)
      XCTAssertEqual(try reversed.read("layer.weight",shape: [2,3]) { [16777216,0,0,0,0,0] }[0],0)
      let noOp = try stack([LoRAAdapter(path: url.path,strength: 0),LoRAAdapter(path: "/missing",strength: 1,enabled: false)])
      XCTAssertEqual(try noOp.read("layer.weight",shape: [2,3]) { [-0,1,2,3,4,5] },[-0,1,2,3,4,5])
      XCTAssertEqual(noOp.admittedWorkspaceBytes,0)
    }
  }
  func testBudgetAndRequestedShapeFailBeforeBaseLoad() throws {
    try fixture([1,2,3,4,5,6],[1,2,3,4]) { url in
      XCTAssertThrowsError(try stack([LoRAAdapter(path: url.path,strength: 1)],budget: 16))
      let value = try stack([LoRAAdapter(path: url.path,strength: 1)])
      var loaded = false
      XCTAssertThrowsError(try value.read("layer.weight",shape: [3,2]) { loaded = true; return Array(repeating: 0,count: 6) })
      XCTAssertFalse(loaded)
    }
  }
  func testCancellationAndReentryReleaseWorkspaceForRetry() throws {
    try fixture([1,2,3,4,5,6],[1,2,3,4]) { url in
      let value = try stack([LoRAAdapter(path: url.path,strength: 1)])
      var checks = 0
      XCTAssertThrowsError(try value.read("layer.weight",shape: [2,3],checkCancelled: {
        checks += 1; if checks == 3 { throw CancellationError() }
      }) { Array(repeating: 0,count: 6) })
      let result = try value.read("layer.weight",shape: [2,3]) {
        XCTAssertThrowsError(try value.read("layer.weight",shape: [2,3]) { [] })
        return Array(repeating: 0,count: 6)
      }
      XCTAssertEqual(result,[9,12,15,19,26,33])
    }
  }
  func testRejectsNonfinitePayloadAndChangedCheckpoint() throws {
    try fixture([.nan,2,3,4,5,6],[1,2,3,4]) { url in
      let value = try stack([LoRAAdapter(path: url.path,strength: 1)])
      XCTAssertThrowsError(try value.read("layer.weight",shape: [2,3]) { Array(repeating: 0,count: 6) })
    }
    try fixture([1,2,3,4,5,6],[1,2,3,4]) { url in
      let value = try stack([LoRAAdapter(path: url.path,strength: 1)])
      let file = try FileHandle(forWritingTo: url); try file.truncate(atOffset: 16); try file.close()
      var loaded = false
      XCTAssertThrowsError(try value.read("layer.weight",shape: [2,3]) { loaded = true; return Array(repeating: 0,count: 6) })
      XCTAssertFalse(loaded)
    }
  }
}

extension LoRAWeightStackTests {
  func testF64MatricesAndOverflowingProductsAreHandledExplicitly() throws {
    let nameA = "layer.lora_A.weight",nameB = "layer.lora_B.weight"
    let a: [Double] = [1,2,3,4,5,6],b: [Double] = [1,2,3,4]
    try withTensorFile(tensors: [(nameA,[2,3],"F64"),(nameB,[2,2],"F64")],
      payloads: [nameA:a.withUnsafeBytes { Data($0) },nameB:b.withUnsafeBytes { Data($0) }]) { url in
      let value = try stack([LoRAAdapter(path: url.path,strength: 1)])
      XCTAssertEqual(try value.read("layer.weight",shape: [2,3]) { Array(repeating: 0,count: 6) },[9,12,15,19,26,33])
    }
    try fixture(Array(repeating: .greatestFiniteMagnitude,count: 6),[1,2,3,4]) { url in
      let value = try stack([LoRAAdapter(path: url.path,strength: 1)])
      XCTAssertThrowsError(try value.read("layer.weight",shape: [2,3]) { Array(repeating: 0,count: 6) })
    }
  }
}

extension LoRAWeightStackTests {
  func testBatchedReadsPreserveMathAcrossPartialReadAndComputeWindows() throws {
    let a: [Float] = [0.3,-0.6,0.9,1.2,-1.5,1.8]
    let b = (0..<18).map { Float($0-9)*0.07 }
    try withTensorFile(tensors: [("layer.lora_A.weight",[2,3],"F32"),("layer.lora_B.weight",[9,2],"F32")],
      payloads: ["layer.lora_A.weight":a.withUnsafeBytes { Data($0) },"layer.lora_B.weight":b.withUnsafeBytes { Data($0) }]) { url in
      func make(_ rows: Int) throws -> LoRAWeightStack {
        try LoRAWeightStack(adapters: [LoRAAdapter(path: url.path,strength: 0.8)],rowsPerTile: 2,rowsPerRead: rows) {
          try LoRAPlan(file: $0,strength: $1,targetShapes: ["layer":[9,3]])
        }
      }
      let small = try make(2),batched = try make(6)
      let expected = try small.read("layer.weight",shape: [9,3]) { Array(repeating: 0.1,count: 27) }
      XCTAssertEqual(try batched.read("layer.weight",shape: [9,3]) { Array(repeating: 0.1,count: 27) },expected)
      XCTAssertEqual(batched.admittedWorkspaceBytes-small.admittedWorkspaceBytes,4*2*4)
      XCTAssertThrowsError(try make(3),"Read windows must align to compute tiles to preserve accumulation kernels")
    }
  }
}
