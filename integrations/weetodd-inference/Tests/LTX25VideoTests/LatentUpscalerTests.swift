import XCTest
import Foundation
import TensorIO
import InferenceTestSupport
@testable import LTX25Video

final class LatentUpscalerTests: XCTestCase {
  func testPlanCountsFullContextAndRejectsBeforeAllocation() throws {
    var c = LatentUpscaleConfiguration()
    let plan = try LatentUpscalePlan(shape: [5,4,8,128],configuration: c)
    XCTAssertEqual(plan.outputShape,[5,8,16,128])
    XCTAssertGreaterThan(plan.activationBytes,5*8*16*1024*4*3)
    c.maximumActivationBytes = plan.activationBytes-1
    XCTAssertThrowsError(try LatentUpscalePlan(shape: [5,4,8,128],configuration: c))
    for shape in [[0,1,1,128],[1,1,1,64],[Int.max,1,1,128],[1,1,128]] {
      XCTAssertThrowsError(try LatentUpscalePlan(shape: shape))
    }
    c = LatentUpscaleConfiguration(); c.maximumWeightBytes = 1
    XCTAssertThrowsError(try LatentUpscalePlan(shape: [1,1,1,128],configuration: c))
  }
  func testZeroTemporalPaddingAndPerFrame2DConvolution() throws {
    let gpu = try VideoGPU()
    let x = try gpu.tensor([1,2],shape: [2,1,1,1])
    for depth in [1,3] {
      let w = try gpu.tensor([Float](repeating: 1,count: depth*9),shape: [1,1,1,depth*9])
      let b = try gpu.tensor([0],shape: [1,1,1,1])
      let y = try gpu.convolution(x,weights: w.buffer,bias: b.buffer,outputChannels: 1,
        causal: false,windowSites: 2,kernelDepth: depth,zeroTemporalPadding: true,checkCancelled: {})
      XCTAssertEqual(gpu.values(y),depth == 1 ? [1,2] : [3,3])
    }
  }
  func testGroupNormReducesWholeVolumeAndAddsResidualBeforeSilu() throws {
    let gpu = try VideoGPU(), shape = [2,2,3,64]
    let x = (0..<768).map { Float($0 % 97-48)/17 }
    let residual = x.map { $0*0.2 }
    let weight = (0..<64).map { Float($0)/100+0.7 }, bias = (0..<64).map { Float($0 % 5)/13 }
    let input = try gpu.tensor(x,shape: shape), r = try gpu.tensor(residual,shape: shape)
    let actual = try gpu.values(gpu.groupNormSilu(input,weight: weight,bias: bias,groups: 32,residual: r))
    var expected = x
    for g in 0..<32 {
      var indices: [Int] = []
      for site in 0..<12 { indices.append(site*64+g*2); indices.append(site*64+g*2+1) }
      var mean: Double = 0
      for i in indices { mean += Double(x[i]) }; mean /= Double(indices.count)
      var variance: Double = 0
      for i in indices { let d = Double(x[i])-mean; variance += d*d }; variance /= Double(indices.count)
      for i in indices {
        let v = Float((Double(x[i])-mean)/sqrt(variance+1e-5))*weight[i%64]+bias[i%64]+residual[i]
        expected[i] = v/(1+exp(-v))
      }
    }
    XCTAssertLessThan(zip(actual,expected).map { abs($0-$1) }.max()!,2e-6)
    XCTAssertThrowsError(try gpu.groupNormSilu(input,weight: weight,bias: bias,groups: 3))
  }
  func testUpscalerRejectsWrongCheckpointAtHeaderValidation() throws {
    try withTensorFile(tensors: [("wrong",[1],"F32")]) { url in
      XCTAssertThrowsError(try LatentUpscaler(checkpoint: url,statisticsCheckpoint: url))
    }
  }
  func testInstalledUpscalerMatchesIndependentMLX() throws {
    let e = ProcessInfo.processInfo.environment
    guard let checkpoint = e["WEETODD_UPSCALER"],let stats = e["WEETODD_VIDEO_VAE"],
      let oraclePath = e["WEETODD_UPSCALER_ORACLE"] else { throw XCTSkip("Installed upscaler qualification is opt-in") }
    let f = try SafeTensorFile(url: URL(fileURLWithPath: oraclePath))
    let shape = try XCTUnwrap(f.tensors["input"]?.shape).map(Int.init)
    let x = try f.readFloat32(named: "input"), expected = try f.readFloat32(named: "output")
    let model = try LatentUpscaler(checkpoint: URL(fileURLWithPath: checkpoint),statisticsCheckpoint: URL(fileURLWithPath: stats))
    var c = LatentUpscaleConfiguration(); c.maximumActivationBytes = 1
    XCTAssertThrowsError(try model.upscale(packed: x,shape: shape,configuration: c))
    var cancelledResident = false
    XCTAssertThrowsError(try model.upscale(packed: x,shape: shape,checkCancelled: {
      if model.residentWeightBytes > 0 { cancelledResident = true; throw CancellationError() }
    }))
    XCTAssertTrue(cancelledResident); XCTAssertEqual(model.residentWeightBytes,0)
    let start = Date()
    var checkedReentry = false
    let actual = try model.upscale(packed: x,shape: shape,progress: { _ in
      if !checkedReentry {
        checkedReentry = true
        XCTAssertThrowsError(try model.upscale(packed: x,shape: shape))
      }
    })
    XCTAssertEqual(actual.count,expected.count); XCTAssertEqual(model.residentWeightBytes,0)
    let peak = zip(actual,expected).map { abs($0-$1) }.max()!
    let square = zip(actual,expected).reduce(0.0) { $0+pow(Double($1.0)-Double($1.1),2) }
    let norm = expected.reduce(0.0) { $0+Double($1)*Double($1) }
    print("UPSCALER_PARITY seconds=\(Date().timeIntervalSince(start)) maxabs=\(peak) relative=\(sqrt(square/max(norm,1e-20)))")
    XCTAssertLessThan(peak,0.002); XCTAssertLessThan(sqrt(square/max(norm,1e-20)),0.0001)
  }
}
