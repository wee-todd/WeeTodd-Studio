import Foundation
import XCTest
@testable import LTX25Video

final class VideoGPUTests: XCTestCase {
  func testCancellationAfterBatchedGPUWorkAllowsCleanRetry() throws {
    let gpu = try VideoGPU()
    let input = try gpu.tensor((0..<128).map { Float($0)/128 },shape: [1,4,4,8])
    let weight = try gpu.tensor([Float](repeating: 0.01,count: 2*8*27),shape: [1,1,1,2*8*27])
    let bias = try gpu.tensor([0,0],shape: [1,1,1,2])
    func run(_ check: () throws -> Void = {}) throws -> [Float] {
      try autoreleasepool {
        try gpu.values(gpu.convolution(input,weights: weight.buffer,bias: bias.buffer,
          outputChannels: 2,causal: false,windowSites: 2,normalizeInput: true,
          windowsPerCommand: 2,checkCancelled: check))
      }
    }
    let expected = try run(), baseline = gpu.device.currentAllocatedSize
    for _ in 0..<3 {
      var checks = 0
      XCTAssertThrowsError(try run { checks += 1; if checks == 5 { throw CancellationError() } })
      XCTAssertEqual(gpu.lastConvolutionCommandCount,2)
      XCTAssertEqual(try run(),expected)
    }
    XCTAssertLessThanOrEqual(gpu.device.currentAllocatedSize,baseline+4*1024*1024)
  }
  func testFusedNormalizationAndBatchedWindowsMatchSeparatePathExactly() throws {
    let gpu = try VideoGPU()
    for channels in [2,128] {
      let input = try gpu.tensor((0..<(36*channels)).map { Float($0 % 19 - 9)/13 },shape: [3,3,4,channels])
      let w = try gpu.tensor((0..<(3*channels*27)).map { Float($0 % 17 - 8)/23 },shape: [1,1,1,3*channels*27])
      let b = try gpu.tensor([0.1,-0.2,0.3],shape: [1,1,1,3])
      for causal in [false,true] {
        let expected = try gpu.values(gpu.convolution(gpu.normalize(input),weights: w.buffer,bias: b.buffer,
          outputChannels: 3,causal: causal,windowSites: 5,checkCancelled: {}))
        for batch in [1,3] {
          let actual = try gpu.values(gpu.convolution(input,weights: w.buffer,bias: b.buffer,
            outputChannels: 3,causal: causal,windowSites: 5,normalizeInput: true,windowsPerCommand: batch,checkCancelled: {}))
          XCTAssertEqual(actual,expected)
          XCTAssertEqual(gpu.lastNormalizationWorkspaceBytes,36*4)
          XCTAssertEqual(gpu.lastConvolutionCommandCount,(8+batch-1)/batch+2)
        }
      }
    }
  }
  func testConvolutionMatchesIndependentMLXAtTemporalAndSpatialEdges() throws {
    let fixture = try JSONDecoder().decode([String: [Float]].self, from: Data(contentsOf:
      Bundle.module.url(forResource: "primitives", withExtension: "json", subdirectory: "Fixtures")!))
    let gpu = try VideoGPU()
    for causal in [false, true] {
      let actual = try gpu.testConvolution(fixture["input"]!, shape: [3,3,4,2],
        weight: fixture["weight"]!, outputChannels: 3, bias: fixture["bias"]!, causal: causal,
        windowSites: 5)
      let expected = fixture[causal ? "causal" : "symmetric"]!
      XCTAssertEqual(actual.count, expected.count)
      for (a,b) in zip(actual,expected) { XCTAssertEqual(a,b,accuracy: 2e-6) }
    }
  }
  func testPixelNormalizationAndSiluMatchesMLX() throws {
    let fixture = try JSONDecoder().decode([String: [Float]].self, from: Data(contentsOf:
      Bundle.module.url(forResource: "primitives", withExtension: "json", subdirectory: "Fixtures")!))
    let actual = try VideoGPU().testNormalize(fixture["input"]!, channels: 2)
    for (a,b) in zip(actual,fixture["norm"]!) { XCTAssertEqual(a,b,accuracy: 2e-6) }
  }
  func testShuffleDropsFirstTemporalFrameAndUnpatchUsesWidthBeforeHeight() throws {
    let gpu = try VideoGPU()
    let input = (0..<16).map(Float.init)
    XCTAssertEqual(try gpu.testShuffle(input, shape: [2,1,1,8], spatial: 2, temporal: 2, unpatch: false),
      [4,5,6,7,8,9,10,11,12,13,14,15])
    XCTAssertEqual(try gpu.testShuffle((0..<16).map(Float.init), shape: [1,1,1,16],
      spatial: 4, temporal: 1, unpatch: true), [0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15])
  }
}

private extension VideoGPU {
  func testConvolution(_ x: [Float], shape: [Int], weight: [Float], outputChannels: Int,
    bias: [Float], causal: Bool, windowSites: Int) throws -> [Float] {
    let input = try tensor(x,shape: shape)
    let weights = try tensor(weight,shape: [1,1,1,weight.count])
    let biases = try tensor(bias,shape: [1,1,1,bias.count])
    return try values(convolution(input,weights: weights.buffer,bias: biases.buffer,
      outputChannels: outputChannels,causal: causal,windowSites: windowSites,checkCancelled: {}))
  }
  func testNormalize(_ x: [Float], channels: Int) throws -> [Float] {
    try values(normalize(tensor(x,shape: [1,1,x.count/channels,channels])))
  }
  func testShuffle(_ x: [Float], shape: [Int], spatial: Int, temporal: Int, unpatch: Bool) throws -> [Float] {
    try values(shuffle(tensor(x,shape: shape),spatial: spatial,temporal: temporal,unpatch: unpatch))
  }
}
