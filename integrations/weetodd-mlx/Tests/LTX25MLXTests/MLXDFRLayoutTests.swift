import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXDFRLayoutTests: XCTestCase {
  func testCanvasSelectsOfficialSegmentAndPadsOnlyTheTail() throws {
    let padded = try MLXDFRCanvas(frames: 41)
    XCTAssertEqual(padded.frames, 49)
    XCTAssertEqual(padded.segmentFrames, 24)
    XCTAssertEqual(padded.slotFrames, [24, 48])
    let exact = try MLXDFRCanvas(frames: 97)
    XCTAssertEqual(exact.frames, 97)
    XCTAssertEqual(exact.segmentFrames, 32)
    XCTAssertEqual(exact.slotFrames, [32, 64, 96])
    XCTAssertThrowsError(try MLXDFRCanvas(frames: 42))
  }

  func testStageOneGeneratedSlotsRemainAfterMainLatents() throws {
    let g = try AVGeometry(width: 64, height: 32, frames: 49, fps: 24)
    let layout = try MLXDFRLayout(geometry: g, slotFrames: [24, 48])
    XCTAssertEqual(layout.slotTokens, 4)
    XCTAssertEqual(layout.videoTokens, g.videoTokens + 4)
    XCTAssertEqual(layout.positions.count, layout.videoTokens * 3)
    XCTAssertEqual(layout.positions[g.videoTokens * 3], Float(24.5 / 24))
    let prepared = try layout.prepare(generated: .ones([g.videoTokens, 128]))
    XCTAssertEqual(prepared.latent.shape, [layout.videoTokens, 128])
    XCTAssertEqual(prepared.condition.mask, [Float](repeating: 1, count: layout.videoTokens))
    XCTAssertEqual(prepared.latent[g.videoTokens, 0].item(Float.self), 0)
    XCTAssertEqual(prepared.latent[0, 0].item(Float.self), 1)
  }

  func testGeneratedSlotsReceiveStageNoiseBeforeSampling() throws {
    let geometry=try AVGeometry(width:64,height:32,frames:49,fps:24)
    let layout=try MLXDFRLayout(geometry:geometry,slotFrames:[24,48])
    let latent=MLXArray.ones([layout.videoTokens,128])*0.2
    let noise=MLXArray.ones([layout.slotTokens,128])*0.8
    let first=try layout.noiseSlots(latent,noise:noise,sigma:1)
    XCTAssertEqual(first[0,0].item(Float.self),0.2,accuracy:0.0001)
    XCTAssertEqual(first[geometry.videoTokens,0].item(Float.self),0.8,accuracy:0.0001)
    let second=try layout.noiseSlots(latent,noise:noise,sigma:0.9)
    XCTAssertEqual(second[geometry.videoTokens,0].item(Float.self),0.74,accuracy:0.0001)
    XCTAssertThrowsError(try layout.noiseSlots(latent,
      noise:.ones([layout.slotTokens+1,128]),sigma:0.9))
  }

  func testStageTwoKeepsReferenceBeforeSeededSlotsAndScalesPositions() throws {
    let low = try AVGeometry(width: 64, height: 32, frames: 49, fps: 24)
    let high = try AVGeometry(width: 128, height: 64, frames: 49, fps: 24)
    let layout = try MLXDFRLayout(geometry: high, slotFrames: [24, 48], reference: low)
    XCTAssertEqual(layout.referenceTokens, low.videoTokens)
    XCTAssertEqual(layout.slotTokens, 16)
    let refPosition = high.videoTokens * 3
    XCTAssertEqual(layout.positions[refPosition], low.videoPositions[0])
    XCTAssertEqual(layout.positions[refPosition + 1], low.videoPositions[1] * 2)
    XCTAssertEqual(layout.positions[refPosition + 2], low.videoPositions[2] * 2)
    let prepared = try layout.prepare(generated: .ones([high.videoTokens, 128]),
      reference: .ones([low.videoTokens, 128]) * 0.3,
      slots: .ones([layout.slotTokens, 128]) * 0.7)
    XCTAssertEqual(prepared.condition.mask[high.videoTokens], 0)
    XCTAssertEqual(prepared.condition.mask.last, 1)
    XCTAssertEqual(prepared.latent[high.videoTokens, 0].item(Float.self), 0.3, accuracy: 0.0001)
    XCTAssertEqual(prepared.latent[high.videoTokens + low.videoTokens, 0].item(Float.self), 0.7, accuracy: 0.0001)
    XCTAssertThrowsError(try MLXDFRLayout(geometry: high, slotFrames: [25], reference: low))
  }

  func testEndpointImagesPrecedeDFRReferenceAndGeneratedSlots() throws {
    let low = try AVGeometry(width:64,height:32,frames:49,fps:24)
    let high = try AVGeometry(width:128,height:64,frames:49,fps:24)
    let layout = try MLXDFRLayout(geometry:high,slotFrames:[24,48],reference:low,
      firstStrength:1,lastStrength:0.8)
    let frameTokens=high.latentHeight*high.latentWidth
    let first = MLXArray.ones([frameTokens,128])*0.2
    let last = MLXArray.ones([frameTokens,128])*0.8
    let prepared = try layout.prepare(generated:.ones([high.videoTokens,128])*(-1),
      first:first,last:last,reference:.ones([low.videoTokens,128])*0.4)
    XCTAssertEqual(layout.endpointTokens,high.videoTokens+frameTokens)
    XCTAssertEqual(prepared.condition.mask[0],0)
    XCTAssertEqual(prepared.condition.mask[high.videoTokens],0.2,accuracy:0.0001)
    XCTAssertEqual(prepared.condition.mask[layout.endpointTokens],0)
    XCTAssertEqual(prepared.condition.mask.last,1)
    XCTAssertEqual(prepared.latent[0,0].item(Float.self),0.2,accuracy:0.0001)
    XCTAssertEqual(prepared.latent[high.videoTokens,0].item(Float.self),0.8,accuracy:0.0001)
    XCTAssertEqual(prepared.latent[layout.endpointTokens,0].item(Float.self),0.4,accuracy:0.0001)
    XCTAssertThrowsError(try layout.prepare(generated:.zeros([high.videoTokens,128]),first:first))
  }

  func testPaddedDFRLastImageTargetsRequestedFrame() throws {
    let geometry=try AVGeometry(width:64,height:32,frames:49,fps:24)
    let layout=try MLXDFRLayout(geometry:geometry,slotFrames:[24,48],
      firstStrength:1,lastStrength:0.8,lastFrame:40)
    XCTAssertEqual(layout.positions[geometry.videoTokens*3],Float(40.5/24),accuracy:0.0001)
    XCTAssertThrowsError(try MLXDFRLayout(geometry:geometry,slotFrames:[24,48],
      firstStrength:1,lastStrength:0.8,lastFrame:41))
  }
}
