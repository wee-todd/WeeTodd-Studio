import Foundation
import MLX
import XCTest
import LTX25Engine
@testable import LTX25MLX

final class MLXReferenceVideoLayoutTests: XCTestCase {
  func testRippleGuideAppendsCleanVideoWithMatchingMotionPositions() throws {
    let geometry = try AVGeometry(width: 64, height: 64, frames: 9, fps: 24)
    let layout = try MLXReferenceVideoLayout(geometry: geometry, strength: 1)
    XCTAssertEqual(layout.videoTokens, 16)
    XCTAssertEqual(layout.positions, geometry.videoPositions + geometry.videoPositions)
    let target = MLXArray((0..<geometry.videoTokens*128).map { Float($0) / 1000 }, [geometry.videoTokens, 128])
    let guide = MLXArray((0..<geometry.videoTokens*128).map { Float($0+1) / 100 }, [geometry.videoTokens, 128])
    let prepared = try layout.prepare(generated: target, reference: guide)
    XCTAssertEqual(prepared.latent.shape, [16, 128])
    XCTAssertEqual(prepared.latent[0..<8].asArray(Float.self), target.asArray(Float.self))
    XCTAssertEqual(prepared.latent[8..<16].asArray(Float.self), guide.asArray(Float.self))
    XCTAssertEqual(prepared.condition.mask, [Float](repeating: 1, count: 8) + [Float](repeating: 0, count: 8))
    XCTAssertEqual(prepared.condition.clean[8..<16].asArray(Float.self), guide.asArray(Float.self))
    XCTAssertThrowsError(try layout.prepare(generated: target, reference: target[0..<7]))
  }

  func testRippleGuideStrengthAndTokenBudgetAreAdmittedBeforeSampling() throws {
    let geometry = try AVGeometry(width: 64, height: 64, frames: 9, fps: 24)
    XCTAssertEqual(try MLXReferenceVideoLayout(geometry: geometry, strength: 0.25).referenceMask, 0.75)
    XCTAssertThrowsError(try MLXReferenceVideoLayout(geometry: geometry, strength: .nan))
    let excessive = try AVGeometry(width: 1024, height: 1024, frames: 513, fps: 24)
    XCTAssertThrowsError(try MLXReferenceVideoLayout(geometry: excessive, strength: 1))
  }

  func testRippleAppendsIndependentlyTimedImageAnchorsAfterFullRateGuide() throws {
    let geometry = try AVGeometry(width: 64, height: 64, frames: 17, fps: 24)
    let anchors = [RippleImageAnchor(frame: 8, strength: 0.9), RippleImageAnchor(frame: 16, strength: 1)]
    let layout = try MLXReferenceVideoLayout(geometry: geometry, strength: 1, anchors: anchors)
    let frameTokens = geometry.latentHeight * geometry.latentWidth
    XCTAssertEqual(layout.videoTokens, geometry.videoTokens * 2 + frameTokens * 2)
    let target = MLXArray.zeros([geometry.videoTokens, 128])
    let guide = MLXArray.ones([geometry.videoTokens, 128])
    let edits = [MLXArray([Float](repeating: 2, count: frameTokens * 128), [frameTokens, 128]),
      MLXArray([Float](repeating: 3, count: frameTokens * 128), [frameTokens, 128])]
    let prepared = try layout.prepare(generated: target, reference: guide, anchors: edits)
    XCTAssertEqual(prepared.latent.shape, [layout.videoTokens, 128])
    XCTAssertEqual(Array(prepared.condition.mask.suffix(frameTokens)), [Float](repeating: 0, count: frameTokens))
    XCTAssertEqual(prepared.condition.mask[prepared.condition.mask.count - frameTokens * 2],
      0.1, accuracy: 0.00001)
    XCTAssertEqual(prepared.latent[geometry.videoTokens * 2..<layout.videoTokens].asArray(Float.self),
      edits.flatMap { $0.asArray(Float.self) })
    let anchorPositions = Array(layout.positions.suffix(frameTokens * 2 * 3))
    XCTAssertEqual(anchorPositions[0], Float(8.5 / 24), accuracy: 0.00001)
    XCTAssertEqual(anchorPositions[frameTokens * 3], Float(16.5 / 24), accuracy: 0.00001)
    XCTAssertThrowsError(try layout.prepare(generated: target, reference: guide, anchors: [edits[0]]))
    XCTAssertThrowsError(try MLXReferenceVideoLayout(geometry: geometry, strength: 1,
      anchors: [RippleImageAnchor(frame: 17, strength: 1)]))
    XCTAssertThrowsError(try MLXReferenceVideoLayout(geometry: geometry, strength: 1,
      anchors: [RippleImageAnchor(frame: 0, strength: 1)]))
    XCTAssertThrowsError(try MLXReferenceVideoLayout(geometry: geometry, strength: 1,
      anchors: Array(anchors.reversed())))
  }
}
