import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXDFRTemporalLayoutTests: XCTestCase {
  func testFrozenAudioUsesOfficialCeilingTokenCount() throws {
    let source=MLXArray.ones([52,128])
    let tile=try MLXDFRFrozenAudio.tile(source,pixelStart:24,frames:73,
      playbackFPS:48,sourceSeconds:49.0/24)
    XCTAssertEqual(tile.latent.shape,[31,128])
    XCTAssertEqual(tile.positions.shape,[31,1])
  }

  func testLaterTilePinsPlaneAndPriorOutputThroughSeam() throws {
    let geometry = try AVGeometry(width:64,height:32,frames:73,fps:60)
    let frameTokens=geometry.latentHeight*geometry.latentWidth
    let layout=try MLXDFRTemporalLayout(geometry:geometry,slots:[48],anchors:[])
    let prefix=MLXArray.ones([4*frameTokens,128])*0.25
    let prepared=try layout.prepare(generated:.ones([geometry.videoTokens,128]),
      initialSlots:.zeros([layout.slotTokens,128]),pinnedPrefix:prefix)
    XCTAssertEqual(Array(prepared.condition.mask.prefix(4*frameTokens)),
      Array(repeating:Float(0),count:4*frameTokens))
    XCTAssertEqual(prepared.condition.mask[4*frameTokens],1)
    XCTAssertEqual(prepared.latent[0,0].item(Float.self),0.25,accuracy:0.0001)
    XCTAssertEqual(prepared.latent[4*frameTokens,0].item(Float.self),1,accuracy:0.0001)
    XCTAssertThrowsError(try layout.prepare(generated:.ones([geometry.videoTokens,128]),
      initialSlots:.zeros([layout.slotTokens,128]),
      pinnedPrefix:.ones([frameTokens+1,128])))
  }
}
