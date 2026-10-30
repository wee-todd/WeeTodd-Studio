import Foundation
import XCTest
@testable import StudioCore

final class VideoProcessingPixelReceiptTests:XCTestCase {
  func testPixelLosslessArtifactAndPreservedPCMRemainSeparateMovieInputs() throws {
    let video=FlashVSRFile(path:"/delivered.mov",sha256:String(repeating:"a",count:64),bytes:1)
    let restored=FlashVSRFile(path:"/restored.mkv",sha256:String(repeating:"b",count:64),bytes:2)
    let media=FlashVSRMediaInfo(width:64,height:64,frames:24,fps:try .init(numerator:24,denominator:1),firstTimestamp:0,hasAudio:true,audioRate:32000,audioChannels:2)
    let native=NativePixelCQReceipt(requestFingerprint:"fixture",source:video,video:video,workerMetadata:video,workerRecipe:video,media:media,elapsedSeconds:0,sourceAudioSHA256:"pcm",outputAudioSHA256:"pcm",restored:restored)
    let receipt=VideoProcessingReceipt.pixelCQ(native)
    XCTAssertEqual(receipt.restored,restored)
    XCTAssertEqual(receipt.audio,video)
    XCTAssertEqual(receipt.video,video)
    var legacy=try JSONSerialization.jsonObject(with:JSONEncoder().encode(native)) as! [String:Any]
    legacy.removeValue(forKey:"restored")
    let old=VideoProcessingReceipt.pixelCQ(try JSONDecoder().decode(NativePixelCQReceipt.self,from:JSONSerialization.data(withJSONObject:legacy)))
    XCTAssertEqual(old.restored,video)
    XCTAssertEqual(old.audio,video)
  }
}
