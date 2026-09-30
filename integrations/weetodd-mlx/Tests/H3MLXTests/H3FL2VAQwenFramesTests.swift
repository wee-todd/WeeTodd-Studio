import Foundation
import XCTest
@testable import H3MLX

final class H3FL2VAQwenFramesTests: XCTestCase {
  func testVisualThumbnailIsBoundedAndLeavesFullCanvasUntouched() throws {
    let width = 768, height = 448
    var pixels = [UInt8](repeating: 0, count: width * height * 3)
    for y in 0..<height {
      for x in 0..<width {
        pixels[(y * width + x) * 3] = UInt8(x / 3)
        pixels[(y * width + x) * 3 + 1] = UInt8(y / 2)
      }
    }
    let original = H3StillReference(rgb8: Data(pixels), width: width, height: height)
    let bounded = try H3FL2VAQwenFrames.thumbnail(original)
    XCTAssertEqual(bounded.width, 256)
    XCTAssertEqual(bounded.height, 160)
    XCTAssertEqual(bounded.rgb8.count, 256 * 160 * 3)
    XCTAssertEqual(original.rgb8.count, width * height * 3)
    XCTAssertEqual(bounded.rgb8[0], 0)
    XCTAssertGreaterThan(bounded.rgb8[bounded.rgb8.count - 3], 200)
  }

  func testInstalledTokenizerFallsBackOnlyWhenFullCanvasExceedsWindow() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_QWEN_TOKENIZER"] else {
      throw XCTSkip("Set an installed H3 tokenizer for Qwen visual-window admission.")
    }
    let tokenizer = try H3QwenTokenizer(url: URL(fileURLWithPath: path))
    let image = H3StillReference(rgb8: Data(count: 768 * 448 * 3),
      width: 768, height: 448)
    let two = try H3FL2VAQwenFrames.prepare(images: [image, image],
      prompt: "A person turns slowly.", tokenizer: tokenizer)
    XCTAssertEqual(two.images[0].width, 768)
    let three = try H3FL2VAQwenFrames.prepare(images: [image, image, image],
      prompt: "A person turns slowly.", tokenizer: tokenizer)
    XCTAssertEqual(three.images.count, 3)
    XCTAssertEqual(three.images[0].width, 256)
    XCTAssertLessThanOrEqual(three.request.tags.count, 1024)
  }
}
