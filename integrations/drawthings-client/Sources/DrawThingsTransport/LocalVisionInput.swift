import CoreGraphics
import Foundation
import ImageIO
import LLM
import NNC

struct LocalPromptImage {
  let path: String
  let label: String
}

struct LocalVisionInput {
  let patches: Tensor<Float>
  let grids: [(t: Int, h: Int, w: Int)]

  static func prepare(_ images: [LocalPromptImage]) throws -> Self {
    guard !images.isEmpty, images.count <= 8 else { throw LocalTextError("vision_inputs_invalid") }
    var pieces = [Tensor<Float>]()
    var grids = [(t: Int, h: Int, w: Int)]()
    for input in images {
      let url = URL(fileURLWithPath: input.path).resolvingSymlinksInPath()
      guard let info = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
        info.isRegularFile == true, let size = info.fileSize, size > 0, size <= 64 * 1024 * 1024,
        let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
        let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
          kCGImageSourceCreateThumbnailFromImageAlways: true,
          kCGImageSourceCreateThumbnailWithTransform: true,
          kCGImageSourceThumbnailMaxPixelSize: 512
        ] as CFDictionary) else { throw LocalTextError("vision_image_unavailable") }
      // Bound the vision budget, retain the whole composition, and honor EXIF orientation.
      let width = min(512, max(32, Int((Double(image.width) / 32).rounded()) * 32))
      let height = min(512, max(32, Int((Double(image.height) / 32).rounded()) * 32))
      var bytes = [UInt8](repeating: 0, count: width * height * 4)
      try bytes.withUnsafeMutableBytes { buffer in
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let context = CGContext(data: buffer.baseAddress, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4, space: space,
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue)
          else { throw LocalTextError("vision_image_unavailable") }
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
      }
      // Qwen3.5 processor: RGB mean 0.5, standard deviation 0.5.
      var tensor = Tensor<Float>(.CPU, .NHWC(1, height, width, 3))
      for y in 0..<height { for x in 0..<width { for c in 0..<3 {
        tensor[0,y,x,c] = Float(bytes[(y * width + x) * 4 + c]) / 127.5 - 1
      } } }
      let prepared = Qwen3_5VisionPreprocess(tensor, size: (height, width))
      pieces.append(prepared.patches); grids.append(prepared.grid)
    }
    let rows = grids.reduce(0) { $0 + $1.t * $1.h * $1.w }
    var merged = Tensor<Float>(.CPU, .WC(rows, Qwen3_5VisionConfiguration.qwen3_5_4B.patchVectorSize))
    merged.withUnsafeMutableBytes { destination in
      var offset = 0
      for piece in pieces {
        piece.withUnsafeBytes { source in
          destination.baseAddress!.advanced(by: offset).copyMemory(from: source.baseAddress!, byteCount: source.count)
          offset += source.count
        }
      }
    }
    return Self(patches: merged, grids: grids)
  }
}
