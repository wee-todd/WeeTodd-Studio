import Foundation

/// Only the Qwen visual presentation may need a smaller copy. The VAE still
/// receives the full generation-canvas keyframes and retains their latent rows.
enum H3FL2VAQwenFrames {
  struct Prepared {
    let images: [H3StillReference]
    let request: H3QwenRequest
  }

  static func prepare(images: [H3StillReference], prompt: String,
    tokenizer: H3QwenTokenizer) throws -> Prepared {
    func presentation(_ frames: [H3StillReference]) throws -> H3QwenRequest {
      try H3QwenRequest.keyframes(prompt: prompt, grids: frames.map {
        H3QwenRequest.Grid(temporal: 1, height: $0.height / 16, width: $0.width / 16)
      }, tokenizer: tokenizer)
    }
    if let request = try? presentation(images) {
      return Prepared(images: images, request: request)
    }
    let bounded = try images.map { try thumbnail($0) }
    return Prepared(images: bounded, request: try presentation(bounded))
  }

  static func thumbnail(_ image: H3StillReference) throws -> H3StillReference {
    let ratio = min(1, 256.0 / Double(max(image.width, image.height)))
    let width = max(64, Int((Double(image.width) * ratio / 32).rounded()) * 32)
    let height = max(64, Int((Double(image.height) * ratio / 32).rounded()) * 32)
    guard width <= image.width, height <= image.height,
      image.rgb8.count == image.width * image.height * 3 else {
      throw H3CheckpointError.invalid("Invalid H3 keyframe visual thumbnail.")
    }
    if width == image.width && height == image.height { return image }
    let source = [UInt8](image.rgb8)
    var output = [UInt8](repeating: 0, count: width * height * 3)
    for y in 0..<height {
      let sourceY = (Double(y) + 0.5) * Double(image.height) / Double(height) - 0.5
      let lowY = max(0, min(image.height - 1, Int(floor(sourceY))))
      let highY = min(image.height - 1, lowY + 1)
      let fy = max(0, min(1, sourceY - Double(lowY)))
      for x in 0..<width {
        let sourceX = (Double(x) + 0.5) * Double(image.width) / Double(width) - 0.5
        let lowX = max(0, min(image.width - 1, Int(floor(sourceX))))
        let highX = min(image.width - 1, lowX + 1)
        let fx = max(0, min(1, sourceX - Double(lowX)))
        for channel in 0..<3 {
          let a = Double(source[(lowY * image.width + lowX) * 3 + channel])
          let b = Double(source[(lowY * image.width + highX) * 3 + channel])
          let c = Double(source[(highY * image.width + lowX) * 3 + channel])
          let d = Double(source[(highY * image.width + highX) * 3 + channel])
          let value = (a * (1 - fx) + b * fx) * (1 - fy) +
            (c * (1 - fx) + d * fx) * fy
          output[(y * width + x) * 3 + channel] = UInt8(max(0, min(255, Int(value.rounded()))))
        }
      }
    }
    return H3StillReference(rgb8: Data(output), width: width, height: height)
  }
}
