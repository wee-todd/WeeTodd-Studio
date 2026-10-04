import Foundation

/// Separate, explicit HiRes admission; ordinary generation retains its limit.
public enum H3CanvasAdmission: String, Sendable, Codable {
  case ordinary
  case spatialRefinement = "spatial_refinement_v2"

  public var maximumPackedRows: Int { self == .ordinary ? 40_000 : 64_000 }
  public var maximumPixels: Int {
    self == .ordinary ? H3Geometry.maximumCanvasPixels : 1920 * 1088
  }

  public func validate(width: Int, height: Int) throws {
    guard width > 0, height > 0, width.isMultiple(of: 32), height.isMultiple(of: 32),
      width <= 4096, height <= 4096, width * height <= maximumPixels,
      self == .ordinary || (max(width, height) <= 1920 && min(width, height) <= 1088) else {
      throw H3CheckpointError.invalid("H3 canvas exceeds its explicit ordinary or spatial-refinement admission.")
    }
  }

  public func validatePackedRows(_ count: Int) throws {
    guard (1...maximumPackedRows).contains(count) else {
      throw H3CheckpointError.invalid("H3 packed text, conditioning and target AV exceed the selected canvas admission.")
    }
    if self == .spatialRefinement {
      // Conservative live tensor accounting for the shared staged block:
      // twelve hidden-width and four feed-width BF16 buffers, plus 8 GiB
      // for streamed weights, decoder/allocator workspace and host staging.
      // This is an admission estimate, not an observed process peak.
      let estimated = UInt64(count) * UInt64((12 * 5376 + 4 * 14336) * 2)
        + 8 * 1024 * 1024 * 1024
      guard estimated <= Self.maximumStageBytes else {
        throw H3CheckpointError.invalid("H3 spatial activation estimate exceeds the 32 GiB stage budget.")
      }
    }
  }

  public static let maximumStageBytes: UInt64 = 32 * 1024 * 1024 * 1024
}

/// Output-core tile plus exact convolution halo. Normalization never uses tiles.
struct H3UpscalerConvolutionPlan: Sendable {
  let frames: Int
  let height: Int
  let width: Int
  let maximumWorkspaceBytes: UInt64

  init(frames: Int = 8, height: Int = 32, width: Int = 32,
    maximumWorkspaceBytes: UInt64 = 512 * 1024 * 1024) throws {
    guard (1...8).contains(frames), (1...32).contains(height), (1...32).contains(width),
      maximumWorkspaceBytes > 0, maximumWorkspaceBytes <= 512 * 1024 * 1024 else {
      throw H3CheckpointError.invalid("Invalid bounded H3 learned-upscaler convolution plan.")
    }
    self.frames = frames; self.height = height; self.width = width
    self.maximumWorkspaceBytes = maximumWorkspaceBytes
  }

  func workspaceElements(channels: Int, kernel: Int) throws -> UInt64 {
    guard (1...512).contains(channels), [1, 3].contains(kernel) else {
      throw H3CheckpointError.invalid("Unsupported learned H3 convolution shape.")
    }
    return UInt64(frames * height * width * channels * kernel * kernel * kernel)
  }

  func validate(channels: Int, kernel: Int, bytesPerElement: Int) throws -> UInt64 {
    guard [2, 4].contains(bytesPerElement) else {
      throw H3CheckpointError.invalid("Unsupported H3 convolution dtype size.")
    }
    let elements = try workspaceElements(channels: channels, kernel: kernel)
    guard elements < UInt64(Int32.max), elements * UInt64(bytesPerElement) <= maximumWorkspaceBytes else {
      throw H3CheckpointError.invalid("H3 learned convolution exceeds its 32-bit workspace or tile budget.")
    }
    return elements * UInt64(bytesPerElement)
  }

  func tileCount(frames: Int, height: Int, width: Int) throws -> Int {
    guard (1...128).contains(frames), (1...120).contains(height), (1...120).contains(width) else {
      throw H3CheckpointError.invalid("H3 learned-upscaler latent geometry exceeds the admitted bound.")
    }
    return ((frames + self.frames - 1) / self.frames)
      * ((height + self.height - 1) / self.height)
      * ((width + self.width - 1) / self.width)
  }
}
