import Diffusion
import Foundation
import ImageIO
import LocalImageGenerator
import ModelZoo
import NNC
import UniformTypeIdentifiers

/// Best-effort latent visualization. No VAE weights, final frames, or reference inputs.
final class LivePreview {
  let file: URL
  let version: ModelVersion
  private var lastUpdate = -Double.infinity
  private var revision = 0
  init(root: URL, version: ModelVersion) {
    file = root.appendingPathComponent("live-preview.png"); self.version = version
  }
  func receive(_ data: Data, now: Double = ProcessInfo.processInfo.systemUptime) -> [String: Any]? {
    guard now - lastUpdate >= 0.5, !data.isEmpty, data.count <= 16 * 1024 * 1024 else { return nil }
    lastUpdate = now
    guard let tensor = Tensor<FloatType>(data: data, using: [.zip, .fpzip]) else { return nil }
    let shape = Array(tensor.shape)
    let channels: Int
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b, .pixart, .auraflow, .kandinsky21: channels = 4
    case .sd3, .sd3Large, .flux1, .hiDreamI1, .zImage, .qwenImage, .krea2: channels = 16
    case .ernieImage, .flux2, .flux2_4b, .flux2_9b, .ideogram4: channels = 32
    default: return nil
    }
    guard shape.count == 4, shape[0] == 1, (1...512).contains(shape[1]),
      (1...512).contains(shape[2]), shape[3] == channels else { return nil }
    guard let image = ImageConverter.cgImages(fromLatent: tensor, canUseTAESD: false, version: version).0.first
    else { return nil }
    let bytes = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(bytes, UTType.png.identifier as CFString, 1, nil)
    else { return nil }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { return nil }
    do { try (bytes as Data).write(to: file, options: .atomic) } catch { return nil }
    revision += 1
    return ["previewPath": file.path, "previewRevision": revision]
  }
}
