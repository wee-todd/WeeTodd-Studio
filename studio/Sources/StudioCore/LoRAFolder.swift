import CryptoKit
import Foundation

public struct LoRAFolder: Codable, Equatable, Identifiable {
  public var id: String { path }
  public var path: String
  public var enabled = true
  public var recursive = true
  public var modelHint: LoRAModel?
  public init(path: String) { self.path = path }
}

public struct LoRAFolderEntry: Decodable, Identifiable {
  public var id: String { path }
  public var name: String
  public var path: String
  public var sourceFolder: String
  public var status: String
  public var detail: String?
  public var loraModel: LoRAModel?
  public var loraProfile: String?
  public var loraLayout: String?
  public var loraAdalnInputGrid: String?
  public var asset: MediaAsset {
    var value = MediaAsset(name: name, kind: .lora, path: path, scope: .global)
    let digest = Array(SHA256.hash(data: Data(path.utf8)).prefix(16))
    value.id = UUID(uuid: (digest[0], digest[1], digest[2], digest[3], digest[4], digest[5],
      digest[6], digest[7], digest[8], digest[9], digest[10], digest[11], digest[12], digest[13],
      digest[14], digest[15]))
    value.loraModel = loraModel
    value.loraProfile = loraProfile
    value.loraLayout = loraLayout
    value.loraAdalnInputGrid = loraAdalnInputGrid
    return value
  }
}
