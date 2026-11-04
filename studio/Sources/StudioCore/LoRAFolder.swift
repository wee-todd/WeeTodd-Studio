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
  public var adapterKind: LTXLibraryAdapterKind?
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

public enum LTXLibraryAdapterKind: String, Codable, Identifiable {
  case distilledRefinement = "distilled_refinement", dfr, msr, ingredients, union
  case motionTrack = "motion_track", crossview
  public var id: String { rawValue }
  public var componentKey: String { switch self {
    case .distilledRefinement: "distilled_lora_path"
    case .dfr: "dfr_detailing_lora_path"
    case .msr: "msr_lora_path"
    case .ingredients: "ingredients_lora_path"
    case .union: "union_lora_path"
    case .motionTrack: "motion_track_lora_path"
    case .crossview: "crossview_lora_path"
  } }
  public var presetID: String { "swift-ltx25-" + (self == .distilledRefinement ? "guided"
    : self == .dfr ? "dfr-spatial" : self == .motionTrack ? "motion-track" : rawValue) }
  public var label: String { switch self {
    case .distilledRefinement: "Dev refinement adapter"
    case .dfr: "DFR spatial refinement"
    case .msr: "MSR references"
    case .ingredients: "Ingredients reference sheet"
    case .union: "Union control guide"
    case .motionTrack: "Motion Track guide"
    case .crossview: "CrossView source and guide"
  } }
  public var requirements: String { switch self {
    case .distilledRefinement: "Dev configs resolve this refinement adapter automatically."
    case .dfr: "Requires an independent distilled shot and the DFR component setup."
    case .msr: "Requires appearance or voice references and the MSR component setup."
    case .ingredients: "Requires a described reference sheet and the Ingredients component setup."
    case .union: "Requires a Canny, depth or pose guide and the Union component setup."
    case .motionTrack: "Requires a motion guide and the Motion Track component setup."
    case .crossview: "Requires a source movie and CrossView guide with the CrossView component setup."
  } }
}
