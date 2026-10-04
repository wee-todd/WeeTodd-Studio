import CoreFoundation
import Foundation

enum NativeH3LoRAComposition {
  struct Result { let pairs:[[Any]];let descriptors:[String:Any]?;let schedulePoints:Int }
  private struct Item { let path:String;let strength:Double;let settings:H3LoRASettings;let explicit:Bool }
  private static func canonical(_ path:String) throws -> String {
    guard path.hasPrefix("/"),path.utf8.count<=4096,!path.utf8.contains(0) else { throw StudioError.invalid("H3 LoRAs require bounded absolute local files.") }
    return URL(fileURLWithPath:path).standardizedFileURL.resolvingSymlinksInPath().path
  }
  private static func number(_ value:Any?) throws -> Double {
    guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,(-10...10).contains(n.doubleValue) else { throw StudioError.invalid("H3 LoRA strength must be finite from −10 to 10.") };return n.doubleValue
  }
  static func supportsProfileStack(_ raw:Any?) -> Bool {
    guard let root=raw as? [String:Any],Set(root.keys)==["version","adapters"],
      let version=root["version"] as? NSNumber,CFGetTypeID(version) != CFBooleanGetTypeID(),version.doubleValue==1,
      let adapters=root["adapters"] as? [[String:Any]],(1...8).contains(adapters.count) else { return false }
    var paths=Set<String>()
    for item in adapters {
      guard Set(item.keys).isSubset(of:["path","strength","profile","qkv_layout","start_after_evaluations","adaln_input_grid"]),
        let path=item["path"] as? String,let canonical=try? canonical(path),paths.insert(canonical).inserted,
        (try? number(item["strength"])) != nil,item["adaln_input_grid"]==nil || item["adaln_input_grid"] is NSNull,
        item["profile"]==nil || item["profile"] is String,item["qkv_layout"]==nil || item["qkv_layout"] is String,
        H3LoRAProfile(rawValue:item["profile"] as? String ?? "auto") != nil,
        H3LoRAQKVLayout(rawValue:item["qkv_layout"] as? String ?? "auto") != nil else { return false }
      if let value=item["start_after_evaluations"] {
        guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,
          n.doubleValue.rounded()==n.doubleValue,(0...99).contains(n.doubleValue) else { return false }
      }
    }
    return true
  }
  static func compose(componentPairs:[[Any]],rootStack:Any?,attachments:[Attachment],assets:[MediaAsset],
    schedulePoints:Int,explicitEvaluations:Int?,samplingMethod:String) throws -> Result {
    var items:[Item]=[]
    for pair in componentPairs {
      guard pair.count==2,let path=pair[0] as? String else { throw StudioError.invalid("Invalid H3 profile LoRA pair.") }
      items.append(Item(path:try canonical(path),strength:try number(pair[1]),settings:.init(),explicit:false))
    }
    if let raw=rootStack {
      guard items.isEmpty,let object=raw as? [String:Any],let adapters=object["adapters"] as? [[String:Any]],(1...8).contains(adapters.count) else { throw StudioError.invalid("Profile H3 descriptors cannot combine component pairs.") }
      let legacy=Set(object.keys)==["adapters"]
      if !legacy {
        guard Set(object.keys)==["version","adapters"],let version=object["version"] as? NSNumber,
          CFGetTypeID(version) != CFBooleanGetTypeID(),version.doubleValue==1 else { throw StudioError.invalid("H3 descriptor stack must use version 1.") }
      }
      for adapter in adapters {
        guard Set(adapter.keys).isSubset(of:["path","strength","profile","qkv_layout","start_after_evaluations","adaln_input_grid"]),
          let path=adapter["path"] as? String,adapter["profile"]==nil || adapter["profile"] is String,
          adapter["qkv_layout"]==nil || adapter["qkv_layout"] is String,adapter["adaln_input_grid"]==nil || adapter["adaln_input_grid"] is NSNull,
          let profile=H3LoRAProfile(rawValue:adapter["profile"] as? String ?? "auto"),
          let layout=H3LoRAQKVLayout(rawValue:adapter["qkv_layout"] as? String ?? "auto") else { throw StudioError.invalid("Unsupported H3 adapter descriptor or AdaLN grid.") }
        var start=0
        if let value=adapter["start_after_evaluations"] {
          guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,
            n.doubleValue.rounded()==n.doubleValue,(0...99).contains(n.doubleValue) else { throw StudioError.invalid("Invalid H3 deferred activation.") };start=n.intValue
        }
        guard !legacy || (adapters.count==1 && profile == .turbo && layout == .contiguousQKV && start==0) else { throw StudioError.invalid("Unsupported historical H3 adapter stack.") }
        items.append(Item(path:try canonical(path),strength:try number(adapter["strength"]),settings:.init(profile:profile,qkvLayout:layout,startAfterEvaluations:start),explicit:!legacy))
      }
    }
    for attachment in attachments where attachment.role == .lora && attachment.isEnabled {
      guard let asset=assets.last(where:{$0.id==attachment.assetID}),asset.kind == .lora,asset.loraModel == .h3,
        asset.loraAdalnInputGrid==nil else { throw StudioError.invalid("Choose an H3 LoRA without an unsupported AdaLN grid.") }
      var settings=attachment.h3LoRA ?? H3LoRASettings()
      if settings.profile == .auto,let text=asset.loraProfile,let profile=H3LoRAProfile(rawValue:text) { settings.profile=profile }
      if settings.qkvLayout == .auto,let text=asset.loraLayout,let layout=H3LoRAQKVLayout(rawValue:text) {
        settings.qkvLayout=layout
      }
      guard asset.loraProfile==nil || H3LoRAProfile(rawValue:asset.loraProfile!) != nil,
        asset.loraLayout==nil || ["contiguous_qkv","native_interleaved"].contains(asset.loraLayout!) else { throw StudioError.invalid("Unsupported H3 library adapter profile or QKV layout.") }
      items.append(Item(path:try canonical(asset.path),strength:attachment.strength,settings:settings,explicit:attachment.h3LoRA != nil || settings.qkvLayout == .nativeInterleaved))
    }
    guard items.count<=8,Set(items.map(\.path)).count==items.count else { throw StudioError.invalid("Choose at most eight distinct H3 LoRAs, including profile adapters.") }
    var hasTurbo=false
    for item in items {
      let metadata=try NativeLoRAInspection.inspect(URL(fileURLWithPath:item.path),modelHint:.h3,
        selectedH3Profile:item.settings.profile == .auto ? nil : item.settings.profile.rawValue)
      hasTurbo=hasTurbo || item.settings.profile == .turbo || metadata["loraProfile"] as? String == "turbo"
    }
    let points=hasTurbo && explicitEvaluations==nil ? 5 : schedulePoints
    guard !hasTurbo || (points==5 && samplingMethod=="euler") else { throw StudioError.invalid("H3 Turbo requires Euler and 4 evaluations (5 schedule points).") }
    for item in items {
      try item.settings.validate(strength:item.strength,evaluations:points-1,samplingMethod:samplingMethod)
      let metadata=try NativeLoRAInspection.validateH3Sampling(path:item.path,
        selectedProfile:item.settings.profile == .auto ? nil : item.settings.profile.rawValue,schedulePoints:points)
      guard metadata["loraModel"] as? String=="h3",metadata["loraRequiresAdalnGrid"] as? Bool != true else { throw StudioError.invalid("H3 LoRA model or AdaLN grid is incompatible.") }
      if metadata["loraProfile"] as? String == "turbo",item.settings.startAfterEvaluations != 0 { throw StudioError.invalid("Header-declared H3 Turbo requires immediate activation.") }
    }
    let explicit=items.contains(where:{$0.explicit})
    return Result(pairs:items.map {[$0.path,$0.strength]},descriptors:explicit ? ["version":1,"adapters":items.map {$0.settings.wire(path:$0.path,strength:$0.strength)}] : nil,schedulePoints:points)
  }
}
