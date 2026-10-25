import Foundation

/// Reusable local component links. Each clip freezes its selected paths.
public struct H3TwoStageModelLinks:Codable,Equatable {
  public var flTurboPath:String?
  public var referenceTurboPath:String?
  public var learnedUpscalerPath:String?
  public var learnedUpscalerHeaderSHA256:String?
  public init() {}
  public mutating func setTurboPath(_ path:String,task:String) {
    if H3TwoStageSettings.turboTask(for:task)=="ref2va" {referenceTurboPath=path} else {flTurboPath=path}
  }
  public func settings(task:String) -> H3TwoStageSettings {
    var value=H3TwoStageSettings()
    value.draftTurboTask=H3TwoStageSettings.turboTask(for:task)
    value.draftTurboPath=value.draftTurboTask == "ref2va" ? referenceTurboPath:flTurboPath
    value.learnedUpscalerPath=learnedUpscalerPath
    value.learnedUpscalerHeaderSHA256=learnedUpscalerHeaderSHA256
    return value
  }
}
