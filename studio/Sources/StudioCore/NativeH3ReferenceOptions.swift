import Foundation

extension H3ReferencePlacement {
  public var isEmpty:Bool {
    frame==nil && soundtrackPath==nil && imagePixelBudgetPercent==nil && videoSizePolicy==nil && videoTemporalDensity==nil
  }
  /// Explicit image budget applies to Ref images and A2V keyframes; FL anchors
  /// already use the generated canvas. Video options are Ref2VA only.
  public func mediaOptions(kind:String,task:String) throws -> [String:Any] {
    guard ["ref2va","a2v"].contains(task),["image","video","audio"].contains(kind),
      task != "a2v" || (kind=="image" && frame==nil && soundtrackPath==nil),
      soundtrackPath==nil || kind=="video",imagePixelBudgetPercent==nil || (kind=="image" && (50...400).contains(imagePixelBudgetPercent!)),
      (videoSizePolicy==nil && videoTemporalDensity==nil) || (task=="ref2va" && kind=="video") else {
      throw StudioError.invalid("H3 reference image budget and movie density/size controls do not match this task or media kind. FL anchors, extension and motion cannot use them.")
    }
    var fields:[String:Any]=[:]
    if let budget=imagePixelBudgetPercent { fields["image_pixel_budget_percent"]=budget }
    if videoSizePolicy != nil || videoTemporalDensity != nil {
      // Make the paired explicit-policy defaults visible in the frozen recipe.
      fields["size_policy"]=(videoSizePolicy ?? .matchOutput).rawValue
      fields["temporal_density"]=(videoTemporalDensity ?? .full).rawValue
    }
    return fields
  }
}
