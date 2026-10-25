import Foundation

/// Output dimensions always describe the final canvas. Only the draft is rounded.
public struct H3TwoStageSettings: Codable, Equatable {
  public var draftScale: Double = 2
  public var strength: Double = 0.35
  public var evaluations: Int = 3
  public var resizeMethod: H3JointResizeMethod? = nil
  public var useTurboDraft: Bool = true
  public var draftTurboPath: String?
  public var draftTurboTask: String?
  public var learnedUpscalerPath: String?
  public var learnedUpscalerHeaderSHA256: String?
  public init() {}
  public static func turboTask(for task:String) -> String {["ref2va","a2v"].contains(task) ? "ref2va":"fl2va"}
  private enum CodingKeys:String,CodingKey {case draftScale,strength,evaluations,resizeMethod,learnedUpscalerPath,learnedUpscalerHeaderSHA256,useTurboDraft,draftTurboPath,draftTurboTask}
  public init(from decoder:Decoder) throws {
    let c=try decoder.container(keyedBy:CodingKeys.self)
    draftScale=try c.decodeIfPresent(Double.self,forKey:.draftScale) ?? 2
    strength=try c.decodeIfPresent(Double.self,forKey:.strength) ?? 0.35
    evaluations=try c.decodeIfPresent(Int.self,forKey:.evaluations) ?? 3
    // Missing policy belongs to a saved custom recipe, never a new default.
    useTurboDraft=try c.decodeIfPresent(Bool.self,forKey:.useTurboDraft) ?? false
    draftTurboPath=try c.decodeIfPresent(String.self,forKey:.draftTurboPath)
    draftTurboTask=try c.decodeIfPresent(String.self,forKey:.draftTurboTask)
    learnedUpscalerPath=try c.decodeIfPresent(String.self,forKey:.learnedUpscalerPath)
    learnedUpscalerHeaderSHA256=try c.decodeIfPresent(String.self,forKey:.learnedUpscalerHeaderSHA256)
    resizeMethod=try c.decodeIfPresent(H3JointResizeMethod.self,forKey:.resizeMethod)
    if !c.contains(.resizeMethod),learnedUpscalerPath==nil,!useTurboDraft {resizeMethod = .bilinear}
  }

  public func draftCanvas(width: Int, height: Int, durationSeconds: Double) throws -> (width: Int, height: Int) {
    guard (64...1920).contains(width), (64...1920).contains(height),
      width.isMultiple(of:32), height.isMultiple(of:32), min(width,height)<=1088 else {
      throw StudioError.invalid("H3 two-stage final dimensions must be multiples of 32, at least 64 on both axes, with longest edge ≤1920 and shortest edge ≤1088. The final canvas is never rounded automatically.")
    }
    guard durationSeconds.isFinite, (2.5...15).contains(durationSeconds),
      draftScale.isFinite, draftScale>1, draftScale<=2,
      strength.isFinite, strength>0, strength<=1, (1...64).contains(evaluations) else {
      throw StudioError.invalid("H3 two-stage requires a 2.5–15 second duration, draft scale above 1 through 2, strength above 0 through 1, and 1–64 refinement evaluations.")
    }
    var frames=Int((durationSeconds*24).rounded(.toNearestOrEven));while frames%17 != 5 {frames+=1}
    let latentFrames=((frames-5)/17)*5+2
    let audioRows=2*Int((Double(frames)/24*40).rounded(.toNearestOrEven))
    guard latentFrames*(width/32)*(height/32)+audioRows+1<=64_000 else {
      throw StudioError.invalid("This H3 final resolution and duration exceed the 64000-row budget. Reduce the final dimensions or duration; references also consume this budget.")
    }
    let w=Int(ceil(Double(width)/draftScale/32))*32
    let h=Int(ceil(Double(height)/draftScale/32))*32
    guard w<width,h<height,w*h<=1376*768,
      latentFrames*(w/32)*(h/32)+audioRows+1<=40_000 else {
      throw StudioError.invalid("This final canvas has no admitted smaller H3 draft at the selected scale. Increase the draft scale toward 2 or reduce resolution/duration.")
    }
    return (w,h)
  }
}
