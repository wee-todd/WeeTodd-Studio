import Foundation
import LTX25Engine

/// Resolve the same native-size/bounded reference grids as the Python route.
/// `sol_auto` uses the quality grid and 33 frames until Swift has a qualified
/// Sol kernel; it never advertises an unimplemented speed path.
struct MLXMSRReferencePlan {
  let geometry:AVGeometry
  let priority:String
  let role:String
  let strength:Float
  let attentionStrength:Float

  static func ordered(_ references:[MLXMSRReference]) -> [MLXMSRReference] {
    references.filter { $0.role != "background" } + references.filter { $0.role == "background" }
  }

  static func resolve(_ request:MLXDistilledRequest,target:AVGeometry) throws
    -> (references:[MLXMSRReference],plans:[MLXMSRReferencePlan],layout:MLXMSRLayout)? {
    guard let msr=request.msr else { return nil }
    let references=ordered(msr.references)
    let plans=try references.enumerated().map { index,reference in
      try MLXMSRReferencePlan(reference:reference,index:index,
        source:MLXReferenceImage.size(URL(fileURLWithPath:reference.path)),target:target)
    }
    let layout=try MLXMSRLayout(target:target,groups:plans.map {
      MLXMSRLayout.Group(geometry:$0.geometry,strength:$0.strength,
        attentionStrength:$0.attentionStrength)
    })
    return (references,plans,layout)
  }

  init(reference:MLXMSRReference,index:Int,source:(width:Int,height:Int),target:AVGeometry) throws {
    guard (0...4).contains(index),source.width >= 32,source.height >= 32 else {
      throw LTXError.invalid("MSR source is too small or the reference slot is invalid.")
    }
    let priority=reference.priority == "auto"
      ? (reference.role == "background" || index >= 4 ? "background" : index >= 2 ? "supporting" : "primary")
      : reference.priority
    let policy=reference.sizePolicy == "sol_auto" ? "quality" : reference.sizePolicy
    var height=target.height,width=target.width
    if policy != "quality" {
      height=min(height,(source.height/32)*32)
      width=min(width,(source.width/32)*32)
      let budget=policy == "balanced" ? 512*288 : 384*224
      if height*width>budget {
        let ratio=Double(source.width)/Double(source.height)
        var choices:[(Int,Int)]=[]
        for h in 1...(height/32) {
          for w in 1...(width/32) where h*w*1024<=budget {
            choices.append((h*32,w*32))
          }
        }
        guard let best=choices.max(by: { a,b in
          let areaA=a.0*a.1,areaB=b.0*b.1
          if areaA != areaB { return areaA<areaB }
          return abs(log(Double(a.1)/Double(a.0)/ratio)) >
            abs(log(Double(b.1)/Double(b.0)/ratio))
        }) else { throw LTXError.invalid("No bounded MSR reference grid fits.") }
        height=best.0;width=best.1
      }
    }
    if priority != "primary" {
      let shortLimit=priority == "supporting" ? 384 : 288
      let longLimit=priority == "supporting" ? 768 : 512
      let hLimit=height<width ? shortLimit : height>width ? longLimit : shortLimit
      let wLimit=height<width ? longLimit : height>width ? shortLimit : shortLimit
      height=max(32,min(height,hLimit)/32*32)
      width=max(32,min(width,wLimit)/32*32)
    }
    let frames=reference.referenceFrames == "auto" ? 33 : Int(reference.referenceFrames)!
    geometry=try AVGeometry(width:width,height:height,frames:frames,fps:target.fps)
    self.priority=priority;role=reference.role
    strength=reference.strength;attentionStrength=reference.attentionStrength
  }
}
