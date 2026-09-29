import Foundation
import NNC
final class StageTimer {
  static var last:UInt64=0
  static var iteration=0
  static func begin(_ i:Int) { iteration=i;last=DispatchTime.now().uptimeNanoseconds }
  static func mark(_ name:String) {
    let now=DispatchTime.now().uptimeNanoseconds
    let row:[String:Any] = ["stage":name,"iteration":iteration,"seconds":Double(now-last)/1e9]
    last=now
    let json=try! JSONSerialization.data(withJSONObject:row,options:.sortedKeys)
    FileHandle.standardError.write(json);FileHandle.standardError.write(Data([10]))
  }
}
