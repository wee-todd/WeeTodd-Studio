import Foundation
import NNC
import Darwin
// Opt-in diagnostic only. Preserve one CPU block state to reproduce the first overflow.
enum RangeTrace {
 static var previous: Data?
 static var previousName: String?
 static func inspect(_ value:Model.IO,name:String,retain:Bool=false) -> Model.IO {
  guard let directory=ProcessInfo.processInfo.environment["WEETODD_NNC_RANGE_TRACE"] else { return value }
  return value.debug(name:name) { tensors,stream in
   stream?.joined()
   guard let raw=tensors.first,let raw=raw else { return }
   let cpu=Tensor<Float>(from:raw).toCPU()
   cpu.withUnsafeBytes { bytes in
    let values=bytes.bindMemory(to:Float.self)
    var maximum:Float=0;var bad=0
    for v in values { if !v.isFinite { bad += 1 } else { maximum=max(maximum,abs(v)) } }
    let row:[String:Any]=["range_stage":name,"max_abs":maximum,"nonfinite":bad]
    let record=try! JSONSerialization.data(withJSONObject:row,options:.sortedKeys)
    FileHandle.standardError.write(record);FileHandle.standardError.write(Data([10]))
    if bad>0 {
     let folder=URL(fileURLWithPath:directory)
     do {
      try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
      if let previous=previous,let previousName=previousName { try previous.write(to:folder.appendingPathComponent(previousName+".f32"),options:.atomic) }
      try record.write(to:folder.appendingPathComponent("first-nonfinite.json"),options:.atomic)
     } catch { FileHandle.standardError.write(Data("Range trace write failed: \(error)\n".utf8)) }
     exit(2)
    }
    if retain { previous=Data(bytes);previousName=name }
   }
  }
 }
}
