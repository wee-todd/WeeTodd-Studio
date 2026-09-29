import Foundation
struct ModulationSpans {
 let ranges:[Range<Int>]
 let count:Int
 let tableRows:Int
 init(indices:[Int32],tableRows:Int) throws {
  guard !indices.isEmpty,tableRows>0,indices.allSatisfy({$0>=0 && Int($0)<tableRows}) else { throw ProbeError.invalid("Invalid modulation span indices") }
  var ranges:[Range<Int>]=[];var start=0
  for index in 1..<indices.count where indices[index] != indices[index-1] { ranges.append(start..<index);start=index }
  ranges.append(start..<indices.count)
  guard ranges.count<=256 else { throw ProbeError.invalid("Modulation layout too fragmented") }
  self.ranges=ranges;self.count=indices.count;self.tableRows=tableRows
 }
 func validate(_ indices:[Int32]) throws {
  guard indices.count==count,indices.allSatisfy({$0>=0 && Int($0)<tableRows}),ranges.allSatisfy({span in indices[span].allSatisfy{$0==indices[span.lowerBound]}}) else { throw ProbeError.invalid("Worker modulation span boundary changed") }
 }
 static func check(input:String,output:String) throws {
  guard let value=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:input))) as? [String:Any],let indices=value["indices"] as? [Int32],let updated=value["updated"] as? [Int32],let rows=value["tableRows"] as? Int else { throw ProbeError.invalid("Invalid span fixture") }
  let layout=try ModulationSpans(indices:indices,tableRows:rows);try layout.validate(updated)
  let result:[String:Any]=["spans":layout.ranges.map{[$0.lowerBound,$0.upperBound]},"valid":true]
  try JSONSerialization.data(withJSONObject:result).write(to:URL(fileURLWithPath:output))
 }
}
