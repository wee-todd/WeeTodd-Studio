import Foundation
import Darwin
final class BoundedLineReader {
 private var buffered=Data()
 func next() throws -> String? {
  let limit=65536
  while true {
   if let newline=buffered.firstIndex(of:10) {
    let count=buffered.distance(from:buffered.startIndex,to:newline)
    guard count<=limit,let line=String(data:buffered.prefix(count),encoding:.utf8) else { throw ProbeError.invalid("Invalid worker command encoding or length") }
    buffered.removeFirst(count+1);return line
   }
   guard buffered.count<=limit else { throw ProbeError.invalid("Worker command exceeds 64KB") }
   var bytes=[UInt8](repeating:0,count:min(4096,limit+1-buffered.count))
   let count=Darwin.read(STDIN_FILENO,&bytes,bytes.count)
   if count<0 {
    if errno==EINTR { continue }
    throw ProbeError.invalid("Worker pipe read failed")
   }
   if count==0 {
    guard buffered.isEmpty else { throw ProbeError.invalid("Unterminated worker command") }
    return nil
   }
   buffered.append(contentsOf:bytes.prefix(count))
  }
 }
}
