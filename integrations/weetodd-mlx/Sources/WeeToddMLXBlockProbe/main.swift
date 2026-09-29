import Foundation
import Darwin
import MLX
import LTX25MLX
import LTX25Engine
import TensorIO

@main struct Probe {
  static func main() {
    do { try run() } catch {
      try? FileHandle.standardError.write(contentsOf:Data("\(error)\n".utf8)); exit(2)
    }
  }
  static func run() throws {
    let a=Array(CommandLine.arguments.dropFirst())
    guard a.count == 6, let index=Int(a[4]), (0..<48).contains(index) else {
      throw LTXError.invalid("Usage: WeeToddMLXBlockProbe CONFIG INPUTS EXPECTED PAGE BLOCK_INDEX REPORT")
    }
    let configURL=URL(fileURLWithPath:a[0])
    guard (try configURL.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? Int.max) <= 16384 else {
      throw LTXError.invalid("Oversized block configuration.")
    }
    let c=try JSONDecoder().decode(AVBlockConfiguration.self,from:Data(contentsOf:configURL))
    let block=try MLXAVBlock(configuration:c)
    let source=try MLXBlockSource(url:URL(fileURLWithPath:a[3]),blockIndex:index,expectedShapes:block.weightShapes)
    let fixture=try SafeTensorFile(url:URL(fileURLWithPath:a[1]))
    let expected=try SafeTensorFile(url:URL(fileURLWithPath:a[2]))
    guard Set(fixture.tensors.keys) == Set(block.inputShapes.keys),
      fixture.tensors.values.reduce(UInt64(0),{ $0+$1.byteCount }) <= 128*1024*1024 else {
      throw LTXError.invalid("Unexpected or oversized input fixture.")
    }
    var inputs: [String:MLXArray]=[:]
    for (name,shape) in block.inputShapes {
      guard fixture.tensors[name]?.shape == shape.map(UInt64.init) else { throw LTXError.invalid("Input shape differs: \(name)") }
      inputs[name]=try MLXWeight.read(fixture,name).asType(.float32)
    }
    eval(Array(inputs.values))
    Memory.cacheLimit=128*1024*1024
    Memory.peakMemory=0
    let begin=Date()
    try block.load { try source.read($0,shape:$1) }
    let loadSeconds=Date().timeIntervalSince(begin)
    var runs: [[String:Any]]=[], passed=true
    for _ in 0..<3 {
      let start=Date(); let result=try block.evaluate(inputs); let seconds=Date().timeIntervalSince(start)
      var outputs: [String:Any]=[:]
      for name in ["video","audio"] {
        let actual=result[name]!.asArray(Float.self), reference=try expected.readFloat32(named:name)
        guard actual.count == reference.count, reference.allSatisfy(\.isFinite) else { throw LTXError.invalid("Invalid reference output.") }
        let maxError=zip(actual,reference).map { abs($0-$1) }.max()!
        let squareError=zip(actual,reference).reduce(0.0) { $0+pow(Double($1.0)-Double($1.1),2) }
        let squareNorm=reference.reduce(0.0) { $0+Double($1)*Double($1) }
        let relative=sqrt(squareError/max(squareNorm,1e-30))
        passed = passed && maxError <= 0.002 && relative <= 0.0001
        outputs[name]=["max_absolute_error":maxError,"relative_l2":relative]
      }
      runs.append(["seconds":seconds,"outputs":outputs,"mlx_active_bytes":Memory.activeMemory,"mlx_peak_bytes":Memory.peakMemory])
    }
    block.release(); Memory.clearCache()
    let report: [String:Any] = ["scope":"one-real-block-packed-q8-swift-mlx","passed":passed,
      "load_seconds":loadSeconds,"packed_weight_bytes":source.storageBytes,"runs":runs,
      "released_mlx_active_bytes":Memory.activeMemory,"released_mlx_cache_bytes":Memory.cacheMemory]
    let data=try JSONSerialization.data(withJSONObject:report,options:[.sortedKeys,.prettyPrinted])
    try data.write(to:URL(fileURLWithPath:a[5]),options:.atomic)
    try FileHandle.standardOutput.write(contentsOf:data+Data([10]))
    guard passed else { throw LTXError.invalid("Installed MLX block numerical comparison failed.") }
  }
}
