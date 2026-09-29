import Foundation
import Darwin
import CryptoKit
import MLX
import LTX25MLX
import LTX25Engine
import AdapterRuntime

/// Real weights and fixed synthetic activations. Not a media-generation route.
@main struct Probe {
  struct Options:Decodable {
    var maximumActivationMiB:Int?
    var cacheMiB:Int?
    var compileGraph:Bool?
    var perTokenVideo:Bool?
    var nativeLoading:Bool?
    var batchLoading:Bool?
  }
  static func memory() throws -> [String:UInt64] {
    var info=task_vm_info_data_t()
    var count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
    let status=withUnsafeMutablePointer(to:&info) { pointer in
      pointer.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
        task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
      }
    }
    guard status == KERN_SUCCESS else { throw LTXError.invalid("Cannot read task memory: \(status)") }
    return ["physical_footprint_bytes":info.phys_footprint,
      "peak_physical_footprint_bytes":UInt64(info.ledger_phys_footprint_peak)]
  }
  static func main() {
    do { try run() } catch {
      try? FileHandle.standardError.write(contentsOf:Data("\(error)\n".utf8)); exit(2)
    }
  }
  static func run() throws {
    let a=Array(CommandLine.arguments.dropFirst())
    guard (4...5).contains(a.count),let count=Int(a[2]),(1...48).contains(count) else {
      throw LTXError.invalid("Usage: WeeToddMLXStackProbe CONFIG PAGED_ROOT BLOCKS REPORT [LORAS.json]")
    }
    let configURL=URL(fileURLWithPath:a[0]),root=URL(fileURLWithPath:a[1])
    guard (try configURL.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? Int.max) <= 16384 else {
      throw LTXError.invalid("Oversized configuration.")
    }
    let configData=try Data(contentsOf:configURL)
    let options=try JSONDecoder().decode(Options.self,from:configData)
    var blockConfiguration=try JSONSerialization.jsonObject(with:configData) as! [String:Any]
    for key in ["maximumActivationMiB","cacheMiB","compileGraph","perTokenVideo","nativeLoading","batchLoading"] { blockConfiguration.removeValue(forKey:key) }
    let c=try JSONDecoder().decode(AVBlockConfiguration.self,from:JSONSerialization.data(withJSONObject:blockConfiguration))
    let activationMiB=options.maximumActivationMiB ?? 2048,cacheMiB=options.cacheMiB ?? 128
    guard (1...32768).contains(activationMiB),(0...1024).contains(cacheMiB) else { throw LTXError.invalid("Invalid probe memory allowance.") }
    let layout=try MLXAVBlock(configuration:c,maximumActivationBytes:activationMiB*1024*1024)
    // Explicit qualification of existing WeeTodd page layout; not model discovery.
    let sources=try (0..<count).map { index in
      try MLXBlockSource(url:root.appendingPathComponent(String(format:"pages/layer-%03d.safetensors",index)),
        blockIndex:index,expectedShapes:layout.weightShapes,nativeLoading:options.nativeLoading ?? false)
    }
    var selections:[LoRAAdapter]=[]
    if a.count == 5 {
      let url=URL(fileURLWithPath:a[4])
      guard (try url.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? Int.max) <= 64*1024 else {
        throw LTXError.invalid("Oversized adapter request.")
      }
      selections=try JSONDecoder().decode([LoRAAdapter].self,from:Data(contentsOf:url))
    }
    let adapterStack=try MLXLoRAStack(adapters:selections)
    var inputs:[String:MLXArray]=[:]
    for (name,shape) in layout.inputShapes {
      let seed=name.utf8.reduce(0) { $0+Int($1) }
      inputs[name]=MLXArray((0..<shape.reduce(1,*)).map { Float(($0*17+seed)%73-36)/128 },shape)
    }
    if options.perTokenVideo == true {
      for name in ["video_modulation","video_av_modulation"] {
        inputs[name]=broadcast(inputs[name]!,to:[c.videoTokens,layout.inputShapes[name]![1]])
      }
    }
    eval(Array(inputs.values)); Memory.clearCache(); Memory.peakMemory=0
    let stack=try MLXAVStack(configuration:c,blockCount:count,cacheBytes:cacheMiB*1024*1024,maximumActivationBytes:activationMiB*1024*1024,compileGraph:options.compileGraph ?? true)
    var runs:[[String:Any]]=[]
    for run in 0..<3 {
      let start=Date()
      var last:MLXAVStack.Progress?
      let output=try stack.evaluate(inputs,weights:{ try sources[$0].read($1,shape:$2,deferEvaluation:options.batchLoading ?? false) },
        adapters:{ try adapterStack.load(block:$0) },progress:{ p in
          last=p
          if p.completedBlocks % 8 == 0 || p.completedBlocks == count {
            let progress:[String:Any] = ["run":run,"completed_blocks":p.completedBlocks,"total_blocks":count,
              "weight_bytes":p.weightBytes,"active_bytes":p.activeBytes,"cache_bytes":p.cacheBytes,
              "load_seconds":p.loadSeconds,"compute_seconds":p.computeSeconds]
            try FileHandle.standardOutput.write(contentsOf:JSONSerialization.data(withJSONObject:progress)+Data([10]))
          }
        })
      let elapsed=Date().timeIntervalSince(start)
      var hashes:[String:String]=[:]
      for name in ["video","audio"] {
        let values=output[name]!.asArray(Float.self)
        hashes[name]=values.withUnsafeBytes { bytes in SHA256.hash(data:Data(bytes)).map { String(format:"%02x",$0) }.joined() }
      }
      runs.append(["run":run,"seconds":elapsed,"load_seconds":last!.loadSeconds,"compute_seconds":last!.computeSeconds,
        "mlx_peak_bytes":Memory.peakMemory,"released_active_bytes":Memory.activeMemory,
        "released_cache_bytes":Memory.cacheMemory,"process":try memory(),"hashes":hashes])
    }
    let report:[String:Any] = ["scope":"packed-q8-streaming-real-weights-synthetic-activations",
      "blocks":count,"video_tokens":c.videoTokens,"audio_tokens":c.audioTokens,"text_tokens":c.textTokens,
      "adapter_count":selections.count,"cache_limit_bytes":cacheMiB*1024*1024,
      "maximum_activation_bytes":activationMiB*1024*1024,"compiled_blocks":options.compileGraph ?? true,
      "per_token_video":options.perTokenVideo ?? false,"native_loading":options.nativeLoading ?? false,
      "batched_weight_evaluation":options.batchLoading ?? false,"runs":runs]
    try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:a[3]),options:.atomic)
  }
}
