import Foundation
import Darwin
import CryptoKit
import MLX
import Cmlx
import TensorIO
import LTX25Video
import LTX25MLX

/// Decode-only qualification from frozen latents: no text encoding or sampling.
@main struct DecodeProbe {
  static func main() async {
    signal(SIGINT,SIG_IGN);signal(SIGTERM,SIG_IGN)
    let job=Task.detached { try run() }
    let sources=[SIGINT,SIGTERM].map { number in
      let source=DispatchSource.makeSignalSource(signal:number,queue:.global())
      source.setEventHandler { @Sendable in job.cancel() };source.resume();return source
    }
    do { try await job.value } catch {
      try? FileHandle.standardError.write(contentsOf:Data("\(error)\n".utf8))
      sources.forEach { $0.cancel() };exit(error is CancellationError ? 130 : 1)
    }
    sources.forEach { $0.cancel() }
  }
  static func run() throws {
    let a=Array(CommandLine.arguments.dropFirst())
    guard (5...6).contains(a.count),["mps-video","mlx-video","mlx-video-bf16","mps-audio","mlx-audio"].contains(a[0]),let mib=Int(a[4]),(1...32768).contains(mib),a.count==5 || a[0].hasPrefix("mlx-video") else {
      throw NSError(domain:"WeeToddMLX",code:1,userInfo:[NSLocalizedDescriptionKey:"Usage: WeeToddMLXDecodeProbe mps-video|mlx-video|mlx-video-bf16|mps-audio|mlx-audio CHECKPOINT LATENTS REPORT WORKSPACE_MIB [MLX_VIDEO_CACHE_MIB]"])
    }
    if a[0].hasSuffix("audio") { try AudioDecodeProbe.run(a,mib:mib);return }
    let reportURL=URL(fileURLWithPath:a[3])
    guard !FileManager.default.fileExists(atPath:reportURL.path) else { throw NSError(domain:"WeeToddMLX",code:1,userInfo:[NSLocalizedDescriptionKey:"Report already exists."]) }
    let file=try SafeTensorFile(url:URL(fileURLWithPath:a[2]))
    let key=file.tensors["video"] == nil ? "latent" : "video"
    guard let descriptor=file.tensors[key] else { throw NSError(domain:"WeeToddMLX",code:1,userInfo:[NSLocalizedDescriptionKey:"Missing video latent."]) }
    let shape=descriptor.shape.map(Int.init)
    var c=VideoDecodeConfiguration();c.maximumActivationBytes=mib*1024*1024
    let plan=try VideoDecodePlan(shape:shape,configuration:c)
    let precision:MLXVideoPrecision = a[0]=="mlx-video-bf16" ? .bfloat16 : .float32
    let usesMLX=a[0].hasPrefix("mlx-video")
    var cacheBytes:Int?
    if a.count==6 {
      guard let cache=Int(a[5]),(0...2048).contains(cache) else { throw NSError(domain:"WeeToddMLX",code:1,userInfo:[NSLocalizedDescriptionKey:"Cache must be 0–2048 MiB."]) }
      cacheBytes=cache*1024*1024
    }
    let admitted=try usesMLX ? MLXVideoDecodePlan(shape:shape,configuration:c,precision:precision).admittedActivationBytes : plan.admittedActivationBytes
    let latent=try file.readFloat32(named:key)
    let expected=file.tensors["output"] == nil ? nil : try file.readFloat32(named:"output",maximumBytes:256*1024*1024)
    if let expected,expected.count != plan.outputShape.reduce(1,*) { throw NSError(domain:"WeeToddMLX",code:1,userInfo:[NSLocalizedDescriptionKey:"Reference pixels differ from decoded geometry."]) }
    var count=0,frames=0,maxabs:Float=0,square=0.0,digest=SHA256()
    let trace=ProcessInfo.processInfo.environment["WEETODD_TRACE_VIDEO_MEMORY"] == "1"
    var memoryTrace:[[String:Int]]=[]
    var decoderSeconds:[String:Double]=[:]
    func receive(_ chunk:VideoFrameChunk) throws {
      try Task.checkCancellation()
      guard chunk.startFrame==frames,chunk.frameCount==1 else { throw NSError(domain:"WeeToddMLX",code:1,userInfo:nil) }
      if let expected { for (offset,value) in chunk.rgb.enumerated() {
        let error=value-expected[count+offset];maxabs=max(maxabs,abs(error));square += Double(error*error)
      } }
      chunk.rgb.withUnsafeBytes { digest.update(bufferPointer:$0) }
      frames += 1;count += chunk.rgb.count
    }
    Memory.peakMemory=0;let start=Date()
    if usesMLX {
      let decoder=try MLXVideoDecoder(checkpoint:URL(fileURLWithPath:a[1]),precision:precision,cacheLimitBytes:cacheBytes)
      cacheBytes=decoder.cacheLimitBytes
      try decoder.decode(latent:MLXArray(latent,shape),configuration:c,progress:{ layer,_ in
        if trace { memoryTrace.append(["layer":layer,"active":Memory.activeMemory,"cache":Memory.cacheMemory,"peak":Memory.peakMemory]) }
      },receive:receive)
      decoderSeconds=decoder.stageSeconds
    } else {
      let decoder=try VideoDecoder(checkpoint:URL(fileURLWithPath:a[1]))
      try decoder.decode(latent:latent,shape:shape,configuration:c,receive:receive)
    }
    let seconds=Date().timeIntervalSince(start)
    var info=task_vm_info_data_t(),size=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
    let status=withUnsafeMutablePointer(to:&info) { p in p.withMemoryRebound(to:integer_t.self,capacity:Int(size)) { task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&size) } }
    guard status==KERN_SUCCESS,frames==plan.outputShape[0] else { throw NSError(domain:"WeeToddMLX",code:1,userInfo:nil) }
    var report:[String:Any]=["backend":a[0],"seconds":seconds,"frames":frames,"pixels":count,"peak_process_footprint_bytes":info.ledger_phys_footprint_peak,"current_process_footprint_bytes":info.phys_footprint,"peak_mlx_bytes":Memory.peakMemory,"pixel_sha256":digest.finalize().map { String(format:"%02x",$0) }.joined(),"scope":"decode and streaming pixel check; no PNG encoding, sampler or text; optional oracle buffer resident","activation_budget_bytes":c.maximumActivationBytes]
    if expected != nil { report["maxabs"]=maxabs;report["rmse"]=sqrt(square/Double(count)) }
    report["admitted_activation_bytes"]=admitted
    report["video_precision"]=precision.rawValue
    report["video_cache_limit_bytes"]=cacheBytes
    report["decoder_seconds"]=decoderSeconds
    var coreVersion=mlx_string_new();defer { mlx_string_free(coreVersion) }
    guard mlx_version(&coreVersion)==0 else { throw NSError(domain:"WeeToddMLX",code:1,userInfo:[NSLocalizedDescriptionKey:"Could not read MLX core version."]) }
    report["mlx_core_version"]=String(cString:mlx_string_data(coreVersion))
    if trace { report["memory_trace"]=memoryTrace }
    let data=try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys])
    try data.write(to:reportURL,options:.withoutOverwriting)
    try FileHandle.standardOutput.write(contentsOf:data+Data([10]))
  }
}
