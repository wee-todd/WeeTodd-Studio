import Foundation
import Darwin
import CryptoKit
import MLX
import TensorIO
import LTX25Audio
import LTX25Engine
import LTX25MLX
import InferenceMedia

enum AudioDecodeProbe {
  static func run(_ args:[String],mib:Int) throws {
    guard (256...8192).contains(mib) else { throw LTXError.invalid("Audio workspace must be 256...8192 MiB.") }
    let report=URL(fileURLWithPath:args[3]),stem=URL(fileURLWithPath:args[3]).deletingPathExtension()
    let raw=stem.appendingPathExtension("f32"),wav=stem.appendingPathExtension("wav")
    for url in [report,raw,wav] { guard !FileManager.default.fileExists(atPath:url.path) else { throw LTXError.invalid("Audio probe output already exists: \(url.path)") } }
    let file=try SafeTensorFile(url:URL(fileURLWithPath:args[2]))
    guard let d=file.tensors["audio"],d.shape.count==3,d.shape[0]==8,d.shape[2]==16,(1...501).contains(d.shape[1]) else {
      throw LTXError.invalid("Expected audio latent [8, 1...501, 16].")
    }
    let frames=Int(d.shape[1]),budget=UInt64(mib)*1024*1024
    let estimated=try args[0]=="mlx-audio" ? MLXAudioDecoder.estimatedPeakBytes(latentFrames:frames) : AudioDecoder.estimatedPeakBytes(latentFrames:frames)
    guard estimated<=budget else { throw LTXError.invalid("Audio probe exceeds workspace admission.") }
    let latent=try file.readFloat32(named:"audio")
    Memory.peakMemory=0;let start=Date()
    var marks:[String:Double]=[:]
    func progress(_ stage:String) throws { try Task.checkCancellation();marks[stage]=Date().timeIntervalSince(start) }
    let wave:AudioWaveform
    if args[0]=="mlx-audio" {
      wave=try MLXAudioDecoder(checkpoint:URL(fileURLWithPath:args[1]),maximumResidentBytes:budget)
        .decode(latent:latent,latentFrames:frames,progress:progress)
    } else {
      wave=try AudioDecoder(checkpoint:URL(fileURLWithPath:args[1]),maximumResidentBytes:budget)
        .decode(latent:latent,latentFrames:frames,progress:progress)
    }
    let seconds=Date().timeIntervalSince(start)
    guard wave.frameCount == (try AudioDecoder.sampleCount(latentFrames:frames)),wave.channels==2,wave.sampleRate==48000 else {
      throw LTXError.invalid("Audio probe sample contract mismatch.")
    }
    var info=task_vm_info_data_t(),count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
    let status=withUnsafeMutablePointer(to:&info) { ptr in ptr.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
      task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
    } }
    guard status==KERN_SUCCESS else { throw LTXError.invalid("Cannot measure process memory.") }
    let data=wave.samples.withUnsafeBytes { Data($0) }
    let result:[String:Any]=["backend":args[0],"seconds":seconds,"latent_frames":frames,"samples_per_channel":wave.frameCount,
      "sample_rate":48000,"channels":2,"peak_mlx_bytes":Memory.peakMemory,"peak_process_footprint_bytes":info.ledger_phys_footprint_peak,
      "current_process_footprint_bytes":info.phys_footprint,"workspace_budget_bytes":budget,"estimated_workspace_bytes":estimated,
      "scope":"audio decode only; excludes latent file IO, WAV publication, sampling and video",
      "samples_layout":"channel-major Float32 little-endian","samples_sha256":SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined(),
      "samples_file":raw.path,"wav_file":wav.path,"stage_elapsed_seconds":marks]
    try Task.checkCancellation()
    try data.write(to:raw,options:.withoutOverwriting)
    try MediaOutput.writeWAV(samples:wave.samples,sampleRate:48000,channels:2,to:wav)
    let json=try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys])
    try json.write(to:report,options:.withoutOverwriting)
    try FileHandle.standardOutput.write(contentsOf:json+Data([10]))
  }
}
