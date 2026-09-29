import Foundation
import Darwin
import LTX25MLX

@main struct PipelineProbe {
  static func main() async {
    // Install before starting work; cancelling the task lets stage and mux
    // cleanup run before the process exits. Signal callbacks do no model work.
    signal(SIGINT,SIG_IGN); signal(SIGTERM,SIG_IGN)
    let job=Task.detached { try execute() }
    let sources=[SIGINT,SIGTERM].map { number in
      let source=DispatchSource.makeSignalSource(signal:number,queue:.global())
      source.setEventHandler { @Sendable in job.cancel() }
      source.resume()
      return source
    }
    do { try await job.value }
    catch {
      try? FileHandle.standardError.write(contentsOf:Data("\(error)\n".utf8))
      for source in sources { source.cancel() }
      exit(error is CancellationError ? 130 : 1)
    }
    for source in sources { source.cancel() }
  }
  private static func execute() throws {
      try Task.checkCancellation()
      var args=Array(CommandLine.arguments.dropFirst())
      var videoActivationBytes=512*1024*1024
      var transformerActivationBytes=2*1024*1024*1024
      var videoBackend=MLXVideoBackend.mps,saveLatents=false,saveFrames=false
      var audioBackend=MLXAudioBackend.mps
      while !args.isEmpty {
        if args.last == "--save-latents" { saveLatents=true;args.removeLast();continue }
        if args.last == "--save-frames" { saveFrames=true;args.removeLast();continue }
        guard args.count>=2,["--video-activation-mib","--transformer-activation-mib","--video-decoder","--audio-decoder"].contains(args[args.count-2]) else { break }
        let option=args[args.count-2]
        if option == "--audio-decoder" {
          guard let backend=MLXAudioBackend(rawValue:args.last!) else {
            throw NSError(domain:"WeeToddMLX",code:1,userInfo:[NSLocalizedDescriptionKey:"Audio decoder must be mps or mlx."])
          }
          audioBackend=backend;args.removeLast(2);continue
        }
        if option == "--video-decoder" {
          guard let backend=MLXVideoBackend(rawValue:args.last!) else {
            throw NSError(domain:"WeeToddMLX",code:1,userInfo:[NSLocalizedDescriptionKey:"Video decoder must be mps or mlx."])
          }
          videoBackend=backend;args.removeLast(2);continue
        }
        guard let mib=Int(args.last!), (1...MLXMediaPipeline.maximumVideoActivationMiB).contains(mib) else {
          throw NSError(domain:"WeeToddMLX",code:1,userInfo:[NSLocalizedDescriptionKey:"Decoder workspace must be 1...\(MLXMediaPipeline.maximumVideoActivationMiB) MiB."])
        }
        if option == "--video-activation-mib" { videoActivationBytes=mib*1024*1024 }
        else { transformerActivationBytes=mib*1024*1024 }
        args.removeLast(2)
      }
      guard (args.count == 2 && args[0] == "preflight") || (args.count == 3 && args[0] == "render") else {
        throw NSError(domain:"WeeToddMLX",code:1,userInfo:[NSLocalizedDescriptionKey:"Usage: WeeToddMLXPipelineProbe preflight REQUEST | render REQUEST FFMPEG [--transformer-activation-mib MiB] [--video-activation-mib 1...\(MLXMediaPipeline.maximumVideoActivationMiB)] [--video-decoder mps|mlx] [--audio-decoder mps|mlx] [--save-latents]"])
      }
      let request=try MLXDistilledRequest.load(URL(fileURLWithPath:args[1]))
      let pipeline=try MLXMediaPipeline(request:request,videoActivationBytes:videoActivationBytes,transformerActivationBytes:transformerActivationBytes,videoBackend:videoBackend,audioBackend:audioBackend,saveLatents:saveLatents,saveFrames:saveFrames)
      if args[0] == "preflight" { print("Preflight passed; no weighted inference executed."); return }
      let output=try pipeline.run(ffmpeg:URL(fileURLWithPath:args[2])) { stage,completed,total in
        let data=try JSONSerialization.data(withJSONObject:["stage":stage,"completed":completed,"total":total])
        try FileHandle.standardOutput.write(contentsOf:data+Data([10]))
      }
      try FileHandle.standardOutput.write(contentsOf:JSONSerialization.data(withJSONObject:["output":output.path])+Data([10]))

  }
}
