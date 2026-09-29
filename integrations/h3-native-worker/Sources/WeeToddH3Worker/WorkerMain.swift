// Independently implemented WeeTodd H3 inference; NNC/ccv are BSD dependencies.
import Foundation
import Darwin
import NNC

@main struct H3WorkerMain {
  static func main() {
    do {
      let args=CommandLine.arguments
      if args.count==2,args[1]=="--capabilities" {
        print("{\"protocol\":1,\"backend\":\"nnc_experimental\",\"blocks\":50,\"max_rows\":40000,\"parent_watchdog\":true}")
        return
      }
      guard args.count==7,args[1]=="serve",args[6]=="1" else {
        throw ProbeError.invalid("Usage: WeeToddH3Worker serve INPUTS CHECKPOINT LORA WORKSPACE 1")
      }
      // A caller's debug environment must not change qualified math or residency.
      for key in ProcessInfo.processInfo.environment.keys where key.hasPrefix("WEETODD_NNC_") {
        unsetenv(key)
      }
      let settings=["PRECISION":"fp16","WEIGHT_STORAGE":"dense","ANE":"disabled",
        "FFN_POLICY":"layer-scaled","ATTENTION_SCALE":"input","PROGRESS":"1",
        "BLOCK_START":"0","BLOCK_COUNT":"50","ATTENTION_ORDER":"paired",
        "ATTENTION_ACCUMULATION":"fp32","PROJECTIONS":"input-scaled","RESIDENCY":"block",
        "PREFETCH":"1","BUFFER_IO":"bounded","QKV_SCHEDULE":"serial"]
      for (key,value) in settings { setenv("WEETODD_NNC_"+key,value,1) }
      DynamicGraph.flags.insert(.disableMFAAppleNeuralEngine)
      let parent=getppid()
      guard parent>1 else { throw ProbeError.invalid("Native worker requires a live parent") }
      // EOF cannot interrupt a GPU evaluation. Also exit if the owning renderer dies.
      let watchdog=DispatchSource.makeTimerSource(queue:DispatchQueue.global(qos:.utility))
      watchdog.schedule(deadline:.now()+1,repeating:1)
      watchdog.setEventHandler { if getppid() != parent { exit(74) } }
      watchdog.resume()
      defer { watchdog.cancel() }
      try withExtendedLifetime(watchdog) {
        try runStack(fixture:args[2],checkpoint:args[3],adapter:args[4],output:args[5],iterations:1,serving:true)
      }
    } catch {
      FileHandle.standardError.write(Data("\(error)\n".utf8));exit(1)
    }
  }
}
