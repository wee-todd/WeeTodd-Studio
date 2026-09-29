import Foundation
import Darwin
import MLX
import LTX25MLX
import LTX25Engine
import TensorIO

/// Installed-weight numerical qualification. No text/media-generation claim.
@main struct Probe {
  struct Recipe:Decodable { let sigmas:[Double]; let eta:Double }
  static func main() {
    do { try run() } catch {
      try? FileHandle.standardError.write(contentsOf:Data("\(error)\n".utf8)); exit(2)
    }
  }
  static func json<T:Decodable>(_ path:String,_ type:T.Type) throws -> T {
    let url=URL(fileURLWithPath:path)
    guard (try url.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? Int.max) <= 16384 else { throw LTXError.invalid("Oversized configuration.") }
    return try JSONDecoder().decode(type,from:Data(contentsOf:url))
  }
  static func fixture(_ path:String) throws -> SafeTensorFile {
    let file=try SafeTensorFile(url:URL(fileURLWithPath:path))
    guard file.tensors.values.allSatisfy({ $0.dtype == "F32" }),
      file.tensors.values.reduce(UInt64(0),{ $0+$1.byteCount }) <= 128*1024*1024 else {
      throw LTXError.invalid("Fixture requires bounded Float32 tensors.")
    }
    return file
  }
  static func memory() throws -> [String:UInt64] {
    var info=task_vm_info_data_t()
    var count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
    let status=withUnsafeMutablePointer(to:&info) { pointer in
      pointer.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
        task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
      }
    }
    guard status == KERN_SUCCESS else { throw LTXError.invalid("Cannot read process footprint.") }
    return ["physical_footprint_bytes":info.phys_footprint,"peak_physical_footprint_bytes":UInt64(info.ledger_phys_footprint_peak)]
  }
  static func sameConfiguration(_ a:AVBlockConfiguration,_ b:AVBlockConfiguration) -> Bool {
    [a.videoDimension,a.audioDimension,a.heads,a.videoHeadDimension,a.audioHeadDimension,a.videoTokens,a.audioTokens,a.textTokens] ==
      [b.videoDimension,b.audioDimension,b.heads,b.videoHeadDimension,b.audioHeadDimension,b.videoTokens,b.audioTokens,b.textTokens]
  }
  static func compare(_ actual:[Float],_ reference:[Float],cpu:[Float]?,name:String,first:[Float]?) throws -> [String:Any] {
    guard !actual.isEmpty, actual.count == reference.count, cpu == nil || cpu!.count == reference.count,
      reference.allSatisfy(\.isFinite), actual.allSatisfy(\.isFinite), cpu?.allSatisfy(\.isFinite) ?? true else {
      throw LTXError.invalid("Invalid numerical comparison tensors.")
    }
    func error(_ a:[Float],_ b:[Float]) -> (Double,Double) {
      let absolute=zip(a,b).map { abs(Double($0)-Double($1)) }.max()!
      let squared=zip(a,b).reduce(0.0) { $0+pow(Double($1.0)-Double($1.1),2) }
      let norm=b.reduce(0.0) { $0+Double($1)*Double($1) }
      return (absolute,sqrt(squared/max(norm,1e-30)))
    }
    let (absolute,relative)=error(actual,reference)
    let (spreadAbsolute,spreadRelative)=error(cpu ?? reference,reference)
    let repeated=error(actual,first ?? actual).0
    let floor:Double=name == "video" ? 0.002 : 0.0001
    let absoluteLimit=max(floor,2*spreadAbsolute), relativeLimit=max(floor,2*spreadRelative)
    return ["absolute":absolute,"relative_l2":relative,"repeat_absolute":repeated,
      "reference_spread_absolute":spreadAbsolute,"reference_spread_relative":spreadRelative,
      "absolute_limit":absoluteLimit,"relative_limit":relativeLimit,
      "strict_component_passed":absolute <= floor && relative <= floor,
      "passed":absolute <= absoluteLimit && relative <= relativeLimit && repeated <= 0.000001]
  }
  static func run() throws {
    let args=Array(CommandLine.arguments.dropFirst())
    guard (5...6).contains(args.count), ["denoiser","sampling"].contains(args[0]),
      args.count == (args[0] == "sampling" ? 6 : 5),
      let repeats=Int(args[3]), (1...3).contains(repeats) else {
      throw LTXError.invalid("Usage: WeeToddMLXDenoiserProbe denoiser|sampling FIXTURE_PREFIX PAGED_ROOT REPEATS REPORT [CPU_FIXTURE_PREFIX (required for sampling)]")
    }
    let sampling=args[0] == "sampling", prefix=args[1]
    let c=try json(prefix+".config.json",AVBlockConfiguration.self)
    let inputsFile=try fixture(prefix+".inputs.safetensors")
    let expectedFile=try fixture(prefix+".expected.safetensors")
    let shapes=DenoiserLayout.inputShapes(c)
    guard Set(inputsFile.tensors.keys) == Set(shapes.keys), Set(expectedFile.tensors.keys) == ["video","audio"] else {
      throw LTXError.invalid("Unexpected fixture keys.")
    }
    let weights=try MLXDenoiserWeights(root:URL(fileURLWithPath:args[2]),configuration:c)
    var inputs:[String:MLXArray]=[:], expected:[String:[Float]]=[:]
    for (name,shape) in shapes {
      guard inputsFile.tensors[name]?.shape == shape.map(UInt64.init) else { throw LTXError.invalid("Input shape mismatch.") }
      inputs[name]=try MLXWeight.read(inputsFile,name)
    }
    for name in ["video","audio"] {
      guard expectedFile.tensors[name]?.shape == shapes[name+"_latent"]!.map(UInt64.init) else { throw LTXError.invalid("Output shape mismatch.") }
      expected[name]=try expectedFile.readFloat32(named:name)
      guard expected[name]!.allSatisfy(\.isFinite) else { throw LTXError.invalid("Nonfinite reference.") }
    }
    let recipe=sampling ? try json(prefix+".schedule.json",Recipe.self) : nil
    let schedule=try recipe.map { try SamplingSchedule(sigmas:$0.sigmas,eta:$0.eta) }
    let noiseFile=schedule?.steps.contains(where:\.ancestral) == true ? try fixture(prefix+".noise.safetensors") : nil
    let gpuSteps=sampling ? try fixture(prefix+".steps.safetensors") : nil
    let cpuSteps=sampling ? try fixture(args[5]+".steps.safetensors") : nil
    let cpuExpected=sampling ? try fixture(args[5]+".expected.safetensors") : nil
    if sampling {
      let other=try json(args[5]+".schedule.json",Recipe.self)
      guard recipe!.sigmas == other.sigmas, recipe!.eta == other.eta,
        try sameConfiguration(c,json(args[5]+".config.json",AVBlockConfiguration.self)) else {
        throw LTXError.invalid("Independent reference contracts differ.")
      }
      // Before using the existing CPU/GPU calibration, establish that both
      // references evaluated exactly the same input tensors and explicit noise.
      func identical(_ a:SafeTensorFile,_ b:SafeTensorFile) throws {
        guard Set(a.tensors.keys) == Set(b.tensors.keys) else { throw LTXError.invalid("Reference keys differ.") }
        for name in a.tensors.keys {
          guard a.tensors[name]!.shape == b.tensors[name]!.shape,
            try a.readFloat32(named:name) == b.readFloat32(named:name) else { throw LTXError.invalid("Reference tensors differ.") }
        }
      }
      try identical(inputsFile,fixture(args[5]+".inputs.safetensors"))
      if let noiseFile { try identical(noiseFile,fixture(args[5]+".noise.safetensors")) }
      let keys=Set(schedule!.steps.indices.flatMap { ["\($0).video","\($0).audio"] })
      guard Set(gpuSteps!.tensors.keys) == keys, Set(cpuSteps!.tensors.keys) == keys,
        Set(cpuExpected!.tensors.keys) == ["video","audio"] else { throw LTXError.invalid("Reference step keys differ.") }
      for step in schedule!.steps.indices {
        for name in ["video","audio"] {
          let key="\(step).\(name)", shape=shapes[name+"_latent"]!.map(UInt64.init)
          guard gpuSteps!.tensors[key]!.shape == shape, cpuSteps!.tensors[key]!.shape == shape else { throw LTXError.invalid("Reference step shape differs.") }
          if step == schedule!.steps.count-1 {
            guard cpuExpected!.tensors[name]?.shape == shape,
              try gpuSteps!.readFloat32(named:key) == expected[name]!,
              try cpuSteps!.readFloat32(named:key) == cpuExpected!.readFloat32(named:name) else {
              throw LTXError.invalid("Final reference differs from its trajectory.")
            }
          }
        }
      }
    }
    guard let sigma=Float(expectedFile.metadata["sigma"] ?? "0.731"), sigma.isFinite, (0...1).contains(sigma) else {
      throw LTXError.invalid("Invalid fixture sigma.")
    }
    let denoiser=try MLXDenoiser(configuration:c), sampler=try MLXSamplingRunner(configuration:c)
    eval(Array(inputs.values)); Memory.clearCache(); Memory.peakMemory=0
    var runs:[[String:Any]]=[], first:[String:[Float]]=[:], passed=true
    for index in 0..<repeats {
      let start=Date()
      var stepErrors:[[String:Any]]=[]
      let report:(Int,MLXDenoiser.Progress) throws -> Void = { step,event in
        if event.stage != "transformer" || event.completedBlocks % 8 == 0 {
          let data:[String:Any]=["run":index,"step":step,"stage":event.stage,"blocks":event.completedBlocks,
            "active_bytes":event.activeBytes,"elapsed_seconds":Date().timeIntervalSince(start)]
          try FileHandle.standardOutput.write(contentsOf:JSONSerialization.data(withJSONObject:data)+Data([10]))
        }
      }
      let result:[String:MLXArray]
      if let schedule {
        result=try sampler.evaluate(inputs,schedule:schedule,fixedWeights:weights.readFixed,blockWeights:weights.readBlock,
          fixedAdapters:weights.fixedAdapters,blockAdapters:weights.blockAdapters,
          noise:{ step,name,shape in
            guard let noiseFile, noiseFile.tensors["\(step).\(name)"]?.shape == shape.map(UInt64.init) else { throw LTXError.invalid("Missing fixture noise.") }
            return try MLXWeight.read(noiseFile,"\(step).\(name)")
          },stageProgress:report,preview:{ values,event in
            for name in ["video","audio"] {
              let key="\(event.completedSteps-1).\(name)"
              let metrics=try compare(values[name]!.asArray(Float.self),gpuSteps!.readFloat32(named:key),
                cpu:cpuSteps!.readFloat32(named:key),name:name,first:first[key])
              passed = passed && (metrics["passed"] as! Bool)
              stepErrors.append(metrics.merging(["step":event.completedSteps,"modality":name]) { _,new in new })
              if index == 0 { first[key]=values[name]!.asArray(Float.self) }
            }
          })
      } else {
        result=try denoiser.evaluate(inputs,sigma:sigma,fixedWeights:weights.readFixed,blockWeights:weights.readBlock,
          fixedAdapters:weights.fixedAdapters,blockAdapters:weights.blockAdapters,progress:{ try report(0,$0) })
      }
      let seconds=Date().timeIntervalSince(start)
      var errors:[String:Any]=[:]
      for name in ["video","audio"] {
        let actual=result[name]!.asArray(Float.self), reference=expected[name]!
        let metrics=try compare(actual,reference,cpu:cpuExpected?.readFloat32(named:name),name:name,first:first[name])
        passed = passed && (metrics["passed"] as! Bool)
        errors[name]=metrics
        if index == 0 { first[name]=actual }
      }
      runs.append(["seconds":seconds,"errors":errors,"steps":stepErrors,"memory":try memory(),"mlx_peak_bytes":Memory.peakMemory,
        "released_weight_bytes":denoiser.residentWeightBytes+sampler.residentWeightBytes])
    }
    let output:[String:Any]=["scope":sampling ? "small-token-three-step-trajectory" : "small-token-full-denoiser",
      "passed":passed,"runs":runs,"calibration":"Existing NNC Float32 trajectory gate: max(component floor, 2x independently measured CPU/GPU spread) per step; strict floor reported separately.","qualification":"Synthetic latent inputs with installed weights; no generated-media quality or production-size performance claim."]
    try JSONSerialization.data(withJSONObject:output,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:args[4]),options:.atomic)
    guard passed else { throw LTXError.invalid("Numerical qualification failed; see report.") }
  }
}
