import CryptoKit
import Darwin
import Foundation

public struct NativeH3RestorationSetupResult: Sendable {
  public let profileURL:URL
  public let publisherEncoderURL:URL
  public let fastDecoderURL:URL?
}

extension NativeUpscalerSetup {
  struct H3PreparationPlan:Sendable {
    let profile:String,version:String,command:String,directory:String,filename:String
    let sources:[(sha:String,bytes:Int64,flag:String)]
    let outputs:[(sha:String,bytes:Int64,filename:String)]
  }
  static let h3BasePlan=H3PreparationPlan(profile:"ref2va-base",version:"h3-ref2va-int8-convrot-legacy-bf16-v1",
    command:"convert-ref2va-base",directory:"WeeTodd/H3/Ref2VA",filename:"minimax_h3_ref2va_int8_convrot_swift_v1.safetensors",
    sources:[("9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9",34_038_894_550,"--source")],
    outputs:[("37488d7799577d40db8e74318311344cb909c38632c0f693c52976a5509811ff",34_004_510_766,"minimax_h3_ref2va_int8_convrot_swift_v1.safetensors"),
      ("600307eba6a67388b333aa5a94537a9ea83da65a6f37664ba22cf079e40a1f3b",34_004_510_718,"minimax_h3_ref2va_int8_convrot_weetodd.safetensors")])
  static let h3EncoderPlan=H3PreparationPlan(profile:"publisher-encoder",version:"h3-encoder-f32-odhwi-v1",
    command:"convert-publisher-encoder",directory:"WeeTodd/H3-Video-VAE",filename:"publisher_encoder_f32_odhwi_v1.safetensors",
    sources:[("2e05e8bc23fa4071043e17fd242be8acd0685e781a43987432b2eae925be4198",4_955_986_528,"--shard2"),
      ("c05d6ac4b1a33de372799d708531da6320f6a3ce6d1ce6d895e770988e004a39",398_539_336,"--shard3")],
    outputs:[("d1be9c027ce656f6aff72fc225899ed7d288a32cbd92c0d93977c8da7671306c",721_364_224,"publisher_encoder_f32_odhwi_v1.safetensors")])
  private static func h3Library(_ url:URL) throws -> URL {
    let root=url.standardizedFileURL.resolvingSymlinksInPath()
    guard root.isFileURL,try root.resourceValues(forKeys:[.isDirectoryKey]).isDirectory == true else {
      throw StudioError.invalid("Reconnect and select an existing model library before H3 preparation.")
    }
    return root
  }
  private static func h3Verified(_ url:URL,sha:String,bytes:Int64) throws -> Bool {
    try NativeModelDownloads.verified(url,file:.init(repo:"native/prepared",revision:String(repeating:"0",count:40),
      filename:url.lastPathComponent,size:bytes,sha256:sha,target:url.lastPathComponent,provider:nil))
  }
  private static func h3Receipt(_ plan:H3PreparationPlan,url:URL,sha:String) -> NativeUpscalerConversionReceipt {
    let identity=plan.sources.map(\.sha).joined(separator:"|")
    let source=plan.sources.count == 1 ? identity:NativeHeadlessJob.hash(Data(identity.utf8))
    let key=NativeHeadlessJob.hash(Data((plan.version+"|"+identity).utf8))
    return .init(version:1,sourceSHA256:source,outputSHA256:sha,outputPath:url.path,
      profile:plan.profile,importerVersion:plan.version,conversionKey:key)
  }
  static func existingH3Preparation(_ plan:H3PreparationPlan,library:URL,existingFiles:[URL]) throws -> NativeUpscalerConversionReceipt? {
    let directory=library.appendingPathComponent(plan.directory)
    let candidates=existingFiles+plan.outputs.map{directory.appendingPathComponent($0.filename)}
    var seen=Set<String>()
    for candidate in candidates {
      try Task.checkCancellation()
      let url=candidate.resolvingSymlinksInPath()
      guard seen.insert(url.path).inserted,FileManager.default.fileExists(atPath:url.path) else {continue}
      for pin in plan.outputs where try h3Verified(url,sha:pin.sha,bytes:pin.bytes) {
        return h3Receipt(plan,url:url,sha:pin.sha)
      }
      if candidate.lastPathComponent == plan.filename {
        throw StudioError.invalid("Existing prepared H3 output differs from its verified identity; it was preserved.")
      }
    }
    return nil
  }
  static func executeH3Preparation(_ plan:H3PreparationPlan,sources:[URL],modelLibrary:URL,worker:URL?,
    existingFiles:[URL],cancellation:NativeHeadlessCancellation,progress:@escaping NativeModelDownloads.Progress) throws -> NativeUpscalerConversionReceipt {
    let library=try h3Library(modelLibrary)
    try cancellation.check();try Task.checkCancellation()
    if let receipt=try existingH3Preparation(plan,library:library,existingFiles:existingFiles) {
      progress("Reusing verified H3 "+plan.profile,1);return receipt
    }
    guard sources.count == plan.sources.count else {throw StudioError.invalid("Download or select the pinned H3 "+plan.profile+" sources before preparation.")}
    for (index,source) in sources.enumerated() {
      try cancellation.check();progress("Verifying H3 source \(index+1) of \(sources.count)",0)
      guard try h3Verified(source,sha:plan.sources[index].sha,bytes:plan.sources[index].bytes) else {
        throw StudioError.invalid("H3 preparation source differs from its pinned released checkpoint.")
      }
    }
    let executable=worker ?? Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("WeeToddH3MLXWorker")
    guard let executable,FileManager.default.isExecutableFile(atPath:executable.path) else {
      throw StudioError.invalid("Connect the native H3 worker before preparing models.")
    }
    let directory=library.appendingPathComponent(plan.directory)
    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
    let canonical=directory.resolvingSymlinksInPath()
    guard canonical.path.hasPrefix(library.path+"/") else {throw StudioError.invalid("H3 preparation output escaped the selected model library.")}
    let lock=Darwin.open(canonical.appendingPathComponent(".native-preparation.lock").path,O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0o600)
    guard lock >= 0 else {throw StudioError.invalid("Cannot acquire the H3 preparation lease.")}
    defer{flock(lock,LOCK_UN);close(lock)}
    guard flock(lock,LOCK_EX|LOCK_NB)==0 else {throw StudioError.invalid("Another native H3 preparation owns this destination.")}
    if let receipt=try existingH3Preparation(plan,library:library,existingFiles:existingFiles){return receipt}
    let output=canonical.appendingPathComponent(plan.filename)
    let staging=canonical.appendingPathComponent(".preparation-"+UUID().uuidString)
    try FileManager.default.createDirectory(at:staging,withIntermediateDirectories:false)
    defer{try? FileManager.default.removeItem(at:staging)}
    let stagedOutput=staging.appendingPathComponent(plan.filename)
    var arguments=[plan.command]
    for (index,source) in sources.enumerated(){arguments += [plan.sources[index].flag,source.path]}
    arguments += ["--output",stagedOutput.path,"--model-root",library.path]
    let log=canonical.appendingPathComponent(".prepare-\(UUID().uuidString).jsonl")
    var final:[String:Any]?
    let result=try ExternalEngineProcess.run(executable:executable,arguments:arguments,directory:canonical,log:log,
      cancellation:cancellation,maximumOutput:1024*1024,onLine:{line in
        guard let object=try? JSONSerialization.jsonObject(with:Data(line.utf8)) as? [String:Any] else{return}
        if object["event"] as? String == "progress",let done=object["completed"] as? Int,let total=object["total"] as? Int,total>0 {
          progress("Preparing native H3 "+plan.profile,Double(done)/Double(total))
        }
        if object["status"] as? String == "success" {final=object}
      })
    guard result.exitCode==0,let final,final["operation"] as? String == plan.command,
      final["format"] as? String == plan.version,final["path"] as? String == stagedOutput.path,
      let sha=final["sha256"] as? String,let pin=plan.outputs.first,
      sha==pin.sha,try h3Verified(stagedOutput,sha:pin.sha,bytes:pin.bytes) else {
      throw StudioError.invalid("Native H3 preparation did not publish the verified output. Inspect "+log.path)
    }
    try cancellation.check();try Task.checkCancellation()
    guard renamex_np(stagedOutput.path,output.path,UInt32(RENAME_EXCL)) == 0 else {
      throw StudioError.invalid("H3 preparation destination appeared during conversion; it was preserved.")
    }
    let receipt=h3Receipt(plan,url:output,sha:sha)
    let receiptURL=output.appendingPathExtension("conversion.json")
    try JSONEncoder().encode(receipt).write(to:receiptURL,options:.withoutOverwriting)
    return receipt
  }
  private static func prepareH3(_ plan:H3PreparationPlan,sources:[URL],modelLibrary:URL,worker:URL?,existingFiles:[URL],
    cancellation:NativeHeadlessCancellation,progress:@escaping NativeModelDownloads.Progress) async throws -> NativeUpscalerConversionReceipt {
    let task=Task.detached {try executeH3Preparation(plan,sources:sources,modelLibrary:modelLibrary,worker:worker,
      existingFiles:existingFiles,cancellation:cancellation,progress:progress)}
    return try await withTaskCancellationHandler(operation:{
      do {try cancellation.onCancellation{task.cancel()}} catch {task.cancel();_ = try? await task.value;throw error}
      defer{try? cancellation.onCancellation(nil)}
      return try await task.value
    },onCancel:{cancellation.cancel();task.cancel()})
  }
  public static func prepareH3Base(source:URL?=nil,modelLibrary:URL,worker:URL?=nil,existingFiles:[URL]=[],
    cancellation:NativeHeadlessCancellation=NativeHeadlessCancellation(),progress:@escaping NativeModelDownloads.Progress={_,_ in}) async throws -> NativeUpscalerConversionReceipt {
    try await prepareH3(h3BasePlan,sources:source.map{[$0]} ?? [],modelLibrary:modelLibrary,worker:worker,
      existingFiles:existingFiles,cancellation:cancellation,progress:progress)
  }
  public static func prepareH3Encoder(shards:[URL]=[],modelLibrary:URL,worker:URL?=nil,existingFiles:[URL]=[],
    cancellation:NativeHeadlessCancellation=NativeHeadlessCancellation(),progress:@escaping NativeModelDownloads.Progress={_,_ in}) async throws -> NativeUpscalerConversionReceipt {
    try await prepareH3(h3EncoderPlan,sources:shards,modelLibrary:modelLibrary,worker:worker,
      existingFiles:existingFiles,cancellation:cancellation,progress:progress)
  }
  static func ensureH3Package(_ id:String,family:String,catalog:URL,library:URL,token:String?=nil,progress:@escaping NativeModelDownloads.Progress,transfer:NativeModelDownloads.Transfer?=nil) async throws -> URL {
    guard let package=try NativeModelDownloads.catalog(at:catalog).first(where:{$0.descriptor.id==id}) else {throw StudioError.invalid("Pinned H3 package is missing from the catalog: "+id)}
    let destination=library.appendingPathComponent(family),installed=destination.appendingPathComponent(id)
    if FileManager.default.fileExists(atPath:installed.path) {
      for file in package.files {
          guard try NativeModelDownloads.verified(installed.appendingPathComponent(file.target),file:file) else {
            throw StudioError.invalid("Installed H3 package changed: "+id)
          }
      }
      return installed
    }
    if let transfer {
      return try await NativeModelDownloads.prepare(id:id,catalog:catalog,destination:destination,
        existingRoots:[library.path],token:token,progress:progress,transfer:transfer)
    }
    return try await NativeModelDownloads.prepare(id:id,catalog:catalog,destination:destination,
      existingRoots:[library.path],token:token,progress:progress)
  }
  public static func prepareH3Restoration(catalog:URL,modelLibrary:URL,profileDirectory:URL,workerPath:String,
    includeFastDecoder:Bool,token:String?=nil,progress:@escaping NativeModelDownloads.Progress) async throws -> NativeH3RestorationSetupResult {
    let library=try h3Library(modelLibrary),worker=URL(fileURLWithPath:workerPath)
    let existingBase=try existingH3Preparation(h3BasePlan,library:library,existingFiles:[])
    let base:NativeUpscalerConversionReceipt
    if let existingBase {base=existingBase} else {
      let raw=try await ensureH3Package("h3-regenerate-base-source",family:"WeeTodd/H3/Ref2VA",catalog:catalog,library:library,token:token,progress:progress)
      base=try await prepareH3Base(source:raw.appendingPathComponent("minimax_h3_ref2va_int8_convrot.safetensors"),modelLibrary:library,worker:worker,progress:progress)
    }
    let existingEncoder=try existingH3Preparation(h3EncoderPlan,library:library,existingFiles:[])
    let encoder:NativeUpscalerConversionReceipt
    if let existingEncoder {encoder=existingEncoder} else {
      let raw=try await ensureH3Package("h3-regenerate-encoder-source",family:"WeeTodd/H3-Video-VAE",catalog:catalog,library:library,token:token,progress:progress)
      encoder=try await prepareH3Encoder(shards:[2,3].map{raw.appendingPathComponent("diffusion_pytorch_model-0000\($0)-of-00003.safetensors")},modelLibrary:library,worker:worker,progress:progress)
    }
    let support=try await ensureH3Package("h3-swift-ref2va-support",family:"WeeTodd/H3/Support",catalog:catalog,library:library,token:token,progress:progress)
    let qwen=try await ensureH3Package("h3-qwen-q8-vision-preconverted",family:"WeeTodd-H3/text_encoders",catalog:catalog,library:library,token:token,progress:progress)
    let video=try await ensureH3Package("h3-video-vae-q8-preconverted",family:"WeeTodd/H3-Video-VAE",catalog:catalog,library:library,token:token,progress:progress)
    let audio=try await ensureH3Package("h3-folded-audio-vae-preconverted",family:"WeeTodd/H3-Audio-VAE",catalog:catalog,library:library,token:token,progress:progress)
    _ = try await ensureH3Package("h3-regenerate-publisher",family:"WeeTodd/LoRAs/H3",catalog:catalog,library:library,token:token,progress:progress)
    _ = try await prepareSharpTable(modelLibrary:library)
    var fast:URL?
    if includeFastDecoder {
      let folder=try await ensureH3Package("h3-fast-decoder",family:"WeeTodd/H3-Video-VAE",catalog:catalog,library:library,token:token,progress:progress)
      fast=folder.appendingPathComponent("vae_fast/fast_h3_vae_noncausal_24l.safetensors")
    }
    guard let preset=NativeModelSetup.catalog().first(where:{$0.id=="swift-h3-reference"}) else {throw StudioError.invalid("Native Ref2VA setup preset is unavailable.")}
    let selected=["checkpoint":support.path,"transformer":base.outputPath,"text_encoder":qwen.path,"vision_encoder":qwen.path,
      "processor":support.appendingPathComponent("processor").path,"tokenizer":support.appendingPathComponent("tokenizer/tokenizer.json").path,
      "video_vae":video.appendingPathComponent("video_vae_affine_q8.safetensors").path,"audio_vae":audio.appendingPathComponent("audio_vae.safetensors").path]
    let recipe=try NativeModelSetup.recipe(preset:preset,selected:selected,memoryMode:.automatic)
    let data=try JSONSerialization.data(withJSONObject:recipe,options:[.prettyPrinted,.sortedKeys])
    try FileManager.default.createDirectory(at:profileDirectory,withIntermediateDirectories:true)
    let profile=profileDirectory.appendingPathComponent("h3-ref2va-"+NativeHeadlessJob.hash(data)+".json")
    if FileManager.default.fileExists(atPath:profile.path) {
      guard try Data(contentsOf:profile)==data else {throw StudioError.invalid("Prepared H3 profile changed; existing file was preserved.")}
    } else {try data.write(to:profile,options:.withoutOverwriting)}
    progress("Native H3 restoration models and profile are prepared.",1)
    return .init(profileURL:profile,publisherEncoderURL:URL(fileURLWithPath:encoder.outputPath),fastDecoderURL:fast)
  }
}
