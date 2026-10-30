import CryptoKit
import Darwin
import Foundation
import H3MLX
import InferenceContracts
import MLX
import MLXRandom

/// One native worker pipeline. Every weighted stage exits before the next begins;
/// only prepared activations and immutable file identities survive those scopes.
enum H3RegenerateWorker {
  private struct Signature:Equatable {
    let device:UInt64,inode:UInt64,size:Int64,modified:Int,modifiedNS:Int,changed:Int,changedNS:Int
    init(_ path:String) throws {
      var value=stat()
      guard lstat(path,&value) == 0,value.st_mode & S_IFMT == S_IFREG,value.st_size > 0 else {
        throw H3CheckpointError.invalid("A regeneration asset is not a nonempty regular file: \(path)")
      }
      device=UInt64(value.st_dev);inode=UInt64(value.st_ino);size=value.st_size
      modified=value.st_mtimespec.tv_sec;modifiedNS=value.st_mtimespec.tv_nsec
      changed=value.st_ctimespec.tv_sec;changedNS=value.st_ctimespec.tv_nsec
    }
  }
  private struct Witness {
    let path:String,resolved:String,signature:Signature
    init(path:String,sha256:String) throws {
      self.path=path;resolved=URL(fileURLWithPath:path).resolvingSymlinksInPath().path
      signature=try Signature(resolved)
      guard try H3RegenerateWorker.hashFile(URL(fileURLWithPath:resolved)) == sha256 else {
        throw H3CheckpointError.invalid("Regeneration asset changed after submission: \(path)")
      }
      try check()
    }
    func check() throws {
      guard URL(fileURLWithPath:path).resolvingSymlinksInPath().path == resolved,try Signature(resolved) == signature else {
        throw H3CheckpointError.invalid("Regeneration asset changed during execution: \(path)")
      }
    }
  }
  private static func digest(_ data:Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
  private static func hashFile(_ url:URL) throws -> String {
    let handle=try FileHandle(forReadingFrom:url);defer { try? handle.close() }
    var hash=SHA256()
    while try autoreleasepool(invoking:{ () throws -> Bool in
      try Task.checkCancellation()
      guard let data=try handle.read(upToCount:4*1024*1024),!data.isEmpty else { return false }
      hash.update(data:data);return true
    }) {}
    return hash.finalize().map { String(format:"%02x",$0) }.joined()
  }
  private static func arrayData(_ array:MLXArray) throws -> Data {
    try H3RegenerateArtifact.copyData(array)
  }
  private static func object<T:Encodable>(_ value:T) throws -> Any { try JSONSerialization.jsonObject(with:JSONEncoder().encode(value)) }
  static func execute(envelope:NativeVideoJobEnvelope,recipeData:Data,render:Bool,
    emitProgress:([String:Any])->Void) throws -> [String:Any] {
    let started=Date(),job=try H3RegenerateJob.decode(recipeData)
    try H3RegenerateProfile.validateBaseSHA256(job.components.transformer.identity.sourceSHA256)
    let sourceEncoderRequiredBytes:UInt64
    if job.source.encoderProfile == .publisherPosteriorFP32 {
      sourceEncoderRequiredBytes=try H3PublisherEncoderAdmission.validate(width:job.source.width,height:job.source.height,
        paddedFrames:job.source.paddedFrames,maximumStageBytes:job.budget.maximumStageBytes,physicalMemory:ProcessInfo.processInfo.physicalMemory)
    } else { sourceEncoderRequiredBytes=0 }
    let inferenceLease:NativeInferenceLease?
    if render {
      inferenceLease=try NativeInferenceLease.acquire {
        emitProgress(["event":"progress","stage":"waiting","message":"Waiting for another local inference job"])
      }
    } else { inferenceLease=nil }
    defer { inferenceLease?.release() }
    try Task.checkCancellation()
    let qualificationValue=ProcessInfo.processInfo.environment["WEETODD_H3_REGENERATE_QUALIFICATION"] ?? "0"
    guard ["0","1"].contains(qualificationValue) else { throw H3CheckpointError.invalid("Regeneration qualification flag requires0 or1.") }
    let qualification=qualificationValue == "1"
    let tokenizer=try H3QwenTokenizer(url:URL(fileURLWithPath:job.qwen.tokenizer.path))
    let textRequest=try H3QwenRequest.text(job.prompt,tokenizer:tokenizer)
    let admission=try H3RegenerateAdmission(width:job.target.width,height:job.target.height,
      visibleFrames:job.source.visibleFrames,paddedFrames:job.source.paddedFrames,fps:job.source.fps,textRows:textRequest.tags.count,
      maximumStageBytes:job.budget.maximumStageBytes,availableTemporaryBytes:job.budget.availableTemporaryBytes)
    try admission.validateExecution(qualificationProbe:qualification)
    guard admission.maximumStageBytes <= ProcessInfo.processInfo.physicalMemory else { throw H3CheckpointError.invalid("Regeneration stage budget exceeds this Mac’s physical memory.") }
    let geometry=try admission.geometry
    let sourceGeometry=try H3Geometry(regenerateWidth:job.source.width,height:job.source.height,paddedFrames:job.source.paddedFrames)
    let plan=try H3LatentUpscalePlan(sourceWidth:job.source.width,sourceHeight:job.source.height,
      targetWidth:admission.workingWidth,targetHeight:admission.workingHeight,mode:H3LatentUpscalePlan.Mode(rawValue:job.latentUpscaleMode.rawValue)!)
    if job.source.encoderProfile == .nativePosteriorMeanF16 {
      guard job.source.width <= 2048,job.source.height <= 2048,
        job.source.paddedFrames*job.source.width*job.source.height*3 <= 1024*1024*1024 else {
        throw H3CheckpointError.invalid("Native F16 source encoding exceeds its established canvas or RGB byte admission.")
      }
    }
    if job.decoder == .releasedH3 { _ = try H3VideoVAEDecoder.preflightRegenerate(geometry:geometry) }
    else {
      try H3FastVideoDecoder.validateGeometry(frames:geometry.videoLatentFrames,height:geometry.height/16,
        width:geometry.width/16,outputFrames:geometry.videoLatentFrames)
    }
    let output=URL(fileURLWithPath:envelope.outputDirectory),manager=FileManager.default
    guard !manager.fileExists(atPath:output.path),manager.fileExists(atPath:output.deletingLastPathComponent().path) else {
      throw H3CheckpointError.invalid("Regeneration output must be a fresh directory with an existing parent.")
    }
    let volume=try output.deletingLastPathComponent().resourceValues(forKeys:[.volumeAvailableCapacityForImportantUsageKey,.volumeAvailableCapacityKey])
    let free=volume.volumeAvailableCapacityForImportantUsage ?? volume.volumeAvailableCapacity.map(Int64.init)
    guard let free,free >= 0,UInt64(free) >= admission.temporaryBytes else { throw H3CheckpointError.invalid("The output volume lacks regeneration temporary space.") }
    // Verify full content once, then retain inode/size/mtime/ctime identities.
    var expected=[String:String]()
    func add(_ path:String,_ sha:String) throws {
      if let old=expected[path],old != sha { throw H3CheckpointError.invalid("Conflicting content identities for one regeneration asset.") }
      expected[path]=sha
    }
    try add(job.source.payload.path,job.source.payload.sha256)
    try add(job.qwen.tokenizer.path,job.qwen.tokenizer.sha256)
    for file in job.qwen.files { try add(file.path,file.sha256) }
    for component in job.components.all { try add(component.path,component.identity.sourceSHA256) }
    var witnesses=[Witness]()
    for (index,entry) in expected.sorted(by:{$0.key<$1.key}).enumerated() {
      emitProgress(["event":"progress","stage":"verify_assets","completed":index,"total":expected.count])
      witnesses.append(try Witness(path:entry.key,sha256:entry.value))
    }
    func checkAssets() throws { try Task.checkCancellation();for witness in witnesses { try witness.check() } }
    guard let sourceWitness=witnesses.first(where:{$0.path == job.source.payload.path}),
      UInt64(sourceWitness.signature.size) == job.source.payloadBytes else { throw H3CheckpointError.invalid("Prepared source payload byte count differs from the frozen source grid.") }
    let qwen=try H3QwenCheckpointLayout.inspect(root:URL(fileURLWithPath:job.qwen.checkpointRoot))
    var qwenPaths=Set(([qwen.embeddingFile]+qwen.layerFiles+[qwen.visionFile].compactMap{$0}).map{$0.resolvingSymlinksInPath().path})
    let manifest=URL(fileURLWithPath:job.qwen.checkpointRoot).appendingPathComponent("paged_text_encoder_manifest.json")
    if manager.fileExists(atPath:manifest.path) { qwenPaths.insert(manifest.resolvingSymlinksInPath().path) }
    guard qwenPaths == Set(job.qwen.files.map{URL(fileURLWithPath:$0.path).resolvingSymlinksInPath().path}) else {
      throw H3CheckpointError.invalid("Qwen frozen files must cover every inspected language/vision page and its manifest exactly.")
    }
    let transformer=try H3CheckpointLayout(url:URL(fileURLWithPath:job.components.transformer.path))
    guard transformer.fastVariant == nil,transformer.omniVariant == nil,transformer.blockCount == 50 else {
      throw H3CheckpointError.invalid("Regeneration requires the supported original H3 transformer layout.")
    }
    let adapter=try H3RegenerateAdapter(url:URL(fileURLWithPath:job.components.refiner.path),profile:.publisherThreeStep,companions:[.sharpCurve]);adapter.close()
    let sharp=try H3SharpCurve(tableURL:URL(fileURLWithPath:job.components.sharpInput.path),branchesURL:URL(fileURLWithPath:job.components.sharpCurve.path));sharp.close()
    let upscalerHeader=try H3LearnedUpscalerLayout.inspectHeader(url:URL(fileURLWithPath:job.components.latentUpscaler.path))
    let vae=try H3VideoVAELayout(url:URL(fileURLWithPath:job.components.videoVAE.path));try vae.validateVideoUse()
    try H3AudioVAEEncoder.inspect(checkpointURL:URL(fileURLWithPath:job.components.audioVAE.path))
    if let encoder=job.components.publisherEncoder {
      let encoderLayout=try H3PublisherEncoderLayout(url:URL(fileURLWithPath:encoder.path))
      guard encoderLayout.mean == vae.latentsMean,encoderLayout.standardDeviation == vae.latentsStandardDeviation else {
        throw H3CheckpointError.invalid("Publisher encoder and selected VAE normalization statistics differ.")
      }
    }
    if job.source.encoderProfile == .nativePosteriorMeanF16 { try H3VideoVAEEncoder.preflight(checkpointURL:URL(fileURLWithPath:job.components.videoVAE.path)) }
    if let fast=job.components.fastDecoder { _ = try H3FastDecoderLayout(weightsURL:URL(fileURLWithPath:fast.path),baseVAEURL:URL(fileURLWithPath:job.components.videoVAE.path)) }
    try checkAssets()
    var metadata:[String:Any]=["nativeRuntime":"swift-mlx","engine":"h3","task":"h3_regenerate","jobID":envelope.jobID.uuidString,
      "fingerprint":envelope.recipeSHA256,"eligibility":"allowed","issues":[],"inferenceExecuted":false,
      "sourceSHA256":job.source.originalSHA256,"sourcePayloadSHA256":job.source.payload.sha256,"sourceAudioSHA256":job.source.audioSHA256,
      "fps":24,"visibleFrames":job.source.visibleFrames,"paddedFrames":job.source.paddedFrames,
      "width":job.target.width,"height":job.target.height,"workingWidth":admission.workingWidth,"workingHeight":admission.workingHeight,
      "packedRows":admission.packedRows,"encoderProfile":job.source.encoderProfile.rawValue,"decoder":job.decoder.rawValue,
      "qualificationProbe":qualification,"publisherSpatialEncoderParityQualified":false,
      "normalizedRequest":["engine":"h3","prompt":job.prompt,"modelID":job.components.transformer.path,
        "configuration":["task":"h3_regenerate","width":job.target.width,"height":job.target.height,"seed":job.seed],"recipe":try object(job)],
      "preflightSeconds":Date().timeIntervalSince(started),"publisherSourceEncoderRequiredBytes":sourceEncoderRequiredBytes]
    guard render else { return metadata }
    let staging=output.deletingLastPathComponent().appendingPathComponent(".h3-regenerate-"+UUID().uuidString)
    try manager.createDirectory(at:staging,withIntermediateDirectories:false)
    defer { try? manager.removeItem(at:staging);Stream.gpu.synchronize();Memory.clearCache() }
    var timings=[String:Double](),stagePeaks=[String:Int](),overallPeak=Memory.activeMemory
    func stage<T>(_ name:String,_ body:() throws -> T) throws -> T {
      try checkAssets();emitProgress(["event":"progress","stage":name,"completed":0,"total":1])
      let time=Date();Memory.peakMemory=Memory.activeMemory
      do {
        let result=try autoreleasepool(invoking:body)
        let peak=Memory.peakMemory;stagePeaks[name]=peak;overallPeak=max(overallPeak,peak)
        Stream.gpu.synchronize();Memory.clearCache();try checkAssets()
        guard UInt64(peak) <= job.budget.maximumStageBytes else { throw H3CheckpointError.invalid("Regeneration stage exceeded its admitted MLX peak budget: \(name)") }
        timings[name]=Date().timeIntervalSince(time)
        emitProgress(["event":"progress","stage":name+"_weights_released","completed":1,"total":1,"residentWeightBytes":0])
        return result
      } catch { Stream.gpu.synchronize();Memory.clearCache();throw error }
    }
    func saveArray(_ array:MLXArray,_ filename:String) throws -> String {
      let data=try arrayData(array);try data.write(to:staging.appendingPathComponent(filename),options:.withoutOverwriting);return digest(data)
    }
    let text=try stage("text") { try H3QwenTextEncoder.encode(prompt:job.prompt,checkpointRoot:URL(fileURLWithPath:job.qwen.checkpointRoot),tokenizerURL:URL(fileURLWithPath:job.qwen.tokenizer.path)) }
    let embeddings=text.hidden.expandedDimensions(axis:0)
    guard text.tags == textRequest.tags,text.tokenIDs == textRequest.tokenIDs,embeddings.dtype == .bfloat16,embeddings.shape == [1,textRequest.tags.count,5120] else { throw H3CheckpointError.invalid("Qwen rows changed after preflight.") }
    let embeddingsSHA=try saveArray(embeddings,"text-embeddings.bf16")
    var posteriorSHA:String?
    var rows:MLXArray=try stage("source_encode") {
      let data=try Data(contentsOf:URL(fileURLWithPath:job.source.payload.path),options:.mappedIfSafe)
      if job.source.kind == .normalizedVideoRows {
        let result=data.withUnsafeBytes { MLXArray($0,[1,sourceGeometry.videoRows,96],type:Float.self) }
        guard all(isFinite(result)).item(Bool.self) else { throw H3CheckpointError.invalid("Source normalized rows contain nonfinite values.") }
        eval(result);return result
      }
      var rgb=[UInt8](data)
      let frameBytes=job.source.width*job.source.height*3,last=Array(rgb.suffix(frameBytes))
      for _ in job.source.visibleFrames..<job.source.paddedFrames { rgb.append(contentsOf:last) }
      if let encoder=job.components.publisherEncoder,let seed=job.posteriorSeed {
        let noise=MLXRandom.normal([1,sourceGeometry.videoLatentFrames,job.source.height/16,job.source.width/16,24],key:MLXRandom.key(seed)).asType(.float32)
        posteriorSHA=try saveArray(noise,"posterior-noise.f32")
        return try H3VideoVAEEncoder.encodePublisherVideoRows(checkpointURL:URL(fileURLWithPath:encoder.path),rgb8:rgb,
          frameCount:job.source.paddedFrames,width:job.source.width,height:job.source.height,posteriorNoise:noise)
      }
      let latent=try H3VideoVAEEncoder.encodeControlVideo(checkpointURL:URL(fileURLWithPath:job.components.videoVAE.path),rgb8:rgb,
        frameCount:job.source.paddedFrames,width:job.source.width,height:job.source.height)
      return try H3LatentCodec.videoEncoderRows(latents:latent,mean:vae.latentsMean,standardDeviation:vae.latentsStandardDeviation)
    }
    for hop in 1..<plan.grids.count {
      let before=plan.grids[hop-1],after=plan.grids[hop]
      rows=try stage("latent_upscale_\(hop)") {
        try H3LearnedLatentUpscaler.upscale(rows:rows,
          source:H3Geometry(regenerateWidth:before.width,height:before.height,paddedFrames:job.source.paddedFrames),
          target:H3Geometry(regenerateWidth:after.width,height:after.height,paddedFrames:job.source.paddedFrames),
          checkpointURL:URL(fileURLWithPath:job.components.latentUpscaler.path),expectedHeaderSHA256:upscalerHeader,
          videoVAE:URL(fileURLWithPath:job.components.videoVAE.path),mathProfile:.publisherBF16Arithmetic).videoRows
      }
    }
    let preparedSourceSHA=try saveArray(rows,"prepared-target-video.f32")
    let silence=try stage("audio_encode") { try H3AudioVAEEncoder.encodeRegenerateSilence(checkpointURL:URL(fileURLWithPath:job.components.audioVAE.path),paddedFrames:job.source.paddedFrames) }
    let silenceSHA=try saveArray(silence,"encoded-silence.bf16")
    let noise=MLXRandom.normal(rows.shape,key:MLXRandom.key(job.seed)).asType(.float32)
    let noiseSHA=try saveArray(noise,"refine-noise.f32")
    let decoderComponent=job.components.fastDecoder ?? job.components.videoVAE
    let decoderIdentity=NativeComponentIdentity(family:.h3,role:.decoder,sourceSHA256:decoderComponent.identity.sourceSHA256,
      layout:decoderComponent.identity.layout,importerVersion:decoderComponent.identity.importerVersion)
    var source:[String:Any]=["contentSHA256":preparedSourceSHA,"audioSHA256":job.source.audioSHA256,"normalization":"h3_sampler_normalized_v1",
      "width":job.source.width,"height":job.source.height,"visibleFrames":job.source.visibleFrames,"paddedFrames":job.source.paddedFrames,"fps":24,"encoderProfile":job.source.encoderProfile.rawValue]
    if let posteriorSHA { source["posteriorNoiseSHA256"]=posteriorSHA }
    let components=[job.components.transformer,job.components.refiner,job.components.sharpCurve,job.components.sharpInput,job.components.latentUpscaler,job.components.videoVAE,job.components.audioVAE]
    let inner:[String:Any]=["version":1,"task":"h3_regenerate","profile":"publisherThreeStep","schedule":job.schedule.rawValue,
      "components":try components.map { try object($0) },"source":source,"target":try object(job.target),
      "promptSHA256":job.promptSHA256,"embeddingsSHA256":embeddingsSHA,"noiseSHA256":noiseSHA,"silenceSHA256":silenceSHA,
      "decoder":["kind":job.decoder.rawValue,"identity":try object(decoderIdentity)],
      "budget":["maximumStageBytes":job.budget.maximumStageBytes,"availableTemporaryBytes":job.budget.availableTemporaryBytes,"textRows":text.tags.count]]
    let prepared=try H3RegenerateRecipe.prepare(data:JSONSerialization.data(withJSONObject:inner,options:.sortedKeys))
    try prepared.encodedRecipe.write(to:staging.appendingPathComponent("engine-recipe.json"),options:.withoutOverwriting)
    let refined=try stage("sampling") {
      try H3RegenerateRunner.run(prepared:prepared,source:rows,noise:noise,encodedSilence:silence,textEmbeddings:embeddings,textTags:text.tags,qualificationProbe:qualification,
        progress:{done,total in emitProgress(["event":"progress","stage":"sampling","completed":done,"total":total])},
        blockProgress:{step,done,total in emitProgress(["event":"progress","stage":"sampling_block","step":step,"completed":done,"total":total])})
    }
    let refinedSHA=try saveArray(refined.video,"refined-video.f32")
    rows=MLXArray([Float](),[0]);Memory.clearCache()
    let rgbURL=staging.appendingPathComponent("video.rgb24")
    guard manager.createFile(atPath:rgbURL.path,contents:nil) else { throw H3CheckpointError.invalid("Cannot create regeneration RGB output.") }
    let handle=try FileHandle(forWritingTo:rgbURL);defer { try? handle.close() }
    var written=0
    func writePixels(_ pixels:MLXArray) throws {
      for index in 0..<pixels.shape[1] where written < job.source.visibleFrames {
        try Task.checkCancellation()
        let selected=pixels[0..<1,index..<(index+1),admission.cropY..<(admission.cropY+job.target.height),admission.cropX..<(admission.cropX+job.target.width),0..<3]
        guard all(isFinite(selected)).item(Bool.self) else { throw H3CheckpointError.invalid("Regeneration decoder produced nonfinite visible pixels.") }
        let rgb=try H3FastVideoDecoder.rgb8(selected)
        try handle.write(contentsOf:arrayData(rgb));written += 1
        emitProgress(["event":"progress","stage":"video_decode","completed":written,"total":job.source.visibleFrames])
      }
    }
    try stage("video_decode") {
      if let fast=job.components.fastDecoder {
        let normalized=try H3LatentCodec.videoDecoderInput(rows:refined.video,latentFrames:geometry.videoLatentFrames,
          latentHeight:geometry.height/16,latentWidth:geometry.width/16,mean:Array(repeating:0,count:24),standardDeviation:Array(repeating:1,count:24))
        let decoder=try H3FastVideoDecoder(layout:H3FastDecoderLayout(weightsURL:URL(fileURLWithPath:fast.path),baseVAEURL:URL(fileURLWithPath:job.components.videoVAE.path)),
          precision:.bfloat16,maximumContextTokens:1_048_575,maximumActiveBytes:job.budget.maximumStageBytes)
        try writePixels(decoder.decode(normalizedLatents:normalized))
      } else {
        let raw=try H3LatentCodec.videoDecoderInput(rows:refined.video,latentFrames:geometry.videoLatentFrames,
          latentHeight:geometry.height/16,latentWidth:geometry.width/16,mean:vae.latentsMean,standardDeviation:vae.latentsStandardDeviation)
        try H3VideoVAEDecoder.decodeRegenerateChunks(checkpointURL:URL(fileURLWithPath:job.components.videoVAE.path),latent:raw,geometry:geometry,onChunk:writePixels)
      }
    }
    guard written == job.source.visibleFrames else { throw H3CheckpointError.invalid("Regeneration decoder did not produce the complete visible source interval.") }
    try handle.synchronize();try handle.close();try checkAssets()
    metadata["inferenceExecuted"]=true;metadata["engineRecipeFingerprint"]=prepared.fingerprint
    metadata["attentionExecution"]=refined.attentionPlan.metadata
    metadata["outputRGBPath"]=output.appendingPathComponent("video.rgb24").path
    metadata["video"]=output.appendingPathComponent("video.rgb24").path
    metadata["pixelFormat"]="rgb24"
    metadata["outputRGBSHA256"]=try hashFile(rgbURL);metadata["outputRGBBytes"]=job.source.visibleFrames*job.target.width*job.target.height*3
    metadata["refinedLatentsPath"]=output.appendingPathComponent("refined-video.f32").path;metadata["refinedLatentsSHA256"]=refinedSHA
    metadata["refinedLatentsShape"]=[1,admission.videoRows,96];metadata["refinedLatentsDType"]="F32LE"
    metadata["preparedTargetVideoSHA256"]=preparedSourceSHA;metadata["noiseSHA256"]=noiseSHA
    metadata["pixelConversion"]="imagenet-clamp-round-even-u8"
    metadata["noiseGenerator"]="mlx-keyed-normal-float32-v1";metadata["sourceAudioPreservationRequired"]=true;metadata["audioDecoded"]=false
    metadata["latentUpscaleMode"]=job.latentUpscaleMode.rawValue;metadata["latentUpscalerMath"]="publisherBF16Arithmetic"
    metadata["stageSeconds"]=timings;metadata["elapsedSeconds"]=Date().timeIntervalSince(started)
    metadata["peakMLXBytes"]=overallPeak;metadata["stagePeakMLXBytes"]=stagePeaks;metadata["residentWeightBytes"]=0
    let usage=try H3RenderResourceUsage.capture(peakMLXBytes:overallPeak)
    metadata["peakPhysicalBytes"]=usage.peakPhysicalBytes;metadata["currentPhysicalBytes"]=usage.currentPhysicalBytes
    try recipeData.write(to:staging.appendingPathComponent("worker-job.json"),options:.withoutOverwriting)
    try JSONSerialization.data(withJSONObject:metadata,options:[.sortedKeys,.prettyPrinted]).write(to:staging.appendingPathComponent("result.json"),options:.withoutOverwriting)
    try H3WorkerReceipt.publish(staging:staging,output:output,validate:checkAssets)
    return metadata
  }
}
