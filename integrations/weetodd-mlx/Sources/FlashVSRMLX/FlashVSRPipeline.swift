import CryptoKit
import Darwin
import Foundation
import InferenceContracts
import MLX
import TensorIO

/// Native, staged raw-RGB execution. The worker owns its inference lease and
/// media clock/audio delivery; this layer never launches a subprocess or Python.
public enum FlashVSRPipeline {
  public struct Progress:Codable,Sendable {
    public let stage:String
    public let completed:Int,total:Int
    public let activeBytes:UInt64
  }
  public struct Admission:Codable,Sendable {
    public let memory:FlashVSRMemoryPlan
    public let temporal:FlashVSRTemporalPlan
    public let sourceSHA256:String
    public let modelSHA256:[String:String]
  }
  public struct Result:Codable,Sendable {
    public let inputSHA256:String,outputSHA256:String
    public let visibleFrames:Int,fpsNumerator:Int,fpsDenominator:Int
    public let peakMLXBytes:UInt64,peakProcessFootprintBytes:UInt64
    public let stageSeconds:[String:Double]
    public let releasedStageActiveBytes:[String:UInt64]
    public let modelSHA256:[String:String]
    public let temporalLayoutVersion:String
  }
  private struct Models {
    let transformer:URL,projection:URL,context:URL,decoder:URL
    let variant:FlashVSRVariant
    let hashes:[String:String]
    init(_ recipe:NativeFlashVSRRecipe)throws {
      let root=URL(fileURLWithPath:recipe.modelRoot,isDirectory:true)
      variant=recipe.variant == .turbo ? .turbo:.base
      let assets=try NativeFlashVSRAssets.resolve(root:root,variant:recipe.variant,decoder:recipe.decoder,policy:.native)
      transformer=assets.transformer;projection=assets.projection;context=assets.context;decoder=assets.decoder
      let decoderProfile=recipe.decoder == .tiny ? TorchTensorImportProfile.tinyDecoder:.wanVAE
      let pins=FlashVSRCheckpoint.releasedDigests(variant)
      hashes=["transformer":pins.transformer,"projection":pins.projection,"context":TorchTensorImportProfile.positivePrompt.sourceSHA256,"decoder":decoderProfile.sourceSHA256]
    }
    var identity:FlashVSRCheckpoint.Identity {
      .init(variant:variant,transformerSHA256:hashes["transformer"]!,projectionSHA256:hashes["projection"]!,contextSHA256:hashes["context"]!)
    }
  }
  /// No learned MLX arrays are loaded. Validate exact source size/hash, all
  /// selected pins and headers, real-path isolation, disk and owned-memory bounds.
  public static func preflight(recipe:NativeFlashVSRRecipe)throws->Admission {
    try Task.checkCancellation();try recipe.validate()
    let memory=try FlashVSRMemoryPlan(recipe:recipe),temporal=try FlashVSRTemporalPlan(recipe:recipe)
    let source=try RGBSource(recipe:recipe)
    try source.verifySHA256(recipe.inputSHA256)
    let output=URL(fileURLWithPath:recipe.outputRGBPath),scratch=URL(fileURLWithPath:recipe.scratchDirectory,isDirectory:true)
    let modelsRoot=URL(fileURLWithPath:recipe.modelRoot,isDirectory:true).resolvingSymlinksInPath().standardizedFileURL.path
    let resolvedScratch=scratch.resolvingSymlinksInPath().standardizedFileURL.path
    let resolvedInput=URL(fileURLWithPath:recipe.inputRGBPath).resolvingSymlinksInPath().standardizedFileURL.path
    let resolvedOutput=output.deletingLastPathComponent().resolvingSymlinksInPath().appendingPathComponent(output.lastPathComponent).standardizedFileURL.path
    guard !containsPath(resolvedScratch,modelsRoot),!containsPath(modelsRoot,resolvedScratch),
      !containsPath(resolvedScratch,resolvedInput),!containsPath(resolvedScratch,resolvedOutput),resolvedInput != resolvedOutput else {
      throw FlashVSRError.invalid("Native scratch must be separate from model weights and source/output after resolving symbolic links.")
    }
    var existing=stat()
    guard lstat(output.path,&existing) != 0,errno==ENOENT else { throw FlashVSRError.invalid("Native RGB output already exists; choose an unused output path.") }
    let outputParent=try writableAncestor(output.deletingLastPathComponent()),scratchParent=try writableAncestor(scratch)
    let free=min(try freeBytes(outputParent),try freeBytes(scratchParent))
    try memory.validate(maximumActiveBytes:recipe.maximumActiveBytes,availableDiskBytes:free,physicalMemoryBytes:ProcessInfo.processInfo.physicalMemory)
    let models=try Models(recipe)
    for component in [models.transformer,models.projection,models.context,models.decoder] {
      guard !containsPath(resolvedScratch,component.path) else { throw FlashVSRError.invalid("Native model weights cannot resolve into disposable scratch.") }
    }
    // Checkpoint initializers perform source hash + complete architecture header
    // admission without materializing any learned tensor array.
    _ = try FlashVSRCheckpoint(url:models.projection,variant:models.variant,component:.projection)
    _ = try FlashVSRCheckpoint(url:models.transformer,variant:models.variant,component:.transformer)
    _ = try TorchTensorArchive.open(url:models.context,expectedSHA256:models.hashes["context"]!,profile:.positivePrompt)
    let selected=recipe.decoder == .tiny ? TorchTensorImportProfile.tinyDecoder:.wanVAE
    _ = try TorchTensorArchive.open(url:models.decoder,expectedSHA256:selected.sourceSHA256,profile:selected)
    try source.checkUnchanged()
    return Admission(memory:memory,temporal:temporal,sourceSHA256:recipe.inputSHA256,modelSHA256:models.hashes)
  }
  public static func run(recipe:NativeFlashVSRRecipe,
    progress:(Progress)throws->Void={ _ in })throws->Result {
    try Task.checkCancellation()
    try progress(.init(stage:"preflight",completed:0,total:1,activeBytes:UInt64(Memory.activeMemory)))
    let admission=try preflight(recipe:recipe),models=try Models(recipe)
    let source=try RGBSource(recipe:recipe);try source.verifySHA256(recipe.inputSHA256)
    let scratchRoot=URL(fileURLWithPath:recipe.scratchDirectory,isDirectory:true)
    try FileManager.default.createDirectory(at:scratchRoot,withIntermediateDirectories:true)
    let owned=scratchRoot.appendingPathComponent("FlashVSR-"+UUID().uuidString,isDirectory:true)
    guard mkdir(owned.path,S_IRWXU)==0 else { throw FlashVSRError.invalid("Cannot create an exclusive native job scratch directory.") }
    defer { try? FileManager.default.removeItem(at:owned) }
    let oldCache=Memory.cacheLimit,oldLimit=Memory.memoryLimit
    let physical=ProcessInfo.processInfo.physicalMemory,activeLimit=min(recipe.maximumActiveBytes,physical-physical/4)
    Memory.cacheLimit=min(oldCache,64*1024*1024);Memory.memoryLimit=Int(activeLimit)
    Stream.gpu.synchronize();Memory.clearCache();Memory.peakMemory=0
    defer { Stream.gpu.synchronize();Memory.clearCache();Memory.cacheLimit=oldCache;Memory.memoryLimit=oldLimit }
    var stageSeconds:[String:Double]=[:],released:[String:UInt64]=[:],activeStage:String?
    func emit(_ stage:String,_ completed:Int,_ total:Int)throws {
      try Task.checkCancellation()
      guard UInt64(Memory.activeMemory)<=activeLimit else { throw FlashVSRError.invalid("Native FlashVSR exceeded its admitted active-memory cap during \(stage).") }
      try progress(.init(stage:stage,completed:completed,total:total,activeBytes:UInt64(Memory.activeMemory)))
    }
    func release(_ stage:String)throws {
      Stream.gpu.synchronize();Memory.clearCache()
      released[stage]=UInt64(Memory.activeMemory);activeStage=nil
      try emit(stage+"_released",1,1)
    }
    do {
      try emit("preflight",1,1)
      activeStage="projection";var started=ProcessInfo.processInfo.systemUptime
      let features=try project(recipe:recipe,models:models,plan:admission.temporal,source:source,root:owned,emit:emit)
      stageSeconds["projection"]=ProcessInfo.processInfo.systemUptime-started;try release("projection")
      activeStage="transformer";started=ProcessInfo.processInfo.systemUptime
      let latents=try transform(recipe:recipe,models:models,features:features,root:owned,emit:emit)
      stageSeconds["transformer"]=ProcessInfo.processInfo.systemUptime-started;try release("transformer")
      activeStage="decoder";started=ProcessInfo.processInfo.systemUptime
      let output=try decode(recipe:recipe,models:models,plan:admission.temporal,latents:latents,source:source,root:owned,emit:emit)
      stageSeconds["decoder"]=ProcessInfo.processInfo.systemUptime-started;try release("decoder")
      try source.checkUnchanged()
      let peak=try processPeak()
      try emit("publishing",0,1)
      let outputHash=try output.publish()
      let result=Result(inputSHA256:recipe.inputSHA256,outputSHA256:outputHash,visibleFrames:recipe.frames,
        fpsNumerator:recipe.fpsNumerator,fpsDenominator:recipe.fpsDenominator,
        peakMLXBytes:UInt64(Memory.peakMemory),peakProcessFootprintBytes:peak,stageSeconds:stageSeconds,
        releasedStageActiveBytes:released,modelSHA256:models.hashes,temporalLayoutVersion:FlashVSRTemporalPlan.version)
      // Publication is the commit point. Progress delivery after a completed
      // atomic commit cannot turn its immutable result into a partial failure.
      try? progress(.init(stage:"completed",completed:recipe.frames,total:recipe.frames,activeBytes:UInt64(Memory.activeMemory)));return result
    } catch {
      // Helpers have unwound, dropping weighted state before synchronization.
      Stream.gpu.synchronize();Memory.clearCache()
      if let stage=activeStage {
        released[stage]=UInt64(Memory.activeMemory)
        try? progress(.init(stage:stage+"_released",completed:1,total:1,activeBytes:UInt64(Memory.activeMemory)))
      }
      throw error
    }
  }
  private static func project(recipe:NativeFlashVSRRecipe,models:Models,plan:FlashVSRTemporalPlan,
    source:RGBSource,root:URL,emit:(String,Int,Int)throws->Void)throws->[Asset] {
    let checkpoint=try FlashVSRCheckpoint(url:models.projection,variant:models.variant,component:.projection)
    let session=try LQProjectionSession(checkpoint:checkpoint,sourceID:recipe.sourceID)
    defer { session.reset(sourceID:recipe.sourceID) }
    var result:[Asset]=[]
    try emit("projection",0,plan.groupCount)
    for group in 0..<plan.groupCount {
      try source.checkUnchanged()
      let indices=try plan.projectionSourceIndices(group:group)
      var start=0
      if group==0 {
        guard try session.process(source.rgb(indices:[indices[0]]),sourceID:recipe.sourceID)==nil else { throw FlashVSRError.invalid("Projection warm-up unexpectedly emitted a latent condition.") }
        start=1
      }
      var parts:[MLXArray]=[]
      for chunk in 0..<4 {
        let offset=start+4*chunk
        guard let current=try session.process(source.rgb(indices:Array(indices[offset..<offset+4])),sourceID:recipe.sourceID),current.count==1 else { throw FlashVSRError.invalid("Projection did not emit its required bounded condition.") }
        parts.append(current[0])
      }
      let condition=concatenated(parts,axis:1);eval(condition)
      guard condition.shape==[1,recipe.workingHeight/16*recipe.workingWidth/16*4,1536] else { throw FlashVSRError.invalid("Projection feature grid differs from selected DiT.") }
      let asset=try Asset.write(condition,name:"condition",url:root.appendingPathComponent("feature-\(group).safetensors"),
        identity:identity(recipe:recipe,models:models,stage:"feature",group:group))
      result.append(asset);try emit("projection",group+1,plan.groupCount)
    }
    return result
  }
  private static func transform(recipe:NativeFlashVSRRecipe,models:Models,features:[Asset],root:URL,
    emit:(String,Int,Int)throws->Void)throws->[Asset] {
    let checkpoint=try FlashVSRCheckpoint(url:models.transformer,variant:models.variant,component:.transformer)
    let model=try FlashVSRTransformer(checkpoint:checkpoint,fixedContextURL:models.context,identity:models.identity)
    let state=FlashVSRStreamState(sourceID:recipe.sourceID)
    defer { state.reset(sourceID:recipe.sourceID) }
    let shape=[1,16,4,recipe.workingHeight/8,recipe.workingWidth/8]
    var result:[Asset]=[]
    try emit("transformer",0,features.count)
    for (group,asset) in features.enumerated() {
      let condition=try asset.load(name:"condition",identity:identity(recipe:recipe,models:models,stage:"feature",group:group))
      // Group-keyed native noise is stable across scheduling/replay. It does not
      // claim PyTorch/CUDA RNG bit parity. This seed layout is versioned above.
      let key=MLXRandom.key(UInt64(recipe.seed)+UInt64(group))
      let noise=MLXRandom.normal(shape,key:key).asType(.bfloat16);eval(noise)
      let velocity=try model.predict(noise:noise,conditions:[condition],sourceID:recipe.sourceID,state:state)
      let latent=try FlashVSRScheduler.step(noise:noise,velocity:velocity);eval(latent)
      guard isFinite(latent).all().item(Bool.self) else { throw FlashVSRError.invalid("Native FlashVSR produced nonfinite latents.") }
      result.append(try Asset.write(latent,name:"latent",url:root.appendingPathComponent("latent-\(group).safetensors"),
        identity:identity(recipe:recipe,models:models,stage:"latent",group:group)))
      try emit("transformer",group+1,features.count)
    }
    return result
  }
  private static func decode(recipe:NativeFlashVSRRecipe,models:Models,plan:FlashVSRTemporalPlan,
    latents:[Asset],source:RGBSource,root:URL,emit:(String,Int,Int)throws->Void)throws->RGBOutput {
    let output=try RGBOutput(url:URL(fileURLWithPath:recipe.outputRGBPath),expectedBytes:recipe.outputBytes)
    let merger=FlashVSRTemporalMerger(plan:plan)
    let mode:FlashVSRColorCorrection.Mode
    switch recipe.colorFix { case .wavelet:mode = .wavelet(levels:recipe.waveletLevels);case .adain:mode = .adain(epsilon:1e-5);case .off:mode = .off }
    var published=0
    func receive(_ offset:Int,_ frames:MLXArray)throws {
      guard frames.ndim==4,frames.shape==[frames.shape[0],recipe.workingHeight,recipe.workingWidth,3] else { throw FlashVSRError.invalid("Selected decoder changed the prepared working canvas.") }
      try merger.consume(offset:offset,frames:frames) { index,value in
        guard index==published,isFinite(value).all().item(Bool.self) else { throw FlashVSRError.invalid("Decoder merge produced nonfinite/discontinuous visible RGB.") }
        let reference=try source.rgb(indices:[index]).transposed(0,2,3,4,1).reshaped([1,recipe.workingHeight,recipe.workingWidth,3])
        let corrected=try FlashVSRColorCorrection.apply(value.expandedDimensions(axis:0),reference:reference,mode:mode)
        guard isFinite(corrected).all().item(Bool.self) else { throw FlashVSRError.invalid("Native color correction produced nonfinite RGB.") }
        let visible=corrected[0,0..<recipe.height,0..<recipe.width,0...]
        let bytes=floor(clip((visible+1)*127.5,min:0,max:255)+0.5).asType(.uint8).reshaped([-1])
        try bytes.asArray(UInt8.self).withUnsafeBytes { try output.append($0) }
        published += 1;try emit("decoder",published,recipe.frames)
      }
    }
    try emit("decoder",0,recipe.frames)
    switch recipe.decoder {
    case .tiny:
      let model=try FlashVSRTinyDecoder(checkpoint:models.decoder)
      var carry=FlashVSRTinyDecoder.Carry(sourceID:recipe.sourceID)
      defer { carry.reset(sourceID:recipe.sourceID);model.unload() }
      for (group,asset) in latents.enumerated() {
        let latent=try asset.load(name:"latent",identity:identity(recipe:recipe,models:models,stage:"latent",group:group))
        guard latent.shape==[1,16,4,recipe.workingHeight/8,recipe.workingWidth/8] else { throw FlashVSRError.invalid("Latent spool geometry changed.") }
        for frame in 0..<4 {
          let offset=carry.emittedFrames,count=offset==0 ? 1:4
          let indices=(offset..<offset+count).map { plan.sourceIndex(expandedIndex:$0) }
          try model.decode(latents:latent[0...,0...,frame..<frame+1],sourceRGB:source.rgb(indices:indices),sourceID:recipe.sourceID,carry:&carry,receive:receive)
        }
      }
    case .wan:
      let model=try FlashVSRWanDecoder(checkpoint:models.decoder)
      var carry=FlashVSRWanDecoder.Carry(sourceID:recipe.sourceID,cacheDirectory:root)
      defer { carry.reset(sourceID:recipe.sourceID);model.unload() }
      for (group,asset) in latents.enumerated() {
        let latent=try asset.load(name:"latent",identity:identity(recipe:recipe,models:models,stage:"latent",group:group))
        guard latent.shape==[1,16,4,recipe.workingHeight/8,recipe.workingWidth/8] else { throw FlashVSRError.invalid("Latent spool geometry changed.") }
        for frame in 0..<4 { try model.decode(latents:latent[0...,0...,frame..<frame+1],sourceID:recipe.sourceID,carry:&carry,receive:receive) }
      }
    }
    try merger.finish();try source.checkUnchanged()
    guard published==recipe.frames else { throw FlashVSRError.invalid("Native output frame count differs from the frozen source interval.") }
    try output.seal();return output
  }
  private static func identity(recipe:NativeFlashVSRRecipe,models:Models,stage:String,group:Int)->[String:String] {
    ["source":recipe.inputSHA256,"source_id":recipe.sourceID,"variant":recipe.variant.rawValue,"stage":stage,"group":String(group),
      "transformer":models.hashes["transformer"]!,"projection":models.hashes["projection"]!,"context":models.hashes["context"]!,
      "layout":FlashVSRTemporalPlan.version,"seed":String(recipe.seed),"compression":String(recipe.temporalCompression)]
  }
  private static func containsPath(_ parent:String,_ child:String)->Bool { child==parent || child.hasPrefix(parent+"/") }
  private static func writableAncestor(_ requested:URL)throws->URL {
    var cursor=requested
    var status=stat()
    while lstat(cursor.path,&status) != 0 {
      guard errno==ENOENT,cursor.path != "/" else { throw FlashVSRError.invalid("Cannot inspect native output/scratch parent.") }
      cursor=cursor.deletingLastPathComponent()
    }
    var resolved=stat()
    guard stat(cursor.path,&resolved)==0,resolved.st_mode&S_IFMT==S_IFDIR,access(cursor.path,W_OK)==0 else { throw FlashVSRError.invalid("Native output/scratch parent is not a writable local directory.") }
    return cursor
  }
  private static func freeBytes(_ url:URL)throws->UInt64 {
    let values=try FileManager.default.attributesOfFileSystem(forPath:url.path)
    guard let bytes=values[.systemFreeSize] as? NSNumber else { throw FlashVSRError.invalid("Cannot preflight native scratch disk space.") };return bytes.uint64Value
  }
  private static func processPeak()throws->UInt64 {
    var info=task_vm_info_data_t(),count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
    let result=withUnsafeMutablePointer(to:&info) { pointer in
      pointer.withMemoryRebound(to:integer_t.self,capacity:Int(count)) { task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count) }
    }
    guard result==KERN_SUCCESS else { throw FlashVSRError.invalid("Cannot measure native worker physical-memory peak.") }
    return UInt64(info.ledger_phys_footprint_peak)
  }
}

private extension FlashVSRPipeline {
  struct FileIdentity:Equatable {
    let device:dev_t,inode:ino_t,size:off_t
    let modifiedSeconds:Int,modifiedNanos:Int,changedSeconds:Int,changedNanos:Int
    init(_ value:stat) {
      device=value.st_dev;inode=value.st_ino;size=value.st_size
      modifiedSeconds=value.st_mtimespec.tv_sec;modifiedNanos=value.st_mtimespec.tv_nsec
      changedSeconds=value.st_ctimespec.tv_sec;changedNanos=value.st_ctimespec.tv_nsec
    }
  }
  final class RGBSource {
    private let url:URL,fd:Int32,identity:FileIdentity
    private let width:Int,height:Int,frames:Int,frameBytes:Int
    private var cached:(index:Int,data:Data)?
    init(recipe:NativeFlashVSRRecipe)throws {
      url=URL(fileURLWithPath:recipe.inputRGBPath);width=recipe.workingWidth;height=recipe.workingHeight;frames=recipe.frames
      frameBytes=width*height*3
      let opened=Darwin.open(url.path,O_RDONLY|O_CLOEXEC|O_NONBLOCK|O_NOFOLLOW)
      guard opened>=0 else { throw FlashVSRError.invalid("Cannot open prepared rawRGB24 source.") }
      var accepted=false;defer { if !accepted { Darwin.close(opened) } }
      var status=stat()
      guard fstat(opened,&status)==0,status.st_mode&S_IFMT==S_IFREG,status.st_size>=0,
        UInt64(status.st_size)==recipe.inputBytes else { throw FlashVSRError.invalid("Prepared RGB24 source byte count differs from working canvas × visible frames ×3.") }
      fd=opened;identity=FileIdentity(status);accepted=true
      try checkUnchanged()
    }
    deinit { Darwin.close(fd) }
    func checkUnchanged()throws {
      try Task.checkCancellation()
      var status=stat(),pathname=stat()
      guard fstat(fd,&status)==0,lstat(url.path,&pathname)==0,FileIdentity(status)==identity,FileIdentity(pathname)==identity else { throw FlashVSRError.invalid("Prepared RGB source changed during native restoration.") }
    }
    private func read(offset:UInt64,count:Int)throws->Data {
      try Task.checkCancellation()
      guard offset<=UInt64(identity.size),count>=0,UInt64(count)<=UInt64(identity.size)-offset else { throw FlashVSRError.invalid("RGB24 source read lies outside its frozen interval.") }
      var data=Data(count:count)
      try data.withUnsafeMutableBytes { bytes in
        var done=0
        while done<count {
          try Task.checkCancellation()
          let n=pread(fd,bytes.baseAddress!.advanced(by:done),min(count-done,4*1024*1024),off_t(offset)+off_t(done))
          if n<0 && errno==EINTR { continue }
          guard n>0 else { throw FlashVSRError.invalid("Prepared RGB source is truncated or unreadable.") };done += n
        }
      }
      return data
    }
    func verifySHA256(_ expected:String)throws {
      var hash=SHA256(),offset:UInt64=0
      while offset<UInt64(identity.size) {
        let data=try read(offset:offset,count:Int(min(4*1024*1024,UInt64(identity.size)-offset)));hash.update(data:data);offset += UInt64(data.count)
      }
      guard hash.finalize().map({ String(format:"%02x",$0) }).joined()==expected else { throw FlashVSRError.invalid("Prepared RGB24 source SHA-256 differs from its resolved request.") }
      try checkUnchanged()
    }
    func rgb(indices:[Int])throws->MLXArray {
      guard !indices.isEmpty,indices.count<=4,indices.allSatisfy({ (0..<frames).contains($0) }) else { throw FlashVSRError.invalid("Native source reads must use one through four frozen source indices.") }
      try checkUnchanged();var images:[MLXArray]=[]
      for index in indices {
        let data:Data
        if let cached,cached.index==index { data=cached.data }
        else { data=try read(offset:UInt64(index)*UInt64(frameBytes),count:frameBytes);cached=(index,data) }
        let image=data.withUnsafeBytes { MLXArray($0,[height,width,3],type:UInt8.self).asType(.float32)/127.5-1 }
        eval(image);images.append(image)
      }
      let output=stacked(images,axis:0).transposed(3,0,1,2).expandedDimensions(axis:0)
      eval(output);try checkUnchanged();return output
    }
  }
  struct Asset {
    let url:URL,sha256:String
    static func write(_ value:MLXArray,name:String,url:URL,identity:[String:String])throws->Self {
      guard value.dtype == .float32 || value.dtype == .bfloat16 else { throw FlashVSRError.invalid("Native feature/latent spool requires FP32/BF16.") }
      let dtype=value.dtype == .float32 ? "F32":"BF16"
      let writer=try SafeTensorStreamWriter(url:url,tensors:[name:try .init(dtype:dtype,shape:value.shape.map(UInt64.init))],metadata:identity)
      let flat=value.reshaped([-1]),width=value.dtype == .float32 ? 4:2,window=4*1024*1024/width
      for start in stride(from:0,to:flat.size,by:window) {
        try Task.checkCancellation();let slice=flat[start..<min(start+window,flat.size)]
        if width==4 { try slice.asArray(Float.self).withUnsafeBytes { try writer.append(tensor:name,bytes:$0) } }
        else { try slice.view(dtype:.uint16).asArray(UInt16.self).withUnsafeBytes { try writer.append(tensor:name,bytes:$0) } }
      }
      return Self(url:url,sha256:try writer.finish())
    }
    func load(name:String,identity:[String:String])throws->MLXArray {
      try FlashVSRCheckpoint.verifyHash(url,expected:sha256)
      let file=try SafeTensorFile(url:url)
      guard file.metadata==identity,file.tensors.count==1,let tensor=file.tensors[name],tensor.dtype=="F32" || tensor.dtype=="BF16" else { throw FlashVSRError.invalid("Native immutable feature/latent identity changed.") }
      let output=try file.withTensorBytes(named:name) { bytes in
        tensor.dtype=="F32" ? MLXArray(bytes,tensor.shape.map(Int.init),type:Float.self):MLXArray(bytes,tensor.shape.map(Int.init),type:UInt16.self).view(dtype:.bfloat16)
      }
      eval(output);try file.checkUnchanged(at:url);return output
    }
  }
  final class RGBOutput {
    private let target:URL,temporary:URL,expectedBytes:UInt64
    private var fd:Int32 = -1,bytes:UInt64=0,hash=SHA256(),sealedHash:String?
    private var published=false,created=false
    init(url:URL,expectedBytes:UInt64)throws {
      target=url;self.expectedBytes=expectedBytes
      try FileManager.default.createDirectory(at:url.deletingLastPathComponent(),withIntermediateDirectories:true)
      temporary=url.deletingLastPathComponent().appendingPathComponent("."+url.lastPathComponent+"."+UUID().uuidString+".partial")
      fd=Darwin.open(temporary.path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,S_IRUSR|S_IWUSR)
      guard fd>=0 else { throw FlashVSRError.invalid("Cannot create exclusive native RGB24 output.") };created=true
    }
    deinit { if fd>=0 { Darwin.close(fd) };if created && !published { Darwin.unlink(temporary.path) } }
    func append(_ data:UnsafeRawBufferPointer)throws {
      guard fd>=0,sealedHash==nil,UInt64(data.count)<=expectedBytes-bytes else { throw FlashVSRError.invalid("Native RGB output exceeded its frozen visible byte interval.") }
      var done=0
      while done<data.count {
        try Task.checkCancellation()
        let amount=min(data.count-done,4*1024*1024)
        let n=Darwin.write(fd,data.baseAddress!.advanced(by:done),amount)
        if n<0 && errno==EINTR { continue }
        guard n>0 else { throw FlashVSRError.invalid("Cannot write native RGB output; check destination disk space.") }
        hash.update(bufferPointer:UnsafeRawBufferPointer(start:data.baseAddress!.advanced(by:done),count:n));done += n;bytes += UInt64(n)
      }
    }
    func seal()throws {
      try Task.checkCancellation()
      var status=stat()
      guard fd>=0,sealedHash==nil,bytes==expectedBytes,fstat(fd,&status)==0,status.st_size>=0,UInt64(status.st_size)==expectedBytes,fsync(fd)==0 else { throw FlashVSRError.invalid("Native RGB output is incomplete or cannot be synchronized.") }
      let closing=fd;fd = -1
      guard Darwin.close(closing)==0 else { throw FlashVSRError.invalid("Cannot close synchronized native RGB output.") }
      sealedHash=hash.finalize().map { String(format:"%02x",$0) }.joined()
    }
    func publish()throws->String {
      try Task.checkCancellation()
      guard !published,let sealedHash,renamex_np(temporary.path,target.path,UInt32(RENAME_EXCL))==0 else { throw FlashVSRError.invalid("Cannot atomically publish native RGB output without replacing an existing file.") }
      published=true;return sealedHash
    }
  }
}
