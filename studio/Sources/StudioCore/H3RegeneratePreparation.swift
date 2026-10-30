import CryptoKit
import Darwin
import Foundation
import InferenceContracts
import TensorIO

public enum H3RegeneratePreparation {
  static let supportedOriginalBaseSHA256:Set<String>=["600307eba6a67388b333aa5a94537a9ea83da65a6f37664ba22cf079e40a1f3b","37488d7799577d40db8e74318311344cb909c38632c0f693c52976a5509811ff"]
  struct Geometry { let sourceWidth:Int,sourceHeight:Int,frames:Int,paddedFrames:Int,temporaryBytes:Int64 }
  static func hash(_ data:Data)->String { SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined() }
  static func installedAsset(root:URL,relativePaths:[String]) throws -> URL {
    for relative in relativePaths {
      let candidate=root.appendingPathComponent(relative)
      var status=stat()
      if lstat(candidate.path,&status) != 0 {
        guard errno==ENOENT else{throw StudioError.invalid("Cannot inspect the selected H3 asset: "+relative)}
        continue
      }
      let resolved=candidate.resolvingSymlinksInPath()
      guard try resolved.resourceValues(forKeys:[.isRegularFileKey]).isRegularFile==true else{
        throw StudioError.invalid("The preferred H3 asset is not a regular file: "+relative)
      }
      return resolved
    }
    throw StudioError.invalid("The selected model library is missing H3 asset: "+(relativePaths.first ?? "unknown"))
  }
  static func boundedJSON(_ url:URL,limit:Int=1024*1024) throws -> [String:Any] {
    let info=try url.resourceValues(forKeys:[.isRegularFileKey,.fileSizeKey])
    guard info.isRegularFile==true,let bytes=info.fileSize,(1...limit).contains(bytes) else { throw StudioError.invalid("H3 profile/manifest must be bounded regular JSON.") }
    let handle=try FileHandle(forReadingFrom:url);defer{try? handle.close()}
    let data=try handle.read(upToCount:limit+1) ?? Data()
    guard data.count==bytes,let object=try JSONSerialization.jsonObject(with:data) as? [String:Any] else { throw StudioError.invalid("Invalid H3 profile/manifest JSON.") }
    return object
  }
  static func profileComponents(_ profile:[String:Any]) throws -> [String:String] {
    let forbidden=["fasth3","fasth3_omni","vdn","loras","joint_refinement","motion_fidelity","continuation"]
    guard profile["format"] as? String=="weetodd-headless-v2",profile["engine"] as? String=="h3",
      forbidden.allSatisfy({profile[$0]==nil}),let components=profile["components"] as? [String:Any],
      components["task"] as? String=="ref2va",components["fun_controlnet"]==nil,
      components["loras"]==nil || (components["loras"] as? [Any])?.isEmpty==true else {
      throw StudioError.invalid("H3 Regenerate requires an ordinary original Ref2VA profile without Fast, VDN, Omni, control, Turbo or other adapters.")
    }
    var result=[String:String]()
    for name in ["transformer","text_encoder","tokenizer","video_vae","audio_vae"] {
      guard let path=components[name] as? String,path.hasPrefix("/"),path.utf8.count<=4096,!path.utf8.contains(0) else {
        throw StudioError.invalid("The ordinary H3 profile is missing its existing " + name + " component.")
      }
      result[name]=path
    }
    return result
  }
  static func geometry(media:FlashVSRMediaInfo,settings:H3RegenerateSettings,firstFrame:Int,frameCount:Int?,latent:H3RegenerateLatentSource?=nil) throws -> Geometry {
    try settings.validate();try settings.validateSourceClock(numerator:media.fps.numerator,denominator:media.fps.denominator)
    let frames=frameCount ?? media.frames-firstFrame
    guard firstFrame>=0,frames>0,frames<=362,firstFrame<=media.frames-frames else { throw StudioError.invalid("The H3 restoration interval exceeds the source or 362-frame limit.") }
    let padded=max(5,5+((max(0,frames-5)+16)/17)*17)
    let w=latent?.width ?? (media.width+31)/32*32,h=latent?.height ?? (media.height+31)/32*32
    let tw=(settings.targetWidth+31)/32*32,th=(settings.targetHeight+31)/32*32
    guard [w,h].allSatisfy({(32...4096).contains($0)}),latent != nil || w*h<=1024*1024,
      settings.targetWidth>=w,settings.targetHeight>=h,
      settings.latentUpscaleMode != .directPublisher || (tw<=2*w && th<=2*h) else {
      throw StudioError.invalid("H3 source padding must fit one megapixel; target axes cannot shrink. Direct publisher latent upscaling is limited to two times each axis; choose explicit multihop for larger targets.")
    }
    let latentFrames=2+5*((padded-5)/17),rows=latentFrames*(tw/32)*(th/32)
    // The authoritative worker also includes exact tokenized text rows. This
    // conservative source-side check never relaxes its high-row execution gate.
    guard rows+2*Int(ceil(Double(padded)/24*40))+4096<=390000 else { throw StudioError.invalid("H3 target exceeds the packed-row indexing bound.") }
    if settings.decoder == .fastH3 {
      let tokens=latentFrames*(tw/16)*(th/16)
      guard tokens<=1_048_575, tokens<=Int(Int32.max)/3072 else { throw StudioError.invalid("Fast H3 decoder target exceeds its measured whole-context indexing bound. Select the released decoder explicitly.") }
    }
    let disk=Int64(tw)*Int64(th)*Int64(padded)*18+Int64(w)*Int64(h)*Int64(frames)*6+Int64(rows)*96*4*3+1024*1024*1024
    return Geometry(sourceWidth:w,sourceHeight:h,frames:frames,paddedFrames:padded,temporaryBytes:disk)
  }
  static func validateLatentSelection(_ latent:H3RegenerateLatentSource?,settings:H3RegenerateSettings,sourceSHA256:String,firstFrame:Int,frames:Int) throws {
    guard let latent else{return}
    guard settings.sourceMode == .nativeLatentsWhenAvailable else { throw StudioError.invalid("Explicit native latents conflict with the selected movie source mode. No fallback was applied.") }
    try latent.validate(sourceSHA256:sourceSHA256,firstFrame:firstFrame,frames:frames)
  }
  static func resources(settings:H3RegenerateSettings,runtime:H3RegenerateRuntime,cancellation:NativeHeadlessCancellation) throws -> H3RegenerateResources {
    try settings.validate();let resolved=try runtime.resolved()
    guard !settings.baseProfilePath.isEmpty,!settings.modelLibraryRoot.isEmpty else { throw StudioError.invalid("Select an existing ordinary native H3 Ref2VA profile and model library.") }
    let profileURL=URL(fileURLWithPath:settings.baseProfilePath),paths=try profileComponents(boundedJSON(profileURL))
    let root=URL(fileURLWithPath:settings.modelLibraryRoot).resolvingSymlinksInPath()
    guard try root.resourceValues(forKeys:[.isDirectoryKey]).isDirectory==true else { throw StudioError.invalid("Reconnect the selected model library.") }
    let profile=try FlashVSRFile.capture(profileURL)
    var modelFiles=[FlashVSRFile]()
    func capture(_ path:String,pin:String?=nil) throws -> FlashVSRFile {
        try cancellation.check();let file=try FlashVSRFile.capture(URL(fileURLWithPath:path))
        guard pin==nil || file.sha256==pin else { throw StudioError.invalid("H3 source identity is not the selected supported original release: " + URL(fileURLWithPath:path).lastPathComponent) }
        modelFiles.append(file);return file
    }
    func component(_ path:String,_ role:NativeComponentRole,_ layout:String,_ pin:String?=nil) throws -> H3RegenerateJob.Component {
        let file=try capture(path,pin:pin)
        return .init(path:file.path,identity:.init(family:.h3,role:role,sourceSHA256:file.sha256,layout:layout,
          importerVersion:layout=="h3-encoder-f32-odhwi-v1" ? layout:nil,
          timestepBasis:[.sharpCurve,.sharpInput].contains(role) ? .sharp1025x8:nil,
          qkvLayout:role == .refiner ? "split-diffusers":nil,swigluOrder:role == .refiner ? "gate-up":nil,scaleMode:role == .refiner ? "folded-unit":nil))
    }
    let transformer=try component(paths["transformer"]!,.transformer,"comfy-int8-convrot-decoded-bf16-v1")
    guard supportedOriginalBaseSHA256.contains(transformer.identity.sourceSHA256) else {throw StudioError.invalid("The selected H3 base is not an admitted original Ref2VA preparation. Use native setup for the pinned source; fused Turbo is unsupported.")}
    let folder=root.appendingPathComponent("WeeTodd/LoRAs/H3/Open-H3-Regenerate-2K")
    func adapterPath(_ filename:String) throws -> String {
      try installedAsset(root:root,relativePaths:["WeeTodd/LoRAs/H3/Open-H3-Regenerate-2K/"+filename,"WeeTodd/LoRAs/H3/h3-regenerate-publisher/lora/"+filename]).path
    }
    let refiner=try component(adapterPath("open_h3_regenerate_2k_lora.safetensors"),.refiner,"open-h3-official-f32-r576","8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04")
    let sharp=try component(adapterPath("sharp_curve_branches.safetensors"),.sharpCurve,"sharp-f32-r64","26ff1c4ea50fc5b0cea252912b67ec308a65d534cef026ef56f415ae92102e4e")
    let table=try component(folder.appendingPathComponent("adaln_t_table.bin").path,.sharpInput,"raw-f32-1025x8",NativeUpscalerSetup.sharpTableSHA256)
    let upscalePath=try installedAsset(root:root,relativePaths:["WeeTodd/H3-Upscalers/minimax_h3_latent_upscaler_3d_bf16.safetensors","WeeTodd/LoRAs/H3/h3-regenerate-publisher/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"]).path
    let upscale=try component(upscalePath,.latentUpscaler,"h3-latent-upscaler-3d-bf16","4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6")
    let video=try component(paths["video_vae"]!,.videoVAE,"h3-native-odhwi-affine-q8","53544746b0e1bc70298574ba549d1c97495c31a446343c8e5a638620b2b5b87f")
    let audio=try component(paths["audio_vae"]!,.audioVAE,"h3-native-dac-f32","8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48")
    let encoderPath=settings.publisherEncoderPath.isEmpty ? root.appendingPathComponent("WeeTodd/H3-Video-VAE/publisher_encoder_f32_odhwi_v1.safetensors").path:settings.publisherEncoderPath
    let encoder=settings.encoderProfile == .publisherPosteriorFP32 ? try component(encoderPath,.videoVAE,"h3-encoder-f32-odhwi-v1","d1be9c027ce656f6aff72fc225899ed7d288a32cbd92c0d93977c8da7671306c"):nil
    let fastPath=settings.decoder != .fastH3 ? "" : settings.fastDecoderPath.isEmpty
      ? try installedAsset(root:root,relativePaths:["WeeTodd/H3-Video-VAE/fast_h3_vae_noncausal_24l.safetensors","WeeTodd/H3-Video-VAE/h3-fast-decoder/vae_fast/fast_h3_vae_noncausal_24l.safetensors"]).path:settings.fastDecoderPath
    let fast=settings.decoder == .fastH3 ? try component(fastPath,.decoder,"fast-h3-noncausal-24l","8658f1e85b5997c4ba716406a2f148bfa36cf97d4d5fcd402c7bd341f54e77a1"):nil
    let textURL=URL(fileURLWithPath:paths["text_encoder"]!).resolvingSymlinksInPath()
    let directory=try textURL.resourceValues(forKeys:[.isDirectoryKey]).isDirectory==true
    let manifestURL=textURL.appendingPathComponent("paged_text_encoder_manifest.json")
    var textFiles=[H3RegenerateJob.File]()
    if directory,FileManager.default.fileExists(atPath:manifestURL.path) {
      let manifest=try boundedJSON(manifestURL,limit:256*1024)
      guard ["weetodd-h3-qwen-paged-v1","weetodd-h3-qwen-paged-v2"].contains(manifest["format"] as? String ?? ""),
        manifest["num_layers"] as? Int==50,let fixed=manifest["fixed"] as? [String:Any],
        let layers=manifest["layers"] as? [[String:Any]],layers.count==50 else { throw StudioError.invalid("H3 requires the original complete 50-layer Qwen export.") }
      let vision=manifest["vision"] as? [String:Any]
      guard (manifest["format"] as? String=="weetodd-h3-qwen-paged-v2")==((vision != nil)) else { throw StudioError.invalid("Qwen vision manifest differs from its format.") }
      let pages=[fixed]+layers+[vision].compactMap{$0}
      let names=["pages/fixed.safetensors"]+(0..<50).map{String(format:"pages/layer-%03d.safetensors",$0)}+(vision == nil ? []:["pages/vision.safetensors"])
      for (page,name) in zip(pages,names) {
        guard page["file"] as? String==name,let sha=page["sha256"] as? String,sha.count==64 else { throw StudioError.invalid("Invalid Qwen page identity.") }
        let file=try capture(textURL.appendingPathComponent(name).path,pin:sha)
        textFiles.append(.init(path:file.path,sha256:file.sha256))
      }
      let file=try capture(manifestURL.path);textFiles.append(.init(path:file.path,sha256:file.sha256))
    } else {
      let file=try capture(directory ? textURL.appendingPathComponent("text_encoder.safetensors").path:textURL.path)
      let tensor=try SafeTensorFile(url:URL(fileURLWithPath:file.path))
      guard tensor.tensors["model.embed_tokens.weight"]?.shape == [151936,5120] else { throw StudioError.invalid("The H3 text encoder has an incompatible embedding.") }
      textFiles=[.init(path:file.path,sha256:file.sha256)]
    }
    let tokenURL=URL(fileURLWithPath:paths["tokenizer"]!)
    let tokenDirectory=try tokenURL.resourceValues(forKeys:[.isDirectoryKey]).isDirectory==true
    let tokenizer=try capture(tokenDirectory ? tokenURL.appendingPathComponent("tokenizer.json").path:tokenURL.path)
    guard tokenizer.bytes<=32*1024*1024 else { throw StudioError.invalid("H3 tokenizer exceeds its bounded JSON contract.") }
    let engine=try [resolved.workerPath,resolved.ffmpegPath,resolved.ffprobePath,NativeMLXMetalLibrary.resolve(executable:URL(fileURLWithPath:resolved.workerPath)).path].map{try FlashVSRFile.capture(URL(fileURLWithPath:$0))}
    try profile.verify();try cancellation.check()
    return .init(components:.init(transformer:transformer,refiner:refiner,sharpCurve:sharp,sharpInput:table,latentUpscaler:upscale,videoVAE:video,audioVAE:audio,publisherEncoder:encoder,fastDecoder:fast),
      qwen:.init(checkpointRoot:textURL.path,tokenizer:.init(path:tokenizer.path,sha256:tokenizer.sha256),files:textFiles),modelFiles:modelFiles,engineFiles:engine,profile:profile,runtime:resolved)
  }
}
