import Foundation

public struct FlashVSRResources: Codable, Equatable, Sendable {
  public let components: FlashVSRComponents
  public let engineFiles: [FlashVSRFile]
}

/// Resource admission happens before generation; source identity is bound only after its worker exits.
public struct NativeVideoProcessingPlan: Codable, Sendable {
  public let recipe: VideoProcessingRecipe
  public let runtime: FlashVSRRuntime
  public let resources: [String: FlashVSRResources]
  public var native: NativePixelCQResources?
  public var h3Runtime:H3RegenerateRuntime?
  public var h3Resources:[String:H3RegenerateResources]?
  public var pixelResources:[String:NativePixelCQResources]?
  public init(recipe: VideoProcessingRecipe, runtime: FlashVSRRuntime, resources: [String: FlashVSRResources], native: NativePixelCQResources? = nil, h3Runtime:H3RegenerateRuntime? = nil,
    h3Resources:[String:H3RegenerateResources]? = nil, pixelResources:[String:NativePixelCQResources]? = nil) {
    self.recipe = recipe; self.runtime = runtime; self.resources = resources; self.native = native
    self.h3Runtime=h3Runtime;self.h3Resources=h3Resources;self.pixelResources=pixelResources
  }
  public init(recipe: VideoProcessingRecipe, native: NativePixelCQResources) {
    self.recipe = recipe; self.runtime = .init(); self.resources = [:]; self.native = native
  }
  /// The same model/profile resolution is used by admission, export and execution.
  public static func pixelRuntime(_ runtime:NativePixelCQRuntime,settings:NativePixelCQSettings) throws -> NativePixelCQRuntime {
    var selected=runtime
    if settings.profileID != "auto" {
      selected=try NativePixelCQRuntime.resolve(settings:settings,profilesDirectory:URL(fileURLWithPath:settings.profileID).deletingLastPathComponent().path,
        workerPath:runtime.workerPath,ffmpegPath:runtime.ffmpegPath,ffprobePath:runtime.ffprobePath,
        cqCandidates:[runtime.cqAdapterPath].compactMap{$0},pixelCandidates:[runtime.components["pixel_spatial_adapter"]].compactMap{$0})
    }
    if let path=settings.cqAdapterPath{selected.cqAdapterPath=path}
    if let path=settings.pixelAdapterPath{selected.components["pixel_spatial_adapter"]=path}
    return selected
  }
  public static func preflight(recipe:VideoProcessingRecipe,runtime:FlashVSRRuntime?,nativeRuntime:NativePixelCQRuntime?,
    h3Runtime:H3RegenerateRuntime?,cancellation:NativeHeadlessCancellation = .init()) async throws -> Self {
    try recipe.validate()
    if !recipe.isSequence {
      if let nativeRuntime, let runtime {return try await preflight(recipe:recipe,runtime:runtime,nativeRuntime:nativeRuntime,cancellation:cancellation)}
      if let nativeRuntime,recipe.isNative {return try await preflight(recipe:recipe,nativeRuntime:nativeRuntime,cancellation:cancellation)}
      guard let runtime else {throw StudioError.invalid("The processing runtime is missing.")}
      return try await preflight(recipe:recipe,runtime:runtime)
    }
    var flash=[String:FlashVSRResources](),h3=[String:H3RegenerateResources](),pixel=[String:NativePixelCQResources]()
    for step in recipe.steps {
      try cancellation.check();try Task.checkCancellation()
      if step.isH3Regenerate {
        guard let h3Runtime,let settings=step.h3Regenerate else{throw StudioError.invalid("Connect the native H3 restoration runtime.")}
        h3[step.id]=try await H3RegenerateService().preflightResources(settings:settings,runtime:h3Runtime)
      } else if step.isNative {
        guard let nativeRuntime,let settings=step.pixelCQ else{throw StudioError.invalid("Connect the native Pixel Spatial runtime.")}
        pixel[step.id]=try await NativePixelCQService.preflight(recipe:.init(pixelCQ:settings),runtime:pixelRuntime(nativeRuntime,settings:settings),cancellation:cancellation)
      } else {
        guard let runtime else{throw StudioError.invalid("Connect the selected FlashVSR runtime.")}
        flash[step.id]=try await FlashVSRService.preflightResources(settings:step.settings,runtime:runtime)
      }
    }
    return .init(recipe:recipe,runtime:runtime ?? .init(),resources:flash,h3Runtime:h3Runtime,h3Resources:h3,pixelResources:pixel)
  }
  public static func preflight(recipe: VideoProcessingRecipe, nativeRuntime: NativePixelCQRuntime, cancellation: NativeHeadlessCancellation = .init()) async throws -> Self {
    try recipe.validate()
    if recipe.isSequence {return try await preflight(recipe:recipe,runtime:nil,nativeRuntime:nativeRuntime,h3Runtime:nil,cancellation:cancellation)}
    guard recipe.isNative else { throw StudioError.invalid("This recipe also needs the FlashVSR runtime.") }
    return .init(recipe: recipe, native: try await NativePixelCQService.preflight(recipe: recipe, runtime: nativeRuntime, cancellation: cancellation))
  }
  public static func preflight(recipe: VideoProcessingRecipe, runtime: FlashVSRRuntime,
    nativeRuntime: NativePixelCQRuntime, cancellation: NativeHeadlessCancellation = .init()) async throws -> Self {
    try recipe.validate(); try cancellation.check(); try Task.checkCancellation()
    if recipe.isSequence {return try await preflight(recipe:recipe,runtime:runtime,nativeRuntime:nativeRuntime,h3Runtime:nil,cancellation:cancellation)}
    let native: NativePixelCQResources?
    if let prefix = recipe.nativePrefix {
      native = try await NativePixelCQService.preflight(recipe: prefix, runtime: nativeRuntime, cancellation: cancellation)
    } else { native = nil }
    try cancellation.check(); try Task.checkCancellation()
    var resources: [String: FlashVSRResources] = [:]
    for step in recipe.steps where !step.isNative {
      resources[step.id] = try await FlashVSRService.preflightResources(settings: step.settings, runtime: runtime)
      try cancellation.check(); try Task.checkCancellation()
    }
    return .init(recipe: recipe, runtime: runtime, resources: resources, native: native)
  }
  public static func preflight(recipe: VideoProcessingRecipe, runtime: FlashVSRRuntime) async throws -> Self {
    try recipe.validate()
    if recipe.isSequence {return try await preflight(recipe:recipe,runtime:runtime,nativeRuntime:nil,h3Runtime:nil)}
    guard !recipe.hasNative else { throw StudioError.invalid("Native Pixel Spatial + CQ requires the native LTX worker, not the optional FlashVSR engine.") }
    var resources: [String: FlashVSRResources] = [:]
    for step in recipe.steps {
      resources[step.id] = try await FlashVSRService.preflightResources(settings: step.settings, runtime: runtime)
    }
    return Self(recipe: recipe, runtime: runtime, resources: resources)
  }
}
