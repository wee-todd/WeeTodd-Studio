import Foundation

/// Weight-free editor validation; the worker admits every page and tensor.
enum NativeH3FastProfile {
  static func ordinary(_ recipe:[String:Any]) throws -> [String:Any] {
    guard let fields = recipe["fasth3"] as? [String:Any],Set(fields.keys) == ["variant"],
      ["dense-v1","vsa-v1"].contains(fields["variant"] as? String ?? ""),
      let components = recipe["components"] as? [String:Any],components["task"] as? String == "t2va",
      components["fun_controlnet"] == nil,
      components["loras"] == nil || (components["loras"] as? [Any])?.isEmpty == true,
      let config = recipe["config"] as? [String:Any],
      let points = config["steps"] as? NSNumber,CFGetTypeID(points) != CFBooleanGetTypeID(),points.doubleValue == 5,
      config["sampling_method"] as? String == "euler",recipe["loras"] == nil,
      recipe["vdn"] == nil,recipe["continuation"] == nil,recipe["joint_refinement"] == nil,
      recipe["motion_fidelity"] == nil else {
      throw StudioError.invalid("FastH3 Preview v1 requires four Euler evaluations without adapters, references or continuity.")
    }
    var result = recipe;result.removeValue(forKey:"fasth3");return result
  }
  static func validate(clip:Clip,project:StudioProject,motion:NativeH3MotionPlan?,recipe:[String:Any]) throws {
    _ = try ordinary(recipe)
    guard ["t2v","t2va"].contains(clip.inferredTask),clip.continuityMode == "independent",motion == nil,
      !project.shouldSaveContinuityContext(for:clip),clip.attachments.isEmpty,
      clip.generationSelection?.steps == nil || clip.generationSelection?.steps == 4,
      clip.generationSelection?.h3SamplingMethod == nil || clip.generationSelection?.h3SamplingMethod == .euler,
      clip.generationSelection?.h3Joint == nil,clip.generationSelection?.h3MotionFidelity == nil else {
      throw StudioError.invalid("FastH3 Preview v1 supports independent T2VA with four Euler evaluations and no additional adapters or controls.")
    }
  }
}
