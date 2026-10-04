import CoreFoundation
import Foundation

/// Adoption uses the actual full source interval, never expanded sampling duration.
public enum NativeH3MotionFidelityAcceptance {
  public static func duration(prepared:[String:Any]?,result:[String:Any],media:[String:Any]) throws -> Double? {
    let metadata=result["metadata"] as? [String:Any] ?? [:]
    guard prepared?["task"] as? String == "motion_fidelity" else {
      guard metadata["task"] as? String != "motion_fidelity" else { throw StudioError.invalid("Unexpected H3 Motion Fidelity result without its prepared source contract.") };return nil
    }
    func failed()->StudioError { .invalid("H3 Motion Fidelity output differs from its frozen source interval, recovered frames, canvas or native audio publication.") }
    func numeric(_ value:Any?) throws -> Double {
      guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite else { throw failed() };return n.doubleValue
    }
    func integer(_ value:Any?) throws -> Int {
      let value=try numeric(value);guard value.rounded()==value,abs(value)<Double(Int.max) else { throw failed() };return Int(value)
    }
    func boolean(_ value:Any?) throws -> Bool {
      guard let n=value as? NSNumber,CFGetTypeID(n)==CFBooleanGetTypeID() else { throw failed() };return n.boolValue
    }
    guard let frozen=prepared?["motion_source"] as? [String:Any],
      let motion=metadata["motionFidelity"] as? [String:Any],try integer(motion["version"])==1,
      result["nativeRuntime"] as? String == "swift-mlx",metadata["nativeRuntime"] as? String == "swift-mlx",
      metadata["task"] as? String == "motion_fidelity",try boolean(result["use_complete_duration"]),
      let path=frozen["path"] as? String,motion["sourcePath"] as? String==path,
      let digest=frozen["sha256"] as? String,H3JointLatentArtifact.validSHA(digest),motion["sourceSHA256"] as? String==digest,
      try boolean(motion["sourceMediaInspected"]),try boolean(motion["sourceAudioPreserved"]) else { throw failed() }
    let frames=try integer(frozen["sourceFrames"]),width=try integer(frozen["width"]),height=try integer(frozen["height"])
    let start=try numeric(frozen["sourceStartSeconds"]),seconds=Double(frames)/24
    guard (60...345).contains(frames),(32...2048).contains(width),(32...2048).contains(height),width%32==0,height%32==0,width*height<=1376*768,
      try numeric(frozen["sourceDurationSeconds"])==seconds,start>=0,start<=86400,
      abs(start*24-(start*24).rounded())<=0.001,
      try integer(motion["sourceFrames"])==frames,try numeric(motion["sourceDuration"])==seconds,
      try numeric(motion["sourceIn"])==start,try integer(metadata["frames"])==frames,try numeric(metadata["fps"])==24,
      try numeric(result["usable_source_in"])==0,abs(try numeric(result["usable_duration"])-seconds)<=1e-7,
      abs(try numeric(media["fps"])-24)<=0.001,try integer(media["width"])==width,try integer(media["height"])==height,
      abs(try numeric(media["videoDuration"])-seconds)<=0.05/24,
      try numeric(media["duration"])+0.05/24>=seconds else { throw failed() }
    if let actualFrames=media["frames"] { guard try integer(actualFrames)==frames else { throw failed() } }
    let noop=try boolean(motion["noop"]),sampled=try integer(metadata["sampledFrames"]),evaluations=try integer(motion["actualSamplingEvaluations"])
    let padded=try integer(motion["paddedFrames"]),expanded=try integer(motion["expandedFrames"])
    guard (frames...345).contains(padded),padded%17==5,(frames...padded).contains(expanded),
      let recovery=motion["recovery"] as? [Any],recovery.count==frames else { throw failed() }
    let indices=try recovery.map(integer)
    guard indices.allSatisfy({ (0..<padded).contains($0) }),zip(indices,indices.dropFirst()).allSatisfy({$0.1>$0.0}),
      noop ? (sampled==0 && evaluations==0 && expanded==frames) : (sampled==padded && (1...99).contains(evaluations)),
      try integer(metadata["audioSampleRate"])==32000,
      try integer(metadata["audioSamplesPerChannel"])==Int((seconds*32000).rounded(.toNearestOrEven)) else { throw failed() }
    return seconds
  }
}
