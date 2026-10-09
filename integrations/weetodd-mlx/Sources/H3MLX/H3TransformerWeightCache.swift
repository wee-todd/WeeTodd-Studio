import Foundation
import Darwin
import MLX
import TensorIO

/// A fixed prefix avoids an LRU cache cycling through all 50 blocks every step.
/// The budget covers retained block matrices, norms and prepared LoRA pairs;
/// activations/temporary loads need a separate, conservative working reserve.
struct H3TransformerCachePlan {
  static let gib = 1024 * 1024 * 1024
  static let budgets = [0,8,16,32,48,64,96]
  let budgetBytes:Int
  let blockCount:Int
  let retainedBytes:Int
  let reserveBytes:Int

  static func budget(config:[String:Any]) throws -> Int {
    guard let value=config["transformer_weight_cache_gb"] else { return 0 }
    guard let number=value as? NSNumber,CFGetTypeID(number) != CFBooleanGetTypeID(),
      number.doubleValue.isFinite,number.doubleValue.rounded(.towardZero) == number.doubleValue,
      budgets.contains(number.intValue) else {
      throw H3CheckpointError.invalid("transformer_weight_cache_gb must be 0, 8, 16, 32, 48, 64 or 96 GiB.")
    }
    return number.intValue
  }
  init(budgetGB:Int,blockBytes:[Int],physicalBytes:Int,recommendedBytes:Int,availableBytes:Int) throws {
    guard Self.budgets.contains(budgetGB),(1...50).contains(blockBytes.count),
      blockBytes.allSatisfy({ $0 > 0 && $0 <= 4*Self.gib }),physicalBytes > 0,
      recommendedBytes > 0,availableBytes >= 0 else {
      throw H3CheckpointError.invalid("Invalid H3 transformer cache plan.")
    }
    budgetBytes=budgetGB*Self.gib
    reserveBytes=max(16*Self.gib,physicalBytes/8)
    // Refuse a requested budget rather than silently taking a different one.
    guard budgetGB == 0 || budgetBytes + reserveBytes <= min(physicalBytes/2,min(recommendedBytes,availableBytes)) else {
      throw H3CheckpointError.invalid("H3 transformer cache budget leaves insufficient available memory or GPU working reserve; choose a smaller budget.")
    }
    var bytes=0,count=0
    for next in blockBytes where budgetGB > 0 {
      if next > budgetBytes-bytes { break }
      bytes += next;count += 1
    }
    retainedBytes=bytes;blockCount=count
  }
  static func availableMemory() throws -> Int {
    var stats=vm_statistics64_data_t()
    var count=mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size/MemoryLayout<integer_t>.size)
    let result=withUnsafeMutablePointer(to:&stats) { pointer in
      pointer.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
        host_statistics64(mach_host_self(),HOST_VM_INFO64,$0,&count)
      }
    }
    guard result == KERN_SUCCESS else { throw H3CheckpointError.invalid("Cannot inspect available memory for H3 weight cache.") }
    return Int(UInt64(stats.free_count)+UInt64(stats.inactive_count))*Int(getpagesize())
  }
  static func inspect(checkpointURL:URL,blockCount:Int,budgetGB:Int,adapters:[H3LoRAAdapter]) throws -> Self {
    guard budgets.contains(budgetGB) else { throw H3CheckpointError.invalid("Invalid H3 transformer cache budget.") }
    try Task.checkCancellation()
    let layout=try H3CheckpointLayout(url:checkpointURL)
    guard layout.fastVariant == nil,adapters.allSatisfy({ $0.startAfterEvaluations == 0 }) else {
      throw H3CheckpointError.invalid("H3 weight cache requires ordinary weights and immediate adapters.")
    }
    let adapterFiles=try adapters.map { try SafeTensorFile(url:$0.url) }
    var sizes:[Int]=[]
    for index in 0..<blockCount {
      let url=try H3CheckpointSource.fileURL(checkpointURL,block:index)
      let file=try SafeTensorFile(url:url),prefix=layout.prefix+"blocks.\(index)."
      var bytes=0
      for (suffix,rows,columns) in [("attn.qkv_proj",21504,5376),("attn.out_proj",5376,7168),("mlp.fc1",28672,5376),("mlp.fc2",5376,14336)] {
        guard let tensor=file.tensors[prefix+suffix+".weight"],
          tensor.shape == [UInt64(rows),UInt64(columns)],
          tensor.dtype == (layout.curveRank == nil ? "I8":"BF16") else {
          throw H3CheckpointError.invalid("H3 cache requires admitted ordinary I8 or BF16 block projections.")
        }
        bytes += rows*columns*2 // cached I8 projections are decoded BF16 too.
      }
      for (suffix,width) in [("norm1.weight",5376),("norm2.weight",5376),("attn.q_norm.weight",128),("attn.k_norm.weight",128)] {
        guard let tensor=file.tensors[prefix+suffix],tensor.dtype == "BF16",tensor.shape == [UInt64(width)] else {
          throw H3CheckpointError.invalid("Invalid H3 cache normalization tensor.")
        }
        bytes += width*2
      }
      // Conservative 3x allowance includes reordered prepared pairs and lazy
      // projection bookkeeping. Other adapter targets are never cached here.
      for adapter in adapterFiles {
        for (name,tensor) in adapter.tensors where name.hasPrefix("diffusion_model.blocks.\(index).") && (name.hasSuffix(".lora_A.weight") || name.hasSuffix(".lora_B.weight")) {
          guard tensor.byteCount <= 256*1024*1024 else { throw H3CheckpointError.invalid("H3 cache adapter target exceeds budget admission.") }
          bytes += Int(tensor.byteCount)*3
        }
      }
      try file.checkUnchanged(at:url);sizes.append(bytes)
    }
    for (adapter,file) in zip(adapters,adapterFiles) { try file.checkUnchanged(at:adapter.url) }
    try H3CheckpointSource.checkUnchanged(checkpointURL)
    return try Self(budgetGB:budgetGB,blockBytes:sizes,
      physicalBytes:Int(ProcessInfo.processInfo.physicalMemory),recommendedBytes:Int(GPU.deviceInfo().maxRecommendedWorkingSetSize),
      availableBytes:availableMemory())
  }
}

/// Admit the untouched recipe before wrappers can remove unsupported controls.
public enum H3TransformerWeightCachePolicy {
  public static func admitRecipe(_ root:[String:Any]) throws -> Int {
    let config=root["config"] as? [String:Any] ?? [:]
    let budget=try H3TransformerCachePlan.budget(config:config)
    if budget == 0 { return budget }
    guard root["fasth3"] == nil,root["vdn"] == nil,root["continuation"] == nil,
      root["refinement"] == nil,root["joint_refinement"] == nil,root["joint_latents"] == nil,root["motion_fidelity"] == nil,
      (root["components"] as? [String:Any])?["fun_controlnet"] == nil,
      (config["transformer_backend"] as? String ?? "mlx") == "mlx" else {
      throw H3CheckpointError.invalid("Transformer weight cache requires ordinary MLX generation without continuation, refinement, FastH3, VDN or controls.")
    }
    let adapters=(root["loras"] as? [String:Any])?["adapters"] as? [[String:Any]] ?? []
    guard adapters.allSatisfy({ ($0["start_after_evaluations"] as? Int ?? 0) == 0 }) else {
      throw H3CheckpointError.invalid("Transformer weight cache cannot retain deferred adapters.")
    }
    return budget
  }
}

public struct H3TransformerCacheReport: Sendable,Equatable {
  public let budgetBytes:Int
  public let admittedBlocks:Int
  public let maximumRetainedBytes:Int
  public let loads:Int
  public let hits:Int
  public let released:Bool
  var metadata:[String:Any] {
    ["budgetBytes":budgetBytes,"admittedBlocks":admittedBlocks,"maximumRetainedBytes":maximumRetainedBytes,
      "loads":loads,"hits":hits,"released":released,"policy":"stage-local fixed prefix; rest streamed",
      "scope":"retained transformer block weights and prepared adapter pairs; excludes activations and decoder"]
  }
}

final class H3TransformerWeightCache {
  let plan:H3TransformerCachePlan
  private let checkpointURL:URL
  private let useMPP:Bool
  private let verificationScope:String
  private var owners:[Int:H3PreparedBlock]=[:]
  private(set) var hits=0
  private(set) var loads=0
  private(set) var maximumRetainedBytes=0
  private(set) var isClosed=false
  var storageBytes:Int { owners.values.reduce(0) { $0+$1.storageBytes } }
  var report:H3TransformerCacheReport { .init(budgetBytes:plan.budgetBytes,admittedBlocks:plan.blockCount,
    maximumRetainedBytes:max(maximumRetainedBytes,storageBytes),loads:loads,hits:hits,released:isClosed) }
  init(checkpointURL:URL,blockCount:Int,budgetGB:Int,adapters:[H3LoRAAdapter],useMPP:Bool,verificationScope:String) throws {
    plan=try H3TransformerCachePlan.inspect(checkpointURL:checkpointURL,blockCount:blockCount,budgetGB:budgetGB,adapters:adapters)
    self.checkpointURL=checkpointURL;self.useMPP=useMPP;self.verificationScope=verificationScope
  }
  func weights(for index:Int) throws -> H3PreparedBlock {
    do {
      try Task.checkCancellation()
      guard !isClosed,(0..<plan.blockCount).contains(index) else { throw H3CheckpointError.invalid("H3 weight cache request is outside its admitted prefix.") }
      if let owner=owners[index] { try owner.checkUnchanged();hits += 1;return owner }
      guard try H3TransformerCachePlan.availableMemory() >= plan.reserveBytes+1024*1024*1024 else {
        throw H3CheckpointError.invalid("Available memory fell below the H3 transformer cache reserve.")
      }
      let owner=try H3PreparedBlock(checkpointURL:checkpointURL,index:index,projectionMode:.weightDecoded,
        useMPP:useMPP,verificationScope:verificationScope,retainProjections:true)
      owners[index]=owner;loads += 1
      try checkBudget();return owner
    } catch { close();throw error }
  }
  func checkBudget() throws {
    maximumRetainedBytes=max(maximumRetainedBytes,storageBytes)
    guard storageBytes <= plan.budgetBytes else { close();throw H3CheckpointError.invalid("H3 retained weights exceeded their admitted cache budget.") }
  }
  func close() {
    guard !isClosed else { return }
    maximumRetainedBytes=max(maximumRetainedBytes,storageBytes)
    for index in owners.keys.sorted() { owners[index]?.close() }
    owners.removeAll();isClosed=true
  }
  deinit { close() }
}
