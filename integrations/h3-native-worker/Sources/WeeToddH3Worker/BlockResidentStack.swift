// One reusable native block slot. Original safetensors remain the source of truth.
import Foundation
import NNC
import Metal

final class BlockResidentStack {
  let graph:DynamicGraph
  let weights:SafeTensorReader
  let adapter:SafeTensorReader
  let rows:Int
  let start:Int
  let count:Int
  let prefetchEnabled:Bool
  private(set) var preparationSeconds:Double=0
  private var block:H3Block?
  private(set) var residentBlocks=0
  private(set) var loadSeconds:Double=0
  private(set) var computeSeconds:Double=0

  init(graph:DynamicGraph,weights:SafeTensorReader,adapter:SafeTensorReader,rows:Int,start:Int,count:Int) throws {
    let env=ProcessInfo.processInfo.environment
    guard env["WEETODD_NNC_PRECISION"]=="fp16",env["WEETODD_NNC_PROJECTIONS"]=="input-scaled",env["WEETODD_NNC_ATTENTION_ACCUMULATION"]=="fp32",env["WEETODD_NNC_FFN_POLICY"]=="layer-scaled",env["WEETODD_NNC_MODULATION"] != "spans",env["WEETODD_NNC_WEIGHT_STORAGE"] ?? "dense" == "dense" else { throw ProbeError.invalid("Block residency requires qualified FP16 projections, FP32 attention, layer scaling and ordinary gathers") }
    guard ["0","1"].contains(env["WEETODD_NNC_PREFETCH"] ?? "0") else { throw ProbeError.invalid("Unknown weight prefetch policy") }
    self.prefetchEnabled=env["WEETODD_NNC_PREFETCH"]=="1"
    self.graph=graph;self.weights=weights;self.adapter=adapter;self.rows=rows;self.start=start;self.count=count
    let specification=try H3Block(rows:rows,diagnostic:false,adapter:adapter,half:true,blockIndex:start,ffnScale:layerFFNScale(start),reusable:true)
    for index in start..<(start+count) { try specification.validateReusableSource(index:index,checkpoint:weights,adapter:adapter) }
  }

  private func arguments(_ inputs:[DynamicGraph.AnyTensor],x:DynamicGraph.AnyTensor,offset:Int) -> [DynamicGraph.AnyTensor] {
    [x,inputs[1]]+Array(inputs[(4+offset*6)..<(10+offset*6)])+[inputs[2],inputs[3]]
  }

  func predict(inputs:[DynamicGraph.AnyTensor],stream:StreamContext) throws -> [DynamicGraph.AnyTensor] {
    guard inputs.count==4+count*6 else { throw ProbeError.invalid("Invalid block residency inputs") }
    loadSeconds=0;computeSeconds=0
    preparationSeconds=0
    let prefetch=prefetchEnabled ? WeightPrefetch() : nil
    defer { prefetch?.drain() }
    var x=inputs[0]
    for offset in 0..<count {
      let index=start+offset
      let args=arguments(inputs,x:x,offset:offset)
      if block==nil {
        let candidate=try H3Block(rows:rows,diagnostic:false,adapter:adapter,half:true,blockIndex:start,ffnScale:layerFFNScale(start),reusable:true)
        candidate.model.maxConcurrency = .limit(4)
        candidate.model.compile(inputs:args)
        block=candidate
      }
      let loading=Date()
      // Previous GPU use has completed before any weight buffer is overwritten.
      if let prefetch=prefetch {
        if offset==0 { try prefetch.submit(index:index,block:block!,checkpoint:weights,adapter:adapter) }
        try block!.installPrepared(try prefetch.take(),checkpoint:weights,adapter:adapter)
        preparationSeconds=prefetch.preparationSeconds
      } else { try block!.load(checkpoint:weights,adapter:adapter,graph:graph,sourceIndex:index) }
      loadSeconds += Date().timeIntervalSince(loading);residentBlocks=1
      if let prefetch=prefetch,offset+1<count {
        try prefetch.submit(index:index+1,block:block!,checkpoint:weights,adapter:adapter)
      }
      let computing=Date()
      let result=block!.model(inputs:args[0],Array(args.dropFirst()),streamContext:stream)
      stream.joined()
      guard let next=result.last else { throw ProbeError.invalid("Missing resident block result") }
      x=next;computeSeconds += Date().timeIntervalSince(computing)
      if ProcessInfo.processInfo.environment["WEETODD_NNC_PROGRESS"]=="1" {
        let event:[String:Any]=["model_scratch_bytes":block!.model.runtimeMemorySize,"cpu_prefetch_blocks":prefetch?.pending == true ? 1 : 0,"weight_preparation_seconds":preparationSeconds,"event":"progress","completed":offset+1,"total":count,"resident_blocks":residentBlocks,"weight_load_seconds":loadSeconds,"compute_seconds":computeSeconds,"metal_allocated_bytes":MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0]
        let data=try JSONSerialization.data(withJSONObject:event);FileHandle.standardOutput.write(data);FileHandle.standardOutput.write(Data([10]))
      }
    }
    return [x]
  }

  @discardableResult func unload() -> Bool {
    weak let previous=block?.model
    block=nil;residentBlocks=0
    graph.garbageCollect()
    return previous==nil
  }
}
