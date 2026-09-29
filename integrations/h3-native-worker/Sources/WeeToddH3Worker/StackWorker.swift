import Foundation
import NNC
import Metal
extension H3WorkerMain {
 static func serveStack(model:Model?,graph:DynamicGraph,stream:StreamContext,inputs:inout [DynamicGraph.AnyTensor],rows:Int,tableRows:Int,half:Bool,loadSeconds:Double,start:Int,count:Int,modulationLayout:ModulationSpans? = nil,bounded:BlockResidentStack? = nil) throws {
  func emit(_ value:[String:Any]) throws { let data=try JSONSerialization.data(withJSONObject:value,options:.sortedKeys);FileHandle.standardOutput.write(data);FileHandle.standardOutput.write(Data([10])) }
  try emit(["protocol":1,"qkv_schedule":ProcessInfo.processInfo.environment["WEETODD_NNC_QKV_SCHEDULE"] ?? "parallel","buffer_io":boundedHostIO ? "bounded" : "copied","weight_prefetch":bounded?.prefetchEnabled ?? false,"prefetch_slot_capacity":bounded?.prefetchEnabled == true ? 1 : 0,"residency":bounded == nil ? "eager" : "block","resident_blocks":bounded?.residentBlocks ?? count,"metal_allocated_bytes":MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0,"event":"ready","projections":ProcessInfo.processInfo.environment["WEETODD_NNC_PROJECTIONS"] ?? "fused","modulation_spans":modulationLayout?.ranges.count ?? 0,"rows":rows,"precision":half ? "fp16" : "bf16","load_seconds":loadSeconds,"start":start,"count":count])
  let reader=BoundedLineReader()
  while let line=try reader.next() {
   guard line.utf8.count<=65536,let raw=line.data(using:.utf8),let command=try JSONSerialization.jsonObject(with:raw) as? [String:Any],let op=command["op"] as? String,let id=command["id"] as? String,id.utf8.count<=128 else { throw ProbeError.invalid("Invalid worker command") }
   if op=="unload",let bounded=bounded {
    stream.joined();inputs[0]=graph.variable(Tensor<Float>([0],kind:.CPU,format:.NHWC,shape:[1]).toGPU(0));let released=bounded.unload()
    try emit(["event":"unloaded","id":id,"packed_input_rows":inputs[0].shape[0],"resident_blocks":bounded.residentBlocks,"model_released":released,"metal_allocated_bytes":MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0]);continue
   }
   if op=="close" { try emit(["event":"closed","id":id]);return }
   guard op=="predict",let fixture=command["input"] as? String,let output=command["output"] as? String else { throw ProbeError.invalid("Unsupported worker command") }
   let data=try SafeTensorReader(fixture)
   guard Set(data.records.keys)==Set(["x","indices"]),data.records["x"]?.shape==[rows,5376],data.records["indices"]?.shape==[rows] else { throw ProbeError.invalid("Invalid worker shapes") }
   let x=try data.read("x"),indices=try data.read("indices")
   guard x.values.allSatisfy({$0.isFinite}),indices.values.allSatisfy({$0.isFinite && $0>=0 && $0<Float(tableRows) && $0.rounded()==$0}) else { throw ProbeError.invalid("Invalid worker values") }
   try modulationLayout?.validate(indices.values.map{Int32($0)})
   if half { inputs[0]=graph.variable(hostTensor(x.values,shape:[rows,5376]).toGPU(0)) }
   else { inputs[0]=graph.variable(makeBF16(x).toGPU(0)) }
   inputs[1]=graph.variable(Tensor<Int32>(indices.values.map{Int32($0)},kind:.CPU,format:.NHWC,shape:[rows]).toGPU(0))
   let target=URL(fileURLWithPath:output)
   guard !FileManager.default.fileExists(atPath:target.path) else { throw ProbeError.invalid("Worker output already exists") }
   let began=DispatchTime.now().uptimeNanoseconds
   var seconds:Double=0
   try graph.withNoGrad {
    let results=try bounded.map { try $0.predict(inputs:inputs,stream:stream) } ?? model!(inputs:inputs[0],Array(inputs.dropFirst()),streamContext:stream);stream.joined()
    seconds=Double(DispatchTime.now().uptimeNanoseconds-began)/1e9
    guard let result=results.last else { throw ProbeError.invalid("Missing worker result") }
    if half {
     let cpu=DynamicGraph.Tensor<Float>(result).toCPU().rawValue
     try cpu.withUnsafeBytes { bytes in
      guard bytes.bindMemory(to:Float.self).allSatisfy({$0.isFinite}) else { throw ProbeError.invalid("Nonfinite worker output") }
      try writeTensorBytes(bytes,to:target)
     }
    } else {
     let cpu=DynamicGraph.Tensor<BFloat16>(result).toCPU().rawValue
     try cpu.withUnsafeBytes { bytes in
      guard bytes.bindMemory(to:UInt16.self).allSatisfy({($0 & 0x7f80) != 0x7f80}) else { throw ProbeError.invalid("Nonfinite worker output") }
      try writeTensorBytes(bytes,to:target)
     }
    }
   }
   graph.garbageCollect()
   try emit(["weight_preparation_seconds":bounded?.preparationSeconds ?? 0,"resident_blocks":bounded?.residentBlocks ?? count,"weight_load_seconds":bounded?.loadSeconds ?? 0,"compute_seconds":bounded?.computeSeconds ?? seconds,"event":"prediction","id":id,"seconds":seconds,"output":output,"dtype":half ? "F32" : "BF16","metal_allocated_bytes":MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0])
  }
 }
}
