import MLX

/// Consume the small timestep table directly instead of gathering full-width
/// scale/shift/gate arrays. Explicit BF16 casts retain the original rounding
/// after every arithmetic operation; this is not an approximate fused FMA.
enum H3Modulation {
  private static let adaptive = MLXFast.metalKernel(
    name:"weetodd_h3_indexed_adaptive_bf16",inputNames:["x","table","indices"],
    outputNames:["output"],source:"""
      uint i=thread_position_in_grid.x;
      if(i>=ROWS*WIDTH) return;
      uint row=i/WIDTH, channel=i%WIDTH;
      int group=indices[row];
      if(group<0 || group>=GROUPS) { output[i]=bfloat(NAN); return; }
      uint base=uint(group)*6*WIDTH+channel;
      bfloat scale=bfloat(1.0f+float(table[base+SCALE*WIDTH]));
      bfloat product=bfloat(float(x[i])*float(scale));
      output[i]=bfloat(float(product)+float(table[base+SHIFT*WIDTH]));
      """)
  private static let gatedResidual = MLXFast.metalKernel(
    name:"weetodd_h3_indexed_residual_bf16",inputNames:["x","branch","table","indices"],
    outputNames:["output"],source:"""
      uint i=thread_position_in_grid.x;
      if(i>=ROWS*WIDTH) return;
      uint row=i/WIDTH, channel=i%WIDTH;
      int group=indices[row];
      if(group<0 || group>=GROUPS) { output[i]=bfloat(NAN); return; }
      bfloat product=bfloat(float(branch[i])*float(table[uint(group)*6*WIDTH+GATE*WIDTH+channel]));
      output[i]=bfloat(float(x[i])+float(product));
      """)

  static func scaleShift(_ x:MLXArray, table:MLXArray, indices:MLXArray,
    shift:Int,scale:Int) -> MLXArray {
    let width=x.shape.last!
    if x.dtype != .bfloat16 || Device.defaultDevice().deviceType != .gpu {
      let s=take(table[.ellipsis,(scale*width)..<((scale+1)*width)],indices,axis:0)
      let b=take(table[.ellipsis,(shift*width)..<((shift+1)*width)],indices,axis:0)
      return x*(1+s)+b
    }
    return adaptive([contiguous(x),contiguous(table),contiguous(indices)],
      template:[("ROWS",x.shape[1]),("WIDTH",width),("GROUPS",table.shape[0]),
        ("SHIFT",shift),("SCALE",scale)],grid:(x.size,1,1),threadGroup:(256,1,1),
      outputShapes:[x.shape],outputDTypes:[.bfloat16])[0]
  }

  static func residual(_ x:MLXArray,branch:MLXArray,table:MLXArray,
    indices:MLXArray,gate:Int) -> MLXArray {
    let width=x.shape.last!
    if x.dtype != .bfloat16 || Device.defaultDevice().deviceType != .gpu {
      return x+take(table[.ellipsis,(gate*width)..<((gate+1)*width)],indices,axis:0)*branch
    }
    return gatedResidual([contiguous(x),contiguous(branch),contiguous(table),contiguous(indices)],
      template:[("ROWS",x.shape[1]),("WIDTH",width),("GROUPS",table.shape[0]),("GATE",gate)],
      grid:(x.size,1,1),threadGroup:(256,1,1),outputShapes:[x.shape],outputDTypes:[.bfloat16])[0]
  }
}
