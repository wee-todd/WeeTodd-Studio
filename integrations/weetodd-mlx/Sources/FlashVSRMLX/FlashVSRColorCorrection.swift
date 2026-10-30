import MLX

/// Correction runs before encoding on matched RGB [frames,height,width,3].
/// Wavelet follows the released replicate-border Gaussian pyramid. AdaIN is an
/// explicit native option using per-frame/channel population variance + epsilon.
public enum FlashVSRColorCorrection {
  public enum Mode:Sendable,Equatable { case off,wavelet(levels:Int),adain(epsilon:Float) }
  public enum Range:Sendable,Equatable { case minusOneToOne,zeroToOne }
  public static func apply(_ generated:MLXArray,reference:MLXArray,mode:Mode,
    range:Range = .minusOneToOne)throws->MLXArray {
    guard generated.ndim==4,generated.shape==reference.shape,generated.shape[0]>0,
      generated.shape[1]>0,generated.shape[2]>0,generated.shape[3]==3,
      generated.dtype.isFloatingPoint,reference.dtype.isFloatingPoint else {
      throw FlashVSRError.invalid("Color correction requires matched floating RGB frame groups [F,H,W,3].")
    }
    if mode == .off { return generated }
    let g=generated.asType(.float32),r=reference.asType(.float32)
    let content=range == .zeroToOne ? g*2-1:g,style=range == .zeroToOne ? r*2-1:r
    let corrected:MLXArray
    switch mode {
    case .off: return generated
    case .wavelet(let levels):
      guard (1...8).contains(levels) else { throw FlashVSRError.invalid("Wavelet correction needs 1...8 Gaussian levels.") }
      var lowContent=content,lowStyle=style
      for level in 0..<levels {
        try Task.checkCancellation()
        lowContent=blur(lowContent,radius:1<<level);lowStyle=blur(lowStyle,radius:1<<level)
        eval(lowContent,lowStyle)
      }
      corrected=content-lowContent+lowStyle
    case .adain(let epsilon):
      guard epsilon.isFinite,epsilon>0 else { throw FlashVSRError.invalid("AdaIN epsilon must be finite and positive.") }
      let cm=mean(content,axes:[1,2],keepDims:true),sm=mean(style,axes:[1,2],keepDims:true)
      let centeredContent=content-cm,centeredStyle=style-sm
      let cv=mean(centeredContent*centeredContent,axes:[1,2],keepDims:true),sv=mean(centeredStyle*centeredStyle,axes:[1,2],keepDims:true)
      corrected=centeredContent/sqrt(cv+epsilon)*sqrt(sv+epsilon)+sm
    }
    let clipped=clip(corrected,min:-1,max:1),result=(range == .zeroToOne ? (clipped+1)/2:clipped).asType(generated.dtype)
    eval(result);try Task.checkCancellation();return result
  }
  private static func blur(_ x:MLXArray,radius:Int)->MLXArray {
    let h=x.shape[1],w=x.shape[2]
    let rows=MLXArray(((-radius)..<(h+radius)).map { Int32(Swift.max(0,Swift.min(h-1,$0))) })
    let cols=MLXArray(((-radius)..<(w+radius)).map { Int32(Swift.max(0,Swift.min(w-1,$0))) })
    let expanded=take(take(x,rows,axis:1),cols,axis:2)
    let kernel=MLXArray([Float(0.0625),0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625],[1,3,3,1])
    return conv2d(expanded,repeated(kernel,count:3,axis:0),dilation:IntOrPair(radius),groups:3)
  }
}
