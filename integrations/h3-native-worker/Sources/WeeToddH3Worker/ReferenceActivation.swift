import NNC
func bf16ReferenceSwish(_ x:ModelIOConvertible) -> Model.IO {
 func rounded(_ v:ModelIOConvertible) -> Model.IO { v.to(.BFloat16).to(.Float32) }
 let wide=x.to(.Float32)
 let pos=wide.clamped(0...),neg=(-1 * wide).clamped(0...)
 let magnitude=pos + neg
 let exponential=rounded(magnitude.exp())
 let denominator=rounded(1 + exponential)
 let tail=rounded(denominator.reciprocal())
 let complement=rounded(1 + (-1 * tail))
 let positive=rounded(pos .* complement)
 let negative=rounded(neg .* tail)
 return (positive + (-1 * negative)).to(.BFloat16)
}
