#include <metal_stdlib>
using namespace metal;

struct EulerParameters {
  float sampleScale;
  float predictionScale;
  float noiseScale;
  uint count;
  uint channels;
  uint flags;
  float velocitySigma;
};

kernel void ltx25_euler(
  device const float* sample [[buffer(0)]],
  device const float* prediction [[buffer(1)]],
  device const float* noise [[buffer(2)]],
  device const float* clean [[buffer(3)]],
  device const float* mask [[buffer(4)]],
  device float* output [[buffer(5)]],
  constant EulerParameters& p [[buffer(6)]],
  uint i [[thread_position_in_grid]]) {
  if (i >= p.count) return;
  const bool conditioned = (p.flags & 1) != 0;
  const bool terminal = (p.flags & 2) != 0;
  const bool ancestral = (p.flags & 4) != 0;
  const float m = conditioned ? mask[i / p.channels] : 1.0f;
  const float reference = conditioned ? clean[i] : 0.0f;
  const float predicted = (p.flags & 8) != 0 ? sample[i] - p.velocitySigma * prediction[i] : prediction[i];
  const float x0 = predicted * m + reference * (1.0f - m);
  if (terminal) { output[i] = x0; return; }
  float next = p.sampleScale * sample[i] + p.predictionScale * x0;
  if (ancestral) {
    next += p.noiseScale * noise[i];
    // Only re-noising requires a second conditioning blend.
    if (conditioned) next = next * m + reference * (1.0f - m);
  }
  output[i] = next;
}
