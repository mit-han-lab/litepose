#pragma once

namespace trt_pose {
namespace parse {

void find_peaks_out_hw(int *count,        // 1
                       float *val,         // M
                       float *tag,         // M
                       int *ind,           // Mx2
                       const float *input, // HxW
                       const float *tmap,  // HxW
                       const int H, const int W, const int M,
                       const float threshold, const int window_size);

void find_peaks_out_chw(int *count,         // C
                        float *val,         // CxM
                        float *tag,         // CxM
                        int *ind,           // CxMx2
                        const float *input, // CxHxW
                        const float *tmap,  // CxHxW
                        const int C, const int H, const int W, const int M,
                        const float threshold, const int window_size);

void find_peaks_out_nchw(int *count,         // NxC
                         float *val,         // NxCxM
                         float *tag,         // NxCxM
                         int *ind,           // NxCxMx2
                         const float *input, // NxCxHxW
                         const float *tmap,  // NxCxHxW
                         const int N, const int C, const int H, const int W,
                         const int M, const float threshold,
                         const int window_size);

} // namespace parse
} // namespace trt_pose
