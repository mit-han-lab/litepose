#include "find_peaks.hpp"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace trt_pose {
namespace parse {

void find_peaks_out_hw(int *count,        // 1
                       float *val,         // M
                       float *tag,         // M
                       int *ind,           // Mx2
                       const float *input, // HxW
                       const float *tmap,  // HxW
                       const int H, const int W, const int M,
                       const float threshold, const int window_size) {
  int win = window_size / 2;
  int cnt = 0;

  for (int i = 0; i < H && cnt < M; i++) {
    for (int j = 0; j < W && cnt < M; j++) {
      float hval = input[i * W + j];

      // skip if below threshold
      if (hval < threshold)
        continue;

      // compute window bounds
      int ii_min = MAX(i - win, 0);
      int jj_min = MAX(j - win, 0);
      int ii_max = MIN(i + win + 1, H);
      int jj_max = MIN(j + win + 1, W);

      // search for larger value in window
      bool is_peak = true;
      for (int ii = ii_min; ii < ii_max; ii++) {
        for (int jj = jj_min; jj < jj_max; jj++) {
          if (input[ii * W + jj] > hval) {
            is_peak = false;
          }
        }
      }

      // add peak
      if (is_peak) {
        ind[cnt * 2] = j;
        ind[cnt * 2 + 1] = i;
        val[cnt] = hval;
        tag[cnt] = tmap[i * W + j];
        cnt++;
      }
    }
  }

  *count = cnt;
}

void find_peaks_out_chw(int *count,         // C
                        float *val,         // CxM
                        float *tag,         // CxM
                        int *ind,           // CxMx2
                        const float *input, // CxHxW
                        const float *tmap,  // CxHxW
                        const int C, const int H, const int W, const int M,
                        const float threshold, const int window_size) {
  for (int c = 0; c < C; c++) {
    int *count_c = &count[c];
    int *ind_c = &ind[c * M * 2];
    float *val_c = &val[c * M];
    float *tag_c = &tag[c * M];    
    const float *input_c = &input[c * H * W];
    const float *tmap_c = &tmap[c * H * W];
    find_peaks_out_hw(count_c, val_c, tag_c, ind_c, input_c, tmap_c, 
                      H, W, M, threshold, window_size);
  }
}

void find_peaks_out_nchw(int *count,         // NxC
                         float *val,         // NxCxM
                         float *tag,         // NxCxM
                         int *ind,           // NxCxMx2
                         const float *input, // NxCxHxW
                         const float *tmap,  // NxCxHxW
                         const int N, const int C, const int H, const int W,
                         const int M, const float threshold,
                         const int window_size) {
  for (int n = 0; n < N; n++) {
    int *count_n = &count[n * C];
    int *ind_n = &ind[n * C * M * 2];
    float *val_n = &val[n * C * M];
    float *tag_n = &tag[n * C * M];    
    const float *input_n = &input[n * C * H * W];
    const float *tmap_n = &tmap[n * C * H * W];
    find_peaks_out_chw(count_n, val_n, tag_n, ind_n, input_n, tmap_n, 
                       C, H, W, M, threshold, window_size);
  }
}

} // namespace parse
} // namespace trt_pose
