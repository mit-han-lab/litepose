#include "parse/find_peaks.hpp"
#include "parse/assign.hpp"
#include <torch/extension.h>
#include <vector>

using namespace trt_pose::parse;
// using namespace trt_pose::train;

void find_peaks_out_torch(torch::Tensor count, torch::Tensor val, torch::Tensor tag, torch::Tensor ind,
                          torch::Tensor input, torch::Tensor tmap, const float threshold,
                          const int window_size, const int max_count) {
  const int N = input.size(0);
  const int C = input.size(1);
  const int H = input.size(2);
  const int W = input.size(3);
  const int M = max_count;

  // get pointers to tensor data
  int *count_ptr = (int *)count.data_ptr();
  int *ind_ptr = (int *)ind.data_ptr();
  float *val_ptr = (float *)val.data_ptr();
  float *tag_ptr = (float *)tag.data_ptr();
  const float *input_ptr = (const float *)input.data_ptr();
  const float *tmap_ptr = (const float *)tmap.data_ptr();

  // find peaks
  find_peaks_out_nchw(count_ptr, val_ptr, tag_ptr, ind_ptr, input_ptr, tmap_ptr, 
                      N, C, H, W, M, threshold, window_size);
}

std::vector<torch::Tensor> find_peaks_torch(torch::Tensor input,
                                            torch::Tensor tmap,
                                            const float threshold,
                                            const int window_size,
                                            const int max_count) {
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);
  auto options_float = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);
  

  const int N = input.size(0);
  const int C = input.size(1);
  const int H = input.size(2);
  const int W = input.size(3);
  const int M = max_count;

  // create output tensors
  auto count = torch::zeros({N, C}, options);
  auto val = torch::zeros({N, C, M}, options_float);
  auto tag = torch::zeros({N, C, M}, options_float);
  auto ind = torch::zeros({N, C, M, 2}, options);

  // find peaks
  find_peaks_out_torch(count, val, tag, ind, input, tmap, threshold, window_size, max_count);

  return {count, val, tag, ind};
}

void assign_out_torch(torch::Tensor num, torch::Tensor ans, torch::Tensor cnt, torch::Tensor val, torch::Tensor tag, 
                      torch::Tensor ind, torch::Tensor joint_order, const float threshold, const int max_count) {
  const int C = val.size(0);
  const int M = max_count;

  // get pointers to tensor data
  int *num_ptr = (int *)num.data_ptr();
  float *ans_ptr = (float *)ans.data_ptr();
  const int *cnt_ptr = (const int *)cnt.data_ptr();
  const int *ind_ptr = (const int *)ind.data_ptr();
  const int *joint_order_ptr = (const int *)joint_order.data_ptr();
  const float *val_ptr = (const float *)val.data_ptr();
  const float *tag_ptr = (const float *)tag.data_ptr();

  // find peaks
  assign_out(num_ptr, ans_ptr, cnt_ptr, val_ptr, tag_ptr, ind_ptr, joint_order_ptr, C, M, threshold);
}

std::vector<torch::Tensor> assign_torch(torch::Tensor cnt, torch::Tensor val, torch::Tensor tag, 
                                        torch::Tensor ind, torch::Tensor joint_order, const float threshold, const int max_count) {
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);
  auto options_float = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);

  const int C = val.size(0);
  const int M = max_count;

  // create output tensors
  auto num = torch::zeros({1}, options);
  auto ans = torch::zeros({M, C, 4}, options_float);

  // find peaks
  assign_out_torch(num, ans, cnt, val, tag, ind, joint_order, threshold, max_count);

  return {num, ans};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("find_peaks", &find_peaks_torch, "find_peaks");
  m.def("find_peaks_out", &find_peaks_out_torch, "find_peaks_out");
  m.def("assign", &assign_torch, "assign");
  m.def("assign_out", &assign_out_torch, "assign_out");
  // m.def("paf_score_graph", &paf_score_graph_torch, "paf_score_graph");
  // m.def("paf_score_graph_out", &paf_score_graph_out_torch,
  //       "paf_score_graph_out");
  // m.def("refine_peaks", &refine_peaks_torch, "refine_peaks");
  // m.def("refine_peaks_out", &refine_peaks_out_torch, "refine_peaks_out");
  // // m.def("munkres", &munkres, "munkres");
  // m.def("connect_parts", &connect_parts_torch, "connect_parts");
  // m.def("connect_parts_out", &connect_parts_out_torch, "connect_parts_out");
  // m.def("assignment", &assignment_torch, "assignment");
  // m.def("assignment_out", &assignment_out_torch, "assignment_out");
  // m.def("generate_cmap", &generate_cmap, "generate_cmap");
  // m.def("generate_paf", &generate_paf, "generate_paf");
}
