#include <torch/extension.h>

#include "include/fused_mars_kernel.cuh"

// x is torch::Tensor
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// C++ interface


// def update_fn(p, grad, exp_avg, exp_avg_sq, lr, wd, beta1, beta2, eps, amsgrad, max_exp_avg_sq):
void mars_single_tensor(at::Tensor& p, 
          at::Tensor& g, 
          at::Tensor& exp_avg, 
          at::Tensor& exp_avg_sq, 
          float beta1, float beta2, 
          float bias_correction1, float bias_correction2, 
          float lr, float decay, float eps, bool ams_grad, at::Tensor& max_exp_avg_sq) {
  CHECK_INPUT(p);
  CHECK_INPUT(exp_avg);
  CHECK_INPUT(exp_avg_sq);
  CHECK_INPUT(g);
  int64_t num_elem = p.numel();
  AT_ASSERTM(exp_avg.numel() == num_elem,
             "number of elements in exp_avg and p tensors should be equal");
  AT_ASSERTM(exp_avg_sq.numel() == num_elem,
             "number of elements in exp_avg_sq and p tensors should be equal");
  AT_ASSERTM(g.numel() == num_elem,
             "number of elements in g and p tensors should be equal");

  fused_mars_cuda(p, g, 
                  exp_avg, exp_avg_sq, 
                  beta1, beta2, 
                  bias_correction1, bias_correction2, 
                  lr, decay, eps, ams_grad, max_exp_avg_sq);
}

void mars_multi_tensor(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float beta1,
  const float beta2,
  const float bias_correction1,
  const float bias_correction2,
  const float lr,
  const float decay,
  const float epsilon,
  const bool ams_grad,
  const float gamma,
  const bool optimize_1d,
  const float lr_1d_factor,
  const float beta_1_1d,
  const float beta_2_1d,
  const float bc_1_1d,
  const float bc_2_1d,
  const float weight_decay_1d){
    multi_tensor_mars_cuda(
      chunk_size,
      noop_flag,
      tensor_lists,
      beta1,
      beta2,
      bias_correction1,
      bias_correction2,
      lr,
      decay,
      epsilon,
      ams_grad,
      gamma,
      optimize_1d,
      lr_1d_factor,
      beta_1_1d,
      beta_2_1d,
      bc_1_1d,
      bc_2_1d,
      weight_decay_1d
    );
  }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mars_single_tensor", &mars_single_tensor, "MARS optimized CUDA single tensor implementation.");
  m.def("mars_multi_tensor", &mars_multi_tensor, "MARS optimized CUDA multi tensor implementation.");
}
