/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA/apex
   Copyright AlexwellChen
   This kernel is adapted from NVIDIA/apex and LightSeq Team
*/
#include <ATen/ATen.h>
#include <torch/extension.h>

// CUDA forward declaration
void fused_mars_cuda(at::Tensor& p, at::Tensor& g, at::Tensor& exp_avg, 
    at::Tensor& exp_avg_sq, 
    float beta1, float beta2, 
    float bias_correction1, float bias_correction2, 
    float lr, float decay, float eps, bool ams_grad, at::Tensor& max_exp_avg_sq);

void multi_tensor_mars_cuda(
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
    const float weight_decay_1d);