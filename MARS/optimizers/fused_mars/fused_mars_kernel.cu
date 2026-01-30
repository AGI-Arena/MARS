/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA/apex
   Copyright AlexwellChen
   This kernel is adapted from NVIDIA/apex and LightSeq Team
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/Exceptions.h>
#include "include/type_shim.h"
#include "include/fused_mars_kernel.cuh"


template <typename T, typename GRAD_T>
__global__ void mars_cuda_kernel(
    T* __restrict__ p,
    GRAD_T* __restrict__ g, T* __restrict__ exp_avg, T* __restrict__ exp_avg_sq, 
    const float b1, const float b2, 
    const float bias_correction1, const float bias_correction2, 
    const float lr, const float decay, const float eps, const bool ams_grad, T* __restrict__ max_exp_avg_sq, 
    const size_t total_size){
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= total_size) return;
    
    float denom;
    float step_size = lr / bias_correction1;
    float decoupled_lr = lr * decay;

    exp_avg[global_id] = b1 * exp_avg[global_id] + (1 - b1) * g[global_id];
    exp_avg_sq[global_id] = b2 * exp_avg_sq[global_id] + (1 - b2) * g[global_id] * g[global_id];

    max_exp_avg_sq[global_id] = ams_grad ? fmaxf(max_exp_avg_sq[global_id], exp_avg_sq[global_id]) : exp_avg_sq[global_id];
    denom = sqrtf(max_exp_avg_sq[global_id]) / bias_correction2 + eps;
    p[global_id] = p[global_id] * (1 - decoupled_lr) - step_size * exp_avg[global_id] / denom;
    
}

template <>
__global__ void mars_cuda_kernel<float, float>(
    float* __restrict__ p,
    float* __restrict__ g, float* __restrict__ exp_avg, float* __restrict__ exp_avg_sq, 
    const float b1, const float b2, 
    const float bias_correction1, const float bias_correction2, 
    const float lr, const float decay, const float eps, const bool ams_grad, float* __restrict__ max_exp_avg_sq, 
    const size_t total_size){

        int global_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (global_id * 4 >= total_size) return;

        float4* p4_ptr = reinterpret_cast<float4*>(p);
        float4* g4_ptr = reinterpret_cast<float4*>(g);
        float4* exp_avg4_ptr = reinterpret_cast<float4*>(exp_avg);
        float4* exp_avg_sq4_ptr = reinterpret_cast<float4*>(exp_avg_sq);
        float4* max_exp_avg_sq4_ptr = reinterpret_cast<float4*>(max_exp_avg_sq);
        
        float4 p4 = p4_ptr[global_id];
        float4 g4 = g4_ptr[global_id];
        float4 exp_avg4 = exp_avg4_ptr[global_id];
        float4 exp_avg_sq4 = exp_avg_sq4_ptr[global_id];
        float4 max_exp_avg_sq4 = max_exp_avg_sq4_ptr[global_id];

        float4 new_p4;
        float4 new_exp_avg4;
        float4 new_exp_avg_sq4;
        float4 new_max_exp_avg_sq4;
        float4 denom4;

        float step_size = lr / bias_correction1;

        new_exp_avg4.x = b1 * exp_avg4.x + (1 - b1) * g4.x;
        new_exp_avg4.y = b1 * exp_avg4.y + (1 - b1) * g4.y;
        new_exp_avg4.z = b1 * exp_avg4.z + (1 - b1) * g4.z;
        new_exp_avg4.w = b1 * exp_avg4.w + (1 - b1) * g4.w;

        new_exp_avg_sq4.x = b2 * exp_avg_sq4.x + (1 - b2) * g4.x * g4.x;
        new_exp_avg_sq4.y = b2 * exp_avg_sq4.y + (1 - b2) * g4.y * g4.y;
        new_exp_avg_sq4.z = b2 * exp_avg_sq4.z + (1 - b2) * g4.z * g4.z;
        new_exp_avg_sq4.w = b2 * exp_avg_sq4.w + (1 - b2) * g4.w * g4.w;

        new_max_exp_avg_sq4.x = ams_grad ? fmaxf(max_exp_avg_sq4.x, new_exp_avg_sq4.x) : new_exp_avg_sq4.x;
        new_max_exp_avg_sq4.y = ams_grad ? fmaxf(max_exp_avg_sq4.y, new_exp_avg_sq4.y) : new_exp_avg_sq4.y;
        new_max_exp_avg_sq4.z = ams_grad ? fmaxf(max_exp_avg_sq4.z, new_exp_avg_sq4.z) : new_exp_avg_sq4.z;
        new_max_exp_avg_sq4.w = ams_grad ? fmaxf(max_exp_avg_sq4.w, new_exp_avg_sq4.w) : new_exp_avg_sq4.w;
        denom4.x = sqrt(new_max_exp_avg_sq4.x) / bias_correction2 + eps;
        denom4.y = sqrt(new_max_exp_avg_sq4.y) / bias_correction2 + eps;
        denom4.z = sqrt(new_max_exp_avg_sq4.z) / bias_correction2 + eps;
        denom4.w = sqrt(new_max_exp_avg_sq4.w) / bias_correction2 + eps;

        new_p4.x = p4.x * (1 - lr * decay) - step_size * new_exp_avg4.x / denom4.x;
        new_p4.y = p4.y * (1 - lr * decay) - step_size * new_exp_avg4.y / denom4.y;
        new_p4.z = p4.z * (1 - lr * decay) - step_size * new_exp_avg4.z / denom4.z;
        new_p4.w = p4.w * (1 - lr * decay) - step_size * new_exp_avg4.w / denom4.w;
        
        g4_ptr[global_id] = g4;
        p4_ptr[global_id] = new_p4;
        exp_avg4_ptr[global_id] = new_exp_avg4;
        exp_avg_sq4_ptr[global_id] = new_exp_avg_sq4;
        max_exp_avg_sq4_ptr[global_id] = new_max_exp_avg_sq4;
}

void fused_mars_cuda(at::Tensor& p, at::Tensor& g, at::Tensor& exp_avg, 
          at::Tensor& exp_avg_sq, 
          float beta1, float beta2, 
          float bias_correction1, float bias_correction2, 
          float lr, float decay, float eps, bool ams_grad, at::Tensor& max_exp_avg_sq){
    // Get tensor size
    int total_size = p.numel();
    AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p),
              "parameter tensor is too large to be indexed with int32");
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (g.scalar_type() == at::ScalarType::Half) {
        const int block_dim = 1024;
        int grid_dim = ((total_size + block_dim - 1) / block_dim);
        const dim3 blocks(grid_dim);
        // all other values should be fp32 for half gradients
        AT_ASSERTM(p.scalar_type() == at::ScalarType::Float,
                  "expected parameter to be of float type");
        // dispatch is done on the gradient type
        using namespace at;  // prevents "toString is undefined" errors
        DISPATCH_FLOAT_AND_HALF(
            g.scalar_type(), 0, "mars_cuda_kernel",
            using accscalar_t = at::acc_type<scalar_t_0, true>;
            mars_cuda_kernel<accscalar_t, scalar_t_0>
            <<<blocks, block_dim, 0, stream>>>(
                p.data_ptr<accscalar_t>(),
                g.data_ptr<scalar_t_0>(), exp_avg.data_ptr<accscalar_t>(), exp_avg_sq.data_ptr<accscalar_t>(), 
                beta1, beta2, bias_correction1, bias_correction2, 
                lr, decay, eps, ams_grad, max_exp_avg_sq.data_ptr<accscalar_t>(), total_size);
            );
    } else {
        using namespace at;
        const int block_dim = 1024;
        int grid_dim = ((total_size + block_dim - 1) / block_dim) >> 2;
        if (grid_dim == 0) grid_dim = 1;
        const dim3 blocks(grid_dim);
        DISPATCH_DOUBLE_AND_FLOAT(
            g.scalar_type(), 0, "mars_cuda_kernel",
            mars_cuda_kernel<scalar_t_0, scalar_t_0>
            <<<blocks, block_dim, 0, stream>>>(
                p.data_ptr<scalar_t_0>(),
                g.data_ptr<scalar_t_0>(), exp_avg.data_ptr<scalar_t_0>(), exp_avg_sq.data_ptr<scalar_t_0>(), 
                beta1, beta2, bias_correction1, bias_correction2,
                lr, decay, eps, ams_grad, max_exp_avg_sq.data_ptr<scalar_t_0>(), total_size);
        );
    }
    AT_CUDA_CHECK(cudaGetLastError());
}

