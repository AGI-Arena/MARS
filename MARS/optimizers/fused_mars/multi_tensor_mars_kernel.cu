/* Copyright NVIDIA/apex
   Copyright AlexwellChen
   This kernel is adapted from NVIDIA/apex.
*/
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "include/type_shim.h" // Used for DISPATCH
#include "include/multi_tensor_apply.cuh" 
#include "include/fused_mars_kernel.cuh"

#define BLOCK_SIZE 512
#define ILP 4

using MATH_T = float;

template<typename T>
struct MARSFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<6>& tl,
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
    const float weight_decay_1d
    )
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T* p = (T*)tl.addresses[0][tensor_loc];
    p += chunk_idx*chunk_size;

    T* g = (T*)tl.addresses[1][tensor_loc];
    g += chunk_idx*chunk_size;

    T* exp_avg = (T*)tl.addresses[2][tensor_loc];
    exp_avg += chunk_idx*chunk_size;

    T* exp_avg_sq = (T*)tl.addresses[3][tensor_loc];
    exp_avg_sq += chunk_idx*chunk_size;

    T* max_exp_avg_sq = (T*)tl.addresses[4][tensor_loc];
    max_exp_avg_sq += chunk_idx*chunk_size;

    T* is_grad_2d = (T*)tl.addresses[5][tensor_loc];
    is_grad_2d += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_p[ILP];
      MATH_T r_g[ILP];
      MATH_T r_exp_avg[ILP];
      MATH_T r_exp_avg_sq[ILP];
      MATH_T r_max_exp_avg_sq[ILP];
      MATH_T r_is_grad_2d[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_p[ii] = p[i];
          r_g[ii] = g[i];
          r_exp_avg[ii] = exp_avg[i];
          r_exp_avg_sq[ii] = exp_avg_sq[i];
          r_max_exp_avg_sq[ii] = max_exp_avg_sq[i];
          r_is_grad_2d[ii] = is_grad_2d[i];
        } else {
          r_p[ii] = MATH_T(0);
          r_g[ii] = MATH_T(0);
          r_exp_avg[ii] = MATH_T(0);
          r_exp_avg_sq[ii] = MATH_T(0);
          r_max_exp_avg_sq[ii] = MATH_T(0);
          r_is_grad_2d[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        MATH_T step_size = r_is_grad_2d[ii] > 0 ? lr / bias_correction1 : lr / bc_1_1d * lr_1d_factor;
        MATH_T decoupled_weight_decay = r_is_grad_2d[ii] > 0 ? lr * decay : lr * lr_1d_factor * weight_decay_1d;
        MATH_T beta_1 = r_is_grad_2d[ii] > 0 ? beta1 : beta_1_1d;
        MATH_T beta_2 = r_is_grad_2d[ii] > 0 ? beta2 : beta_2_1d;
        r_exp_avg[ii] = beta_1 * r_exp_avg[ii] + (1 - beta_1) * r_g[ii];
        r_exp_avg_sq[ii] = beta_2 * r_exp_avg_sq[ii] + (1 - beta_2) * r_g[ii] * r_g[ii];
        r_max_exp_avg_sq[ii] = ams_grad ? fmaxf(r_max_exp_avg_sq[ii], r_exp_avg_sq[ii]) : r_exp_avg_sq[ii];
        MATH_T denom = sqrtf(r_max_exp_avg_sq[ii]) / bc_2_1d + epsilon;          
        r_p[ii] = r_p[ii] * (1 - decoupled_weight_decay) - step_size * r_exp_avg[ii] / denom;
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          g[i] = r_g[ii];
          p[i] = r_p[ii];
          exp_avg[i] = r_exp_avg[ii];
          exp_avg_sq[i] = r_exp_avg_sq[ii];
          max_exp_avg_sq[i] = r_max_exp_avg_sq[ii];
          is_grad_2d[i] = r_is_grad_2d[ii];
        }
      }
    }
  }
};

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
  const float weight_decay_1d)
{
  using namespace at;
  TORCH_CHECK(!tensor_lists.empty(), "tensor list cannot be empty")
  if (tensor_lists[0].empty()) {
    return;
  }

  // Assume single type across p,g,m1,m2 now
  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
    tensor_lists[0][0].scalar_type(), 0, "mars",
    multi_tensor_apply<6>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      MARSFunctor<scalar_t_0>(),
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
      ); )

  AT_CUDA_CHECK(cudaGetLastError());

}
