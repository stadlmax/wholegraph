/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "gnn_ops.h"

#include "data_type.h"
#include "macros.h"
#include "whole_chunked_memory.cuh"
#include "whole_chunked_memory.h"
#include "whole_graph_mixed_graph.cuh"
#include "whole_memory.h"

namespace whole_graph {

// aggregator
// 0: sum
// 1: mean
// 2: gcn
template<typename T = float, int AGG = 0>
__global__ void SpmmCsrNoWeightForwardSimpleKernel(const int *csr_row_ptr,
                                                   const int *csr_col_ind,
                                                   const T *x,
                                                   int64_t embedding_dim,
                                                   int64_t embedding_stride,
                                                   int64_t input_count,
                                                   T *output,
                                                   int64_t output_stride) {
  int row_idx = blockIdx.x;
  int emb_idx = threadIdx.x;
  int row_ptr_start = csr_row_ptr[row_idx];
  int row_ptr_end = csr_row_ptr[row_idx + 1];
  int agg_count = row_ptr_end - row_ptr_start;
  for (; emb_idx < embedding_dim; emb_idx += blockDim.x) {
    float agg_value = 0.0f;
    if (AGG == 2) {
      agg_value += (float) x[(int64_t) row_idx * embedding_stride + emb_idx];
    }

    for (int row_ptr = row_ptr_start; row_ptr < row_ptr_end; row_ptr++) {
      int col_idx = csr_col_ind[row_ptr];
      assert(col_idx >= 0 && col_idx < input_count);
      float value = (float) x[(int64_t) col_idx * embedding_stride + emb_idx];
      agg_value += value;
    }
    if (AGG == 1) {
      if (agg_count > 0) {
        agg_value /= agg_count;
      }
    }
    if (AGG == 2) {
      agg_value /= (agg_count + 1);
    }
    output[(int64_t) row_idx * output_stride + emb_idx] = (T) agg_value;
  }
}

template<typename T>
void SpmmCsrNoWeightForwardCommon(const int *csr_row_ptr, int64_t row_count,
                                  const int *csr_col_ind, int64_t col_count,
                                  const void *x, int64_t embedding_dim, int64_t embedding_stride, int64_t input_count,
                                  void *output, int64_t output_stride, int aggregator,
                                  cudaStream_t stream) {
  int block_count = row_count;
  // Now only support simple
  int thread_count = embedding_dim;
  if (thread_count > 512) {
    thread_count = 512;
  }
  if (aggregator == 0) {
    SpmmCsrNoWeightForwardSimpleKernel<T, 0><<<block_count, thread_count, 0, stream>>>(
        csr_row_ptr,
        csr_col_ind,
        (const T *) x,
        embedding_dim,
        embedding_stride,
        input_count,
        (T *) output,
        output_stride);
  } else if (aggregator == 1) {
    SpmmCsrNoWeightForwardSimpleKernel<T, 1><<<block_count, thread_count, 0, stream>>>(
        csr_row_ptr,
        csr_col_ind,
        (const T *) x,
        embedding_dim,
        embedding_stride,
        input_count,
        (T *) output,
        output_stride);
  } else if (aggregator == 2) {
    SpmmCsrNoWeightForwardSimpleKernel<T, 2><<<block_count, thread_count, 0, stream>>>(
        csr_row_ptr,
        csr_col_ind,
        (const T *) x,
        embedding_dim,
        embedding_stride,
        input_count,
        (T *) output,
        output_stride);
  }
}

REGISTER_DISPATCH_ONE_TYPE(SpmmCsrNoWeightForwardCommon, SpmmCsrNoWeightForwardCommon, HALF_FLOAT)

void SpmmCsrNoWeightForward(WMType data_type,
                            const int *csr_row_ptr,
                            int m,// total row count
                            const int *csr_col_ind,
                            int total_nz_element,
                            const void *input,
                            int n,// embedding dim
                            int ldb,
                            int k,// total column count
                            void *output,
                            int ldc,
                            int aggregator,
                            cudaStream_t stream) {
  DISPATCH_ONE_TYPE(data_type, SpmmCsrNoWeightForwardCommon,
                    csr_row_ptr, (int64_t) m,
                    csr_col_ind, (int64_t) total_nz_element,
                    input, (int64_t) n, (int64_t) ldb, (int64_t) k,
                    output, (int64_t) ldc, aggregator,
                    stream);
}

template<typename T = float>
__global__ void SpmmCsrNoWeightAtomicBackwardFillKernel(T *grad_x,
                                                        int64_t grad_x_stride,
                                                        const int *sample_dup_count,
                                                        int64_t embedding_dim) {
  int input_idx = blockIdx.x;
  int emb_idx = threadIdx.x;
  int dup_count = sample_dup_count != nullptr ? sample_dup_count[input_idx] : 0;
  if (dup_count == 1) return;
  for (; emb_idx < embedding_dim; emb_idx += blockDim.x) {
    grad_x[(int64_t) input_idx * grad_x_stride + emb_idx] = (T) 0.0f;
  }
}

template<typename T = float>
__global__ void SpmmGCNCsrNoWeightAtomicBackwardFillKernel(T *grad_x,
                                                           int64_t grad_x_stride,
                                                           const int *sample_dup_count,
                                                           int64_t embedding_dim,
                                                           const T *grad_output,
                                                           int64_t grad_out_stride,
                                                           int64_t target_vec_count,
                                                           const int *csr_row_ptr) {
  int input_idx = blockIdx.x;
  int dup_count = sample_dup_count != nullptr ? sample_dup_count[input_idx] : 0;
  if (input_idx < target_vec_count) {
    int coefficient = csr_row_ptr[input_idx + 1] - csr_row_ptr[input_idx] + 1;
    for (int emb_idx = threadIdx.x; emb_idx < embedding_dim; emb_idx += blockDim.x) {
      float value = grad_output[(int64_t) input_idx * grad_out_stride + emb_idx];
      value /= (float) coefficient;
      grad_x[(int64_t) input_idx * grad_x_stride + emb_idx] = value;
    }
    return;
  }
  if (dup_count == 1) return;
  for (int emb_idx = threadIdx.x; emb_idx < embedding_dim; emb_idx += blockDim.x) {
    grad_x[(int64_t) input_idx * grad_x_stride + emb_idx] = (T) 0.0f;
  }
}

template<typename T = float, int AGG = 0>
__global__ void SpmmCsrNoWeightAtomicBackwardInputKernel(const int *csr_row_ptr,
                                                         const int *csr_col_ind,
                                                         const int *sample_dup_count,
                                                         const T *grad_output,
                                                         int64_t grad_out_stride,
                                                         T *grad_x,
                                                         int64_t grad_x_stride,
                                                         int64_t input_count,
                                                         int64_t embedding_dim,
                                                         int64_t target_vec_count = 0) {
  int row_idx = blockIdx.x;
  int emb_idx = threadIdx.x;
  int row_ptr_start = csr_row_ptr[row_idx];
  int row_ptr_end = csr_row_ptr[row_idx + 1];
  int agg_count = row_ptr_end - row_ptr_start;
  for (; emb_idx < embedding_dim; emb_idx += blockDim.x) {
    float value = (float) grad_output[(int64_t) row_idx * grad_out_stride + emb_idx];
    if (AGG == 1) {
      if (agg_count > 0) {
        value /= agg_count;
      }
    }
    if (AGG == 2) {
      if (agg_count > 0) {
        value /= (agg_count + 1);
      }
    }
    for (int row_ptr = row_ptr_start; row_ptr < row_ptr_end; row_ptr++) {
      int col_idx = csr_col_ind[row_ptr];
      assert(col_idx >= 0 && col_idx < input_count);
      int dup_count = sample_dup_count != nullptr ? sample_dup_count[col_idx] : 0;
      if (AGG != 2) {
        if (dup_count == 1) {
          grad_x[(int64_t) col_idx * grad_x_stride + emb_idx] = (T) value;
        } else {
          atomicAdd(grad_x + (int64_t) col_idx * grad_x_stride + emb_idx, (T) value);
        }
      }
      if (AGG == 2) {
        if (dup_count == 1 && col_idx < target_vec_count) {
          grad_x[(int64_t) col_idx * grad_x_stride + emb_idx] += (T) value;
        } else if (dup_count == 1 && col_idx >= target_vec_count) {
          grad_x[(int64_t) col_idx * grad_x_stride + emb_idx] = (T) value;
        } else {
          atomicAdd(grad_x + (int64_t) col_idx * grad_x_stride + emb_idx, (T) value);
        }
      }
    }
  }
}

template<typename T = float>
void SpmmCsrNoWeightBackwardCommon(const int *csr_row_ptr,
                                   const int *csr_col_ind,
                                   const int *sample_dup_count,
                                   const void *grad_output,
                                   int64_t grad_out_stride,
                                   void *grad_x,
                                   int64_t grad_x_stride,
                                   int64_t m,
                                   int64_t total_nz_element,
                                   int64_t k,
                                   int64_t embedding_dim,
                                   int aggregator,
                                   cudaStream_t stream) {
  int block_count = m;
  // Now only support simple
  int thread_count = embedding_dim;
  if (thread_count > 512) {
    thread_count = 512;
  }
  if (aggregator == 0 || aggregator == 1) {
    SpmmCsrNoWeightAtomicBackwardFillKernel<T><<<k, thread_count, 0, stream>>>(
        (T *) grad_x, grad_x_stride, sample_dup_count, embedding_dim);
  } else if (aggregator == 2) {
    SpmmGCNCsrNoWeightAtomicBackwardFillKernel<T><<<k, thread_count, 0, stream>>>(
        (T *) grad_x, grad_x_stride, sample_dup_count, embedding_dim,
        (const T *) grad_output, grad_out_stride, m,
        csr_row_ptr);
  }

  if (aggregator == 0) {
    SpmmCsrNoWeightAtomicBackwardInputKernel<T, 0><<<block_count, thread_count, 0, stream>>>(
        csr_row_ptr,
        csr_col_ind,
        sample_dup_count,
        (const T *) grad_output,
        grad_out_stride,
        (T *) grad_x,
        grad_x_stride,
        k,
        embedding_dim);
  } else if (aggregator == 1) {
    SpmmCsrNoWeightAtomicBackwardInputKernel<T, 1><<<block_count, thread_count, 0, stream>>>(
        csr_row_ptr,
        csr_col_ind,
        sample_dup_count,
        (const T *) grad_output,
        grad_out_stride,
        (T *) grad_x,
        grad_x_stride,
        k,
        embedding_dim);
  } else if (aggregator == 2) {
    SpmmCsrNoWeightAtomicBackwardInputKernel<T, 2><<<block_count, thread_count, 0, stream>>>(
        csr_row_ptr,
        csr_col_ind,
        sample_dup_count,
        (const T *) grad_output,
        grad_out_stride,
        (T *) grad_x,
        grad_x_stride,
        k,
        embedding_dim,
        m);
  }
}

REGISTER_DISPATCH_ONE_TYPE(SpmmCsrNoWeightBackwardCommon, SpmmCsrNoWeightBackwardCommon, HALF_FLOAT)

void SpmmCsrNoWeightBackword(WMType data_type,
                             const int *csr_row_ptr,
                             const int *csr_col_ind,
                             const int *sample_dup_count,
                             const void *grad_output,
                             int grad_out_stride,
                             void *grad_x,
                             int grad_x_stride,
                             int m,// total row count,
                             int total_nz_element,
                             int k,// total column count
                             int embedding_dim,
                             int aggregator,
                             cudaStream_t stream) {
  DISPATCH_ONE_TYPE(data_type, SpmmCsrNoWeightBackwardCommon,
                    csr_row_ptr,
                    csr_col_ind,
                    sample_dup_count,
                    grad_output,
                    (int64_t) grad_out_stride,
                    grad_x,
                    (int64_t) grad_x_stride,
                    (int64_t) m,
                    (int64_t) total_nz_element,
                    (int64_t) k,
                    (int64_t) embedding_dim,
                    aggregator,
                    stream);
}

template<typename T = float>
__global__ void gSpmmCsrWeightedForwardKernel(const int *csr_row_ptr,
                                              const int *csr_col_ind,
                                              const T *edge_weight,
                                              int num_head,
                                              const T *x,
                                              int embedding_dim,
                                              int ldb,
                                              int ldb_head,
                                              int input_count,
                                              T *output,
                                              int ldc,
                                              int ldc_head) {
  // ldb_head may be ldb * num_head
  int row_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int emb_idx = threadIdx.x;
  int row_ptr_start = csr_row_ptr[row_idx];
  int row_ptr_end = csr_row_ptr[row_idx + 1];

  for (; emb_idx < embedding_dim; emb_idx += blockDim.x) {
    float agg_value = 0.0f;
    for (int row_ptr = row_ptr_start; row_ptr < row_ptr_end; row_ptr++) {
      float ew = (float) edge_weight[row_ptr * num_head + head_idx];
      int col_idx = csr_col_ind[row_ptr];
      assert(col_idx >= 0 && col_idx < input_count);
      float value = (float) x[(int64_t) col_idx * ldb_head + head_idx * ldb + emb_idx];
      agg_value += value * ew;
    }
    output[(int64_t) row_idx * ldc_head + head_idx * ldc + emb_idx] = (T) agg_value;
  }
}

template<typename T = float>
void gSpmmCsrWeightedForwardCommon(const int *csr_row_ptr, int row_count,
                                   const int *csr_col_ind, int nz_count,
                                   const void *edge_weight, int num_head,
                                   const void *x, int embedding_dim, int ldb, int ldb_head,
                                   int input_count,
                                   void *output, int ldc, int ldc_head, cudaStream_t stream) {
  int block_count = row_count;
  // Now only support simple
  int thread_count = embedding_dim;
  if (thread_count > 512) {
    thread_count = 512;
  }
  dim3 block(block_count, num_head);
  gSpmmCsrWeightedForwardKernel<T><<<block, thread_count, 0, stream>>>(
      csr_row_ptr,
      csr_col_ind,
      (const T *) edge_weight,
      num_head,
      (const T *) x,
      embedding_dim,
      ldb,
      ldb_head,
      input_count,
      (T *) output,
      ldc,
      ldc_head);
}

REGISTER_DISPATCH_ONE_TYPE(gSpmmCsrWeightedForwardCommon, gSpmmCsrWeightedForwardCommon, HALF_FLOAT)

void gSpmmCsrWeightedForward(WMType data_type,
                             const int *csr_row_ptr,
                             int m,// total row count
                             const int *csr_col_ind,
                             int total_nz_element,
                             const void *weight,
                             int num_head,
                             const void *input,
                             int n,// embedding dim
                             int ldb,
                             int ldb_head,
                             int k,// total column count
                             void *output,
                             int ldc,
                             int ldc_head,
                             cudaStream_t stream) {
  DISPATCH_ONE_TYPE(data_type, gSpmmCsrWeightedForwardCommon,
                    csr_row_ptr, m,
                    csr_col_ind, total_nz_element,
                    weight, num_head,
                    input, n, ldb, ldb_head, k,
                    output, ldc, ldc_head,
                    stream);
}

template<typename T = float>
__global__ void gSpmmCsrAtomicBackwardFillInputKernel(T *grad_x,
                                                      const int *sample_dup_count,
                                                      int embedding_dim,
                                                      int embedding_stride,
                                                      int head_stride,
                                                      int num_head) {
  int input_idx = blockIdx.x;
  int emb_idx = threadIdx.x;
  int dup_count = sample_dup_count != nullptr ? sample_dup_count[input_idx] : 0;
  if (dup_count == 1) return;
  for (; emb_idx < embedding_dim * num_head; emb_idx += blockDim.x) {
    int head_idx = emb_idx / embedding_dim;
    int idx_in_emb = emb_idx % embedding_dim;
    grad_x[(int64_t) input_idx * head_stride + head_idx * embedding_stride + idx_in_emb] = (T) 0.0f;
  }
}

template<typename T = float, bool NeedBackwardX = true, bool NeedBackwardW = true>
__global__ void gSpmmCsrWeightedFusedSharedMemoryBackwardKernel(
    const int *csr_row_ptr, int target_count,
    const int *csr_col_ind, int edge_count,
    const T *edge_weight,
    const int *sample_dup_count,
    const T *x, int embedding_dim, int num_head,
    int ldx, int ldx_head,
    int neighbor_count,
    const T *output_grad,
    int ld_grad_out, int ld_grad_out_head,
    T *x_grad,
    int ld_grad_x, int ld_grad_x_head,
    T *edge_weight_grad) {
  int row_idx = blockIdx.x;// target index
  int head_idx = blockIdx.y;
  int emb_idx_start = threadIdx.x;
  int row_ptr_start = csr_row_ptr[row_idx];
  int row_ptr_end = csr_row_ptr[row_idx + 1];
  assert(row_ptr_start >= 0 && row_ptr_start <= edge_count);
  assert(row_ptr_end >= 0 && row_ptr_end <= edge_count);
  assert(row_ptr_end >= row_ptr_start);
  extern __shared__ float w_grad_data[];
  for (int row_ptr = row_ptr_start; row_ptr < row_ptr_end; row_ptr++) {
    int col_idx = csr_col_ind[row_ptr];
    assert(col_idx >= 0 && col_idx < neighbor_count);
    float edge_weight_value = (float) edge_weight[row_ptr * num_head + head_idx];
    float edge_weight_grad_value = 0.0f;
    for (int emb_idx = emb_idx_start; emb_idx < embedding_dim; emb_idx += blockDim.x) {
      float grad_out_value =
          (float) output_grad[(int64_t) row_idx * ld_grad_out_head + (int64_t) head_idx * ld_grad_out + emb_idx];
      float x_value = (float) x[(int64_t) col_idx * ldx_head + (int64_t) head_idx * ldx + emb_idx];
      int dup_count = sample_dup_count != nullptr ? sample_dup_count[col_idx] : 0;
      // update x_grad here.
      float x_grad_value = edge_weight_value * grad_out_value;
      if (NeedBackwardX) {
        if (dup_count == 1) {
          x_grad[(int64_t) col_idx * ld_grad_x_head + (int64_t) head_idx * ld_grad_x + emb_idx] = (T) x_grad_value;
        } else {
          atomicAdd(x_grad + (int64_t) col_idx * ld_grad_x_head + (int64_t) head_idx * ld_grad_x + emb_idx,
                    (T) x_grad_value);
        }
      }
      edge_weight_grad_value += x_value * grad_out_value;
    }
    if (NeedBackwardW) {
      if (embedding_dim > 32) {
        w_grad_data[threadIdx.x] = edge_weight_grad_value;
        __syncthreads();
      }
      if (threadIdx.x < 32) {
        for (int widx = threadIdx.x + 32; widx < embedding_dim && widx < blockDim.x; widx += 32) {
          edge_weight_grad_value += w_grad_data[widx];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
          edge_weight_grad_value += __shfl_down_sync(0xffffffff, edge_weight_grad_value, offset, 32);
        }
        if (threadIdx.x == 0) {
          edge_weight_grad[row_ptr * num_head + head_idx] = (T) edge_weight_grad_value;
        }
      }
    }
  }
}

template<typename T = float, bool NeedBackwardX = true, bool NeedBackwardW = true>
void gSpmmCsrWeightedFusedSharedMemoryBackwardFunc(
    const int *csr_row_ptr, int target_count,
    const int *csr_col_ind, int edge_count,
    const T *edge_weight,
    const int *sample_dup_count,
    const T *x, int embedding_dim, int num_head,
    int ldx, int ldx_head, int neighbor_count,
    const T *output_grad,
    int ld_grad_out, int ld_grad_out_head,
    T *x_grad,
    int ld_grad_x, int ld_grad_x_head,
    T *edge_weight_grad,
    cudaStream_t stream) {
  if (NeedBackwardX) {
    cudaMemsetAsync(x_grad, 0, neighbor_count * embedding_dim * num_head * sizeof(T), stream);
    gSpmmCsrAtomicBackwardFillInputKernel<T>
        <<<target_count, std::min<int>(512, embedding_dim * num_head), embedding_dim * sizeof(T), stream>>>(
            x_grad, sample_dup_count, embedding_dim, ld_grad_x, ld_grad_x_head, num_head);
  }

  if (NeedBackwardW) {
    cudaMemsetAsync(edge_weight_grad, 0, edge_count * num_head * sizeof(T), stream);
  }
  dim3 block(target_count, num_head);
  int thread_count = std::min<int>(512, embedding_dim);
  thread_count = std::max<int>(32, thread_count);// at least one warp
  gSpmmCsrWeightedFusedSharedMemoryBackwardKernel<T, NeedBackwardX, NeedBackwardW>
      <<<block, thread_count, thread_count * sizeof(float), stream>>>(
          csr_row_ptr, target_count, csr_col_ind, edge_count,
          edge_weight, sample_dup_count,
          x, embedding_dim, num_head, ldx, ldx_head, neighbor_count,
          output_grad, ld_grad_out, ld_grad_out_head,
          x_grad, ld_grad_x, ld_grad_x_head, edge_weight_grad);
}

template<typename T = float>
void gSpmmCsrWeightedFusedSharedMemoryBackwardCommon(bool backward_x, bool backward_w,
                                                     const int *csr_row_ptr, int target_count,
                                                     const int *csr_col_ind, int edge_count,
                                                     const void *edge_weight,
                                                     const int *sample_dup_count,
                                                     const void *x, int embedding_dim, int num_head,
                                                     int ldx, int ldx_head, int neighbor_count,
                                                     const void *output_grad,
                                                     int ld_grad_out, int ld_grad_out_head,
                                                     void *x_grad,
                                                     int ld_grad_x, int ld_grad_x_head,
                                                     void *edge_weight_grad,
                                                     cudaStream_t stream) {
  auto fn = gSpmmCsrWeightedFusedSharedMemoryBackwardFunc<T, true, true>;
  if (backward_x) {
    if (!backward_w) {
      fn = gSpmmCsrWeightedFusedSharedMemoryBackwardFunc<T, true, false>;
    }
  } else {
    if (backward_w) {
      fn = gSpmmCsrWeightedFusedSharedMemoryBackwardFunc<T, false, true>;
    } else {
      fn = nullptr;
    }
  }
  fn(csr_row_ptr, target_count, csr_col_ind, edge_count, (const T *) edge_weight, sample_dup_count,
     (const T *) x, embedding_dim, num_head,
     ldx, ldx_head, neighbor_count,
     (const T *) output_grad, ld_grad_out, ld_grad_out_head,
     (T *) x_grad, ld_grad_x, ld_grad_x_head,
     (T *) edge_weight_grad, stream);
}

REGISTER_DISPATCH_ONE_TYPE(
    gSpmmCsrWeightedFusedSharedMemoryBackwardCommon,
    gSpmmCsrWeightedFusedSharedMemoryBackwardCommon,
    HALF_FLOAT)

void gSpmmCsrWeightedFusedSharedMemoryBackward(WMType data_type,
                                               bool backward_x, bool backward_w,
                                               const int *csr_row_ptr, int target_count,
                                               const int *csr_col_ind, int edge_count,
                                               const void *edge_weight,
                                               const int *sample_dup_count,
                                               const void *x, int embedding_dim, int num_head,
                                               int ldx, int ldx_head, int neighbor_count,
                                               const void *output_grad,
                                               int ld_grad_out, int ld_grad_out_head,
                                               void *x_grad,
                                               int ld_grad_x, int ld_grad_x_head,
                                               void *edge_weight_grad,
                                               cudaStream_t stream) {
  DISPATCH_ONE_TYPE(data_type, gSpmmCsrWeightedFusedSharedMemoryBackwardCommon,
                    backward_x, backward_w,
                    csr_row_ptr, target_count, csr_col_ind, edge_count, edge_weight, sample_dup_count,
                    x, embedding_dim, num_head,
                    ldx, ldx_head, neighbor_count,
                    output_grad, ld_grad_out, ld_grad_out_head,
                    x_grad, ld_grad_x, ld_grad_x_head,
                    edge_weight_grad, stream);
}

template<typename T = float>
__global__ void SpAddGATCSRForwardSimpleKernel(const int *csr_row_ptr,
                                               int target_vec_count,
                                               const int *csr_col_ind,
                                               const T *edge_weight_left,
                                               const T *edge_weight_right,
                                               int num_head,
                                               T *output) {

  int row_id = blockIdx.x * blockDim.y + threadIdx.y;
  int head_id = threadIdx.x;
  if (row_id >= target_vec_count) return;

  int row_ptr_start = csr_row_ptr[row_id];
  int row_ptr_end = csr_row_ptr[row_id + 1];

  T left_weight = edge_weight_left[row_id * num_head + head_id];
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    int col_id = csr_col_ind[row_ptr_id];
    T right_weight = edge_weight_right[col_id * num_head + head_id];
    T value = left_weight + right_weight;

    output[row_ptr_id * num_head + head_id] = value;
  }
}

template<typename T = float>
void SpAddGATCSRForwardCommon(const int *csr_row_ptr,
                              int target_vec_count,
                              const int *csr_col_ind,
                              const void *edge_weight_left,
                              const void *edge_weight_right,
                              int num_head,
                              void *output, cudaStream_t stream) {
  assert(num_head <= 512);
  int row_num_per_block = 512 / num_head;
  dim3 block(num_head, row_num_per_block);
  int block_count = (target_vec_count + row_num_per_block - 1) / row_num_per_block;

  SpAddGATCSRForwardSimpleKernel<T><<<block_count, block, 0, stream>>>(
      csr_row_ptr,
      target_vec_count,
      csr_col_ind,
      (const T *) edge_weight_left,
      (const T *) edge_weight_right,
      num_head,
      (T *) output);
}

REGISTER_DISPATCH_ONE_TYPE(SpAddGATCSRForwardCommon, SpAddGATCSRForwardCommon, HALF_FLOAT)

void SpAddGATCSRForward(WMType data_type,
                        const int *csr_row_ptr,
                        int target_vec_count,
                        const int *csr_col_ind,
                        const void *edge_weight_left,
                        const void *edge_weight_right,
                        int num_head,
                        void *output, cudaStream_t stream) {
  DISPATCH_ONE_TYPE(data_type, SpAddGATCSRForwardCommon,
                    csr_row_ptr,
                    target_vec_count,
                    csr_col_ind,
                    edge_weight_left,
                    edge_weight_right,
                    num_head,
                    output,
                    stream);
}

template<typename T = float>
__global__ void SpAddGATCSRBackwardInitGradWeightRight(const int *sample_dup_count,
                                                       T *grad_weight_right,
                                                       int neighbor_count,
                                                       int num_head) {

  int row_id = blockIdx.x * blockDim.y + threadIdx.y;
  int head_id = threadIdx.x;
  if (row_id >= neighbor_count) return;
  int dup_count = sample_dup_count != nullptr ? sample_dup_count[row_id] : 0;
  if (dup_count == 1) return;
  grad_weight_right[row_id * num_head + head_id] = (T) 0.0f;
}

template<typename T = float>
__global__ void SpAddGATCSRBackwardSimpleKernel(const int *csr_row_ptr,
                                                const int *csr_col_ind,
                                                const int *sample_dup_count,
                                                const T *grad_y,
                                                int target_vec_count,
                                                int num_head,
                                                T *grad_weight_left,
                                                T *grad_weight_right) {

  int row_id = blockIdx.x * blockDim.y + threadIdx.y;
  int head_id = threadIdx.x;
  if (row_id >= target_vec_count) return;

  int row_ptr_start = csr_row_ptr[row_id];
  int row_ptr_end = csr_row_ptr[row_id + 1];
  T grad_weight_left_value = (T) 0.0f;
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    T grad_y_value = grad_y[row_ptr_id * num_head + head_id];
    grad_weight_left_value += grad_y_value;
    int col_id = csr_col_ind[row_ptr_id];
    int dup_count = sample_dup_count != nullptr ? sample_dup_count[col_id] : 0;
    if (dup_count == 1) {
      grad_weight_right[col_id * num_head + head_id] = grad_y_value;
    } else {
      atomicAdd(grad_weight_right + col_id * num_head + head_id, (T) grad_y_value);
    }
  }
  grad_weight_left[row_id * num_head + head_id] = grad_weight_left_value;
}

template<typename T = float>
void SpAddGATCSRBackwardCommon(const int *csr_row_ptr,
                               const int *csr_col_ind,
                               const int *sample_dup_count,
                               const void *grad_y,
                               int target_vec_count,
                               int neighbor_count,
                               int num_head,
                               void *grad_weight_left,
                               void *grad_weight_right,
                               cudaStream_t stream) {
  assert(num_head <= 512);

  int rows_per_block = 512 / num_head;
  dim3 block(num_head, rows_per_block);
  int block_count = (neighbor_count + rows_per_block - 1) / rows_per_block;

  SpAddGATCSRBackwardInitGradWeightRight<T><<<block_count, block, 0, stream>>>(sample_dup_count,
                                                                               (T *) grad_weight_right,
                                                                               neighbor_count,
                                                                               num_head);

  block_count = (target_vec_count + rows_per_block - 1) / rows_per_block;
  SpAddGATCSRBackwardSimpleKernel<T><<<block_count, block, 0, stream>>>(csr_row_ptr,
                                                                        csr_col_ind,
                                                                        sample_dup_count,
                                                                        (const T *) grad_y,
                                                                        target_vec_count,
                                                                        num_head,
                                                                        (T *) grad_weight_left,
                                                                        (T *) grad_weight_right);
}

REGISTER_DISPATCH_ONE_TYPE(SpAddGATCSRBackwardCommon, SpAddGATCSRBackwardCommon, HALF_FLOAT)

void SpAddGATCSRBackward(WMType data_type,
                         const int *csr_row_ptr,
                         const int *csr_col_ind,
                         const int *sample_dup_count,
                         const void *grad_y,
                         int target_vec_count,
                         int neighbor_count,
                         int num_head,
                         void *grad_weight_left,
                         void *grad_weight_right,
                         cudaStream_t stream) {
  DISPATCH_ONE_TYPE(data_type, SpAddGATCSRBackwardCommon,
                    csr_row_ptr,
                    csr_col_ind,
                    sample_dup_count,
                    grad_y,
                    target_vec_count,
                    neighbor_count,
                    num_head,
                    grad_weight_left,
                    grad_weight_right,
                    stream);
}

template<typename T = float>
__global__ void EdgeWeightSoftmaxForwardSimpleKernel(const int *csr_row_ptr,
                                                     const int64_t target_vec_count,
                                                     const T *edge_weight,
                                                     const int num_head,
                                                     T *edge_weight_softmax) {

  int row_id = blockIdx.x * blockDim.y + threadIdx.y;
  int head_id = threadIdx.x;
  if (row_id >= target_vec_count) return;

  int row_ptr_start = csr_row_ptr[row_id];
  int row_ptr_end = csr_row_ptr[row_id + 1];
  float max_val = -1e38f;
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    float value = (float) edge_weight[row_ptr_id * num_head + head_id];
    max_val = max(max_val, value);
  }
  float total_exp_value = 0.0f;
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    float value = (float) edge_weight[row_ptr_id * num_head + head_id];
    total_exp_value += __expf(value - max_val);
  }

  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    float value = (float) edge_weight[row_ptr_id * num_head + head_id];
    value = __expf(value - max_val);
    value /= total_exp_value;
    //if (isnan(value)) {
    //  printf("max_value=%f, total_exp_value=%f\n", max_val, total_exp_value);
    //}
    edge_weight_softmax[row_ptr_id * num_head + head_id] = (T)(value);
  }
}

template<typename T = float>
void EdgeWeightSoftmaxForwardCommon(const int *csr_row_ptr,
                                    int64_t target_vec_count,
                                    const void *edge_weight,
                                    int num_head,
                                    void *edge_weight_softmax,
                                    cudaStream_t stream) {

  assert(num_head <= 512);
  int row_num_per_block = 512 / num_head;
  dim3 block(num_head, row_num_per_block);
  int block_count = DivUp(target_vec_count, row_num_per_block);
  EdgeWeightSoftmaxForwardSimpleKernel<T><<<block_count, block, 0, stream>>>(csr_row_ptr,
                                                                             target_vec_count,
                                                                             (const T *) edge_weight,
                                                                             num_head,
                                                                             (T *) edge_weight_softmax);
}

REGISTER_DISPATCH_ONE_TYPE(EdgeWeightSoftmaxForwardCommon, EdgeWeightSoftmaxForwardCommon, HALF_FLOAT)

void EdgeWeightSoftmaxForward(WMType data_type,
                              const int *csr_row_ptr,
                              const int64_t target_vec_count,
                              const void *edge_weight,
                              const int num_head,
                              void *edge_weight_softmax,
                              cudaStream_t stream) {
  DISPATCH_ONE_TYPE(data_type, EdgeWeightSoftmaxForwardCommon,
                    csr_row_ptr,
                    target_vec_count,
                    edge_weight,
                    num_head,
                    edge_weight_softmax,
                    stream);
}

template<typename T = float>
__global__ void EdgeWeightSoftmaxBackwardSimpleKernel(const int *csr_row_ptr,
                                                      const T *edge_weight,
                                                      const T *grad_y,
                                                      const int64_t target_vec_count,
                                                      const int num_head,
                                                      T *grad_weight_softmax) {

  int row_id = blockIdx.x * blockDim.y + threadIdx.y;
  int head_id = threadIdx.x;
  if (row_id >= target_vec_count) return;

  int row_ptr_start = csr_row_ptr[row_id];
  int row_ptr_end = csr_row_ptr[row_id + 1];

  float max_val = -1e38f;
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    float value = (float) edge_weight[row_ptr_id * num_head + head_id];
    max_val = max(max_val, value);
  }
  float total_exp_value = 0.0f;
  float total_exp_dy_value = 0.0f;
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    float value = (float) edge_weight[row_ptr_id * num_head + head_id];
    float value_exp = __expf(value - max_val);
    total_exp_value += value_exp;
    total_exp_dy_value += value_exp * (float) grad_y[row_ptr_id * num_head + head_id];
  }
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    float y = __expf((float) edge_weight[row_ptr_id * num_head + head_id] - max_val) / total_exp_value;
    float dy = (float) grad_y[row_ptr_id * num_head + head_id];
    float dx = y * (dy - total_exp_dy_value / total_exp_value);
    grad_weight_softmax[row_ptr_id * num_head + head_id] = dx;
  }
}

template<typename T = float>
void EdgeWeightSoftmaxBackwardCommon(const int *csr_row_ptr,
                                     const void *edge_weight_softmax,
                                     const void *grad_y,
                                     int64_t target_vec_count,
                                     int num_head,
                                     void *grad_weight_softmax,
                                     cudaStream_t stream) {

  assert(num_head <= 512);

  int rows_per_block = 512 / num_head;
  dim3 block(num_head, rows_per_block);
  int block_count = DivUp(target_vec_count, rows_per_block);

  EdgeWeightSoftmaxBackwardSimpleKernel<T><<<block_count, block, 0, stream>>>(csr_row_ptr,
                                                                              (const T *) edge_weight_softmax,
                                                                              (const T *) grad_y,
                                                                              target_vec_count,
                                                                              num_head,
                                                                              (T *) grad_weight_softmax);
}

REGISTER_DISPATCH_ONE_TYPE(EdgeWeightSoftmaxBackwardCommon, EdgeWeightSoftmaxBackwardCommon, HALF_FLOAT)

void EdgeWeightSoftmaxBackward(WMType data_type,
                               const int *csr_row_ptr,
                               const void *edge_weight_softmax,
                               const void *grad_y,
                               const int64_t target_vec_count,
                               const int num_head,
                               void *grad_weight_softmax,
                               cudaStream_t stream) {
  DISPATCH_ONE_TYPE(data_type, EdgeWeightSoftmaxBackwardCommon,
                    csr_row_ptr,
                    edge_weight_softmax,
                    grad_y,
                    target_vec_count,
                    num_head,
                    grad_weight_softmax,
                    stream);
}

__global__ void AddSelfLoopKernel(const int *csr_row_ptr,
                                  const int *csr_col_ind,
                                  const int *sample_dup_count,
                                  int *csr_row_ptr_looped,
                                  int *csr_col_ind_looped,
                                  int *sample_dup_count_looped) {
  int row_idx = blockIdx.x;
  int row_ptr_start = csr_row_ptr[row_idx];
  int row_ptr_end = csr_row_ptr[row_idx + 1];
  if (threadIdx.x == 0) {
    csr_row_ptr_looped[row_idx] = row_ptr_start + row_idx;
    if (sample_dup_count_looped != nullptr) {
      sample_dup_count_looped[row_idx] = sample_dup_count != nullptr ? sample_dup_count[row_idx] + 1 : 0;
    }
    if (blockIdx.x == gridDim.x - 1) {
      csr_row_ptr_looped[row_idx + 1] = row_ptr_end + row_idx + 1;
    }
  }
  for (int nidx = threadIdx.x; nidx <= row_ptr_end - row_ptr_start; nidx += blockDim.x) {
    int neighbor_idx = row_idx;
    if (nidx > 0) {
      neighbor_idx = csr_col_ind[row_ptr_start + nidx - 1];
    }
    csr_col_ind_looped[row_ptr_start + row_idx + nidx] = neighbor_idx;
  }
}

void CSRAddSelfLoop(const int *csr_row_ptr,
                    const int *csr_col_ind,
                    const int *sample_dup_count,
                    int total_target_count,
                    int unique_neighbor_and_target_count,
                    int *csr_row_ptr_looped,
                    int *csr_col_ind_looped,
                    int *sample_dup_count_looped,
                    cudaStream_t stream) {
  WM_CUDA_CHECK(cudaMemcpyAsync(sample_dup_count_looped + total_target_count,
                                sample_dup_count + total_target_count,
                                sizeof(int) * (unique_neighbor_and_target_count - total_target_count),
                                cudaMemcpyDeviceToDevice,
                                stream));
  AddSelfLoopKernel<<<total_target_count, 64, 0, stream>>>(
      csr_row_ptr,
      csr_col_ind,
      sample_dup_count,
      csr_row_ptr_looped,
      csr_col_ind_looped,
      sample_dup_count_looped);
}

template<typename ParamT, typename IdxT, typename ParamHandleT, typename GraphRowHandleT, typename GraphColHandleT, typename ToTypedHandleT>
__global__ void MixedGraphSGCKernel(ParamHandleT *param,
                                    GraphRowHandleT *csr_row_ptr,
                                    GraphColHandleT *csr_col_idx,
                                    ToTypedHandleT *to_typed,
                                    int target_type,
                                    int neighbor_type,
                                    int64_t storage_offset,
                                    int64_t embedding_dim,
                                    int64_t embedding_stride,
                                    int64_t start_mix_id,
                                    int64_t end_mix_id) {
  int64_t mix_id = blockIdx.x + start_mix_id;
  PtrGen<ToTypedHandleT, int64_t> to_typed_gen(to_typed);
  PtrGen<ParamHandleT, ParamT> param_gen(param, storage_offset);
  PtrGen<GraphRowHandleT, int64_t> csr_row_ptr_gen(csr_row_ptr);
  PtrGen<GraphColHandleT, IdxT> csr_col_idx_gen(csr_col_idx);
  extern __shared__ float buffer[];
  for (; mix_id < end_mix_id; mix_id += gridDim.x) {
    int64_t typed_id = *to_typed_gen.At(mix_id);
    if (TypeOfTypedID(typed_id) != target_type) continue;
    int64_t cols_start = *csr_row_ptr_gen.At(mix_id);
    int64_t cols_end = *csr_row_ptr_gen.At(mix_id + 1);
    for (int idx = threadIdx.x; idx < embedding_dim; idx += blockDim.x) {
      buffer[idx] = 0.0f;
    }
    __syncthreads();
    int count = 0;
    for (int64_t cols_idx = cols_start; cols_idx < cols_end; cols_idx++) {
      IdxT neighbor_mixid = *csr_col_idx_gen.At(cols_idx);
      int64_t neighbor_typed_id = *to_typed_gen.At(neighbor_mixid);
      if (TypeOfTypedID(neighbor_typed_id) != neighbor_type) continue;
      for (int idx = threadIdx.x; idx < embedding_dim; idx += blockDim.x) {
        ParamT fvalue = *param_gen.At(neighbor_mixid * embedding_stride + idx);
        buffer[idx] += (float) fvalue;
      }
      count++;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < embedding_dim; idx += blockDim.x) {
      float fvalue = count > 0 ? buffer[idx] / count : 0.0f;
      *param_gen.At(mix_id * embedding_stride + idx) = (ParamT) fvalue;
    }
    __syncthreads();
  }
}

template<typename ParamT, typename IdxT>
void MixedGraphSGCFunc(void *param,
                       const int64_t *csr_row_ptr,
                       const void *csr_col_idx,
                       const int64_t *to_typed,
                       int target_type,
                       int neighbor_type,
                       int64_t storage_offset,
                       int64_t embedding_dim,
                       int64_t embedding_stride,
                       int64_t node_count,
                       cudaStream_t stream) {
  int shared_memory_size = embedding_dim * sizeof(float);
  auto *bc_ptr = WmmpGetBootstrapCommunicator(csr_row_ptr);
  int rank = bc_ptr->Rank();
  int size = bc_ptr->Size();
  int64_t mix_start = rank * node_count / size;
  int64_t mix_end = (rank + 1) * node_count / size;
  int max_block_count = 65536;
  int block_count = max_block_count;
  if (mix_end - mix_start < (int64_t) max_block_count) {
    block_count = (int) (mix_end - mix_start);
  }
  int thread_count = embedding_dim;
  if (thread_count > 256) thread_count = 256;
  WM_CHECK(storage_offset == 0);
  MixedGraphSGCKernel<ParamT, IdxT, ParamT, const int64_t, const IdxT, const int64_t>
      <<<block_count, thread_count, shared_memory_size, stream>>>((ParamT *) param,
                                                                  csr_row_ptr,
                                                                  (const IdxT *) csr_col_idx,
                                                                  to_typed,
                                                                  target_type,
                                                                  neighbor_type,
                                                                  storage_offset,
                                                                  embedding_dim,
                                                                  embedding_stride,
                                                                  mix_start,
                                                                  mix_end);
  WM_CUDA_CHECK(cudaGetLastError());
}

REGISTER_DISPATCH_TWO_TYPES(MixedGraphSGCFunc, MixedGraphSGCFunc, HALF_FLOAT_DOUBLE, SINT3264)

void MixedGraphSGC(WMType param_type,
                   WMType id_type,
                   void *param,
                   const int64_t *csr_row_ptr,
                   const void *csr_col_idx,
                   const int64_t *to_typed,
                   int target_type,
                   int neighbor_type,
                   int64_t storage_offset,
                   int64_t embedding_dim,
                   int64_t embedding_stride,
                   int64_t node_count,
                   cudaStream_t stream) {
  DISPATCH_TWO_TYPES(param_type, id_type, MixedGraphSGCFunc,
                     param,
                     csr_row_ptr,
                     csr_col_idx,
                     to_typed,
                     target_type,
                     neighbor_type,
                     storage_offset,
                     embedding_dim,
                     embedding_stride,
                     node_count,
                     stream);
}

template<typename ParamT, typename IdxT>
void MixedGraphSGCChunkedFunc(WholeChunkedMemory_t param,
                              WholeChunkedMemory_t csr_row_ptr,
                              WholeChunkedMemory_t csr_col_idx,
                              WholeChunkedMemory_t to_typed,
                              int target_type,
                              int neighbor_type,
                              int64_t storage_offset,
                              int64_t embedding_dim,
                              int64_t embedding_stride,
                              int64_t node_count,
                              cudaStream_t stream) {
  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WholeChunkedMemoryHandle *param_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) param, dev_id);
  WholeChunkedMemoryHandle *row_ptr_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) csr_row_ptr, dev_id);
  WholeChunkedMemoryHandle *col_idx_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) csr_col_idx, dev_id);
  WholeChunkedMemoryHandle *to_typed_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) to_typed, dev_id);
  int shared_memory_size = embedding_dim * sizeof(float);
  auto *bc_ptr = WcmmpGetBootstrapCommunicator(csr_row_ptr);
  int rank = bc_ptr->Rank();
  int size = bc_ptr->Size();
  int64_t mix_start = rank * node_count / size;
  int64_t mix_end = (rank + 1) * node_count / size;
  int max_block_count = 65536;
  int block_count = max_block_count;
  if (mix_end - mix_start < (int64_t) max_block_count) {
    block_count = (int) (mix_end - mix_start);
  }
  int thread_count = embedding_dim;
  if (thread_count > 256) thread_count = 256;
  MixedGraphSGCKernel<ParamT, IdxT, WholeChunkedMemoryHandle, const WholeChunkedMemoryHandle,
                      const WholeChunkedMemoryHandle, const WholeChunkedMemoryHandle>
      <<<block_count, thread_count, shared_memory_size, stream>>>(param_handle,
                                                                  row_ptr_handle,
                                                                  col_idx_handle,
                                                                  to_typed_handle,
                                                                  target_type,
                                                                  neighbor_type,
                                                                  storage_offset,
                                                                  embedding_dim,
                                                                  embedding_stride,
                                                                  mix_start,
                                                                  mix_end);
  WM_CUDA_CHECK(cudaGetLastError());
}
REGISTER_DISPATCH_TWO_TYPES(MixedGraphSGCChunkedFunc, MixedGraphSGCChunkedFunc, HALF_FLOAT_DOUBLE, SINT3264)

void MixedGraphSGCChunked(WMType param_type,
                          WMType id_type,
                          WholeChunkedMemory_t param,
                          WholeChunkedMemory_t csr_row_ptr,
                          WholeChunkedMemory_t csr_col_idx,
                          WholeChunkedMemory_t to_typed,
                          int target_type,
                          int neighbor_type,
                          int64_t storage_offset,
                          int64_t embedding_dim,
                          int64_t embedding_stride,
                          int64_t node_count,
                          cudaStream_t stream) {
  DISPATCH_TWO_TYPES(param_type, id_type, MixedGraphSGCChunkedFunc,
                     param,
                     csr_row_ptr,
                     csr_col_idx,
                     to_typed,
                     target_type,
                     neighbor_type,
                     storage_offset,
                     embedding_dim,
                     embedding_stride,
                     node_count,
                     stream);
}

static const int kMaxRelationCount = 32;
static const int kRGCNAggeratorSum = 0;
static const int kRGCNAggeratorStart = kRGCNAggeratorSum;
static const int kRGCNAggeratorMean = 1;
static const int kRGCNAggeratorEnd = kRGCNAggeratorMean + 1;

template<typename T = float>
__global__ void SpmmCsrRelationalNoWeightForwardKernel(const int *csr_row_ptr,
                                                       const int *csr_col_ind,
                                                       const int8_t *edge_type,
                                                       const T *x,
                                                       int64_t embedding_dim,
                                                       int64_t embedding_stride,
                                                       int64_t input_count,
                                                       T *output,
                                                       int64_t output_row_stride,
                                                       int64_t relation_num,
                                                       int aggregator) {
  __shared__ int relation_vec_count[kMaxRelationCount];
  extern __shared__ float relation_vec[];
  int row_idx = blockIdx.x;
  int row_ptr_start = csr_row_ptr[row_idx];
  int row_ptr_end = csr_row_ptr[row_idx + 1];
  int agg_count = row_ptr_end - row_ptr_start;
  for (int idx = threadIdx.x; idx < kMaxRelationCount; idx += blockDim.x) {
    relation_vec_count[idx] = 0;
  }
  __syncthreads();
  if (aggregator == kRGCNAggeratorMean && threadIdx.x < 32) {
    for (int neighbor_idx = threadIdx.x; neighbor_idx < AlignUp(agg_count, 32); neighbor_idx += 32) {
      int edge_type_id = -1;
      if (neighbor_idx < agg_count) edge_type_id = (int) edge_type[row_ptr_start + neighbor_idx];
      assert(edge_type_id >= -1 && edge_type_id < relation_num);
      if (edge_type_id >= 0) atomicAdd_block(&relation_vec_count[edge_type_id], 1);
    }
  }
  __syncthreads();
  for (int emb_start = 0; emb_start < embedding_dim; emb_start += blockDim.x) {
    for (int r = 0; r < relation_num; r++) {
      if (emb_start + threadIdx.x < embedding_dim) relation_vec[r * blockDim.x + threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int row_ptr = row_ptr_start; row_ptr < row_ptr_end; row_ptr++) {
      int edge_type_id = (int) edge_type[row_ptr];
      assert(edge_type_id >= 0 && edge_type_id < relation_num);
      int col_idx = csr_col_ind[row_ptr];
      assert(col_idx >= 0 && col_idx < input_count);
      if (emb_start + threadIdx.x < embedding_dim) {
        float value = (float) x[(int64_t) col_idx * embedding_stride + emb_start + threadIdx.x];
        relation_vec[edge_type_id * blockDim.x + threadIdx.x] += value;
      }
      __syncthreads();
    }
    for (int r = 0; r < relation_num; r++) {
      int count = relation_vec_count[r];
      if (count == 0) count = 1;
      if (emb_start + threadIdx.x < embedding_dim) {
        output[(int64_t) row_idx * output_row_stride + r * embedding_dim + emb_start + threadIdx.x] =
            (T)(relation_vec[r * blockDim.x + threadIdx.x] / (float) count);
      }
    }
    __syncthreads();
  }
}

template<typename T>
void SpmmCsrRelationalNoWeightForwardCommon(const int *csr_row_ptr,
                                            int64_t row_count,
                                            const int *csr_col_ind,
                                            const int8_t *edge_type,
                                            const void *x,
                                            int64_t embedding_dim,
                                            int64_t embedding_stride,
                                            int64_t input_count,
                                            void *output,
                                            int64_t output_row_stride,
                                            int64_t relation_num,
                                            int aggregator,
                                            cudaStream_t stream) {
  int block_count = row_count;
  // Now only support simple
  int thread_count = AlignUp(embedding_dim, 32);
  if (thread_count > 256) {
    thread_count = 256;
  }
  int dynamic_shared_memory_size = sizeof(float) * relation_num * thread_count;
  SpmmCsrRelationalNoWeightForwardKernel<T><<<block_count, thread_count, dynamic_shared_memory_size, stream>>>(
      csr_row_ptr,
      csr_col_ind,
      edge_type,
      (const T *) x,
      embedding_dim,
      embedding_stride,
      input_count,
      (T *) output,
      output_row_stride,
      relation_num,
      aggregator);
  WM_CUDA_CHECK(cudaGetLastError());
}

REGISTER_DISPATCH_ONE_TYPE(SpmmCsrRelationalNoWeightForwardCommon, SpmmCsrRelationalNoWeightForwardCommon, HALF_FLOAT)

void SpmmCsrRelationalNoWeightForward(WMType data_type,
                                      const int *csr_row_ptr,
                                      int m,// total row count
                                      const int *csr_col_ind,
                                      int total_nz_element,
                                      const int8_t *edge_type,
                                      const void *input,
                                      int n,// embedding dim
                                      int ldb,
                                      int k,// total column count
                                      void *output,
                                      int ldc,
                                      int num_relations,
                                      int aggregator,
                                      cudaStream_t stream) {
  WM_CHECK(num_relations < kMaxRelationCount);
  WM_CHECK(aggregator >= kRGCNAggeratorStart && aggregator < kRGCNAggeratorEnd);
  DISPATCH_ONE_TYPE(data_type, SpmmCsrRelationalNoWeightForwardCommon,
                    csr_row_ptr,
                    m,
                    csr_col_ind,
                    edge_type,
                    input,
                    n,
                    ldb,
                    k,
                    output,
                    ldc,
                    num_relations,
                    aggregator,
                    stream);
}

template<typename T = float>
__global__ void SpmmCsrRelationalNoWeightAtomicBackwardInputKernel(const int *csr_row_ptr,
                                                                   const int *csr_col_ind,
                                                                   const int8_t *edge_type,
                                                                   const int *sample_dup_count,
                                                                   const T *grad_output,
                                                                   int64_t grad_out_stride,
                                                                   T *grad_x,
                                                                   int64_t grad_x_stride,
                                                                   int64_t input_count,
                                                                   int64_t embedding_dim,
                                                                   int relation_num,
                                                                   int aggregator) {
  __shared__ int relation_vec_count[kMaxRelationCount];
  extern __shared__ float relation_vec[];
  int row_idx = blockIdx.x;
  int row_ptr_start = csr_row_ptr[row_idx];
  int row_ptr_end = csr_row_ptr[row_idx + 1];
  int agg_count = row_ptr_end - row_ptr_start;
  for (int idx = threadIdx.x; idx < kMaxRelationCount; idx += blockDim.x) {
    relation_vec_count[idx] = 0;
  }
  __syncthreads();
  if (aggregator == kRGCNAggeratorMean && threadIdx.x < 32) {
    for (int neighbor_idx = threadIdx.x; neighbor_idx < AlignUp(agg_count, 32); neighbor_idx += 32) {
      int edge_type_id = -1;
      if (neighbor_idx < agg_count) edge_type_id = (int) edge_type[row_ptr_start + neighbor_idx];
      assert(edge_type_id >= -1 && edge_type_id < relation_num);
      if (edge_type_id >= 0) atomicAdd_block(&relation_vec_count[edge_type_id], 1);
    }
  }
  __syncthreads();
  for (int emb_start = 0; emb_start < embedding_dim; emb_start += blockDim.x) {
    for (int r = 0; r < relation_num; r++) {
      if (emb_start + threadIdx.x < embedding_dim)
        relation_vec[r * blockDim.x + threadIdx.x] =
            (float) grad_output[(int64_t) row_idx * grad_out_stride
                                + r * embedding_dim + emb_start + threadIdx.x];
    }
    __syncthreads();
    for (int row_ptr = row_ptr_start; row_ptr < row_ptr_end; row_ptr++) {
      int edge_type_id = (int) edge_type[row_ptr];
      assert(edge_type_id >= 0 && edge_type_id < relation_num);
      int col_idx = csr_col_ind[row_ptr];
      assert(col_idx >= 0 && col_idx < input_count);
      int dup_count = sample_dup_count[col_idx];
      int count = relation_vec_count[edge_type_id];
      if (count == 0) count = 1;
      float value = relation_vec[edge_type_id * blockDim.x + threadIdx.x] / (float) count;
      if (emb_start + threadIdx.x < embedding_dim) {
        if (dup_count == 1) {
          grad_x[(int64_t) col_idx * grad_x_stride + emb_start + threadIdx.x] = (T) value;
        } else {
          atomicAdd(&grad_x[(int64_t) col_idx * grad_x_stride + emb_start + threadIdx.x], (T) value);
        }
      }
    }
    __syncthreads();
  }
}

template<typename T = float>
void SpmmCsrRelationalNoWeightBackwardCommon(const int *csr_row_ptr,
                                             const int *csr_col_ind,
                                             const int8_t *edge_type,
                                             const int *sample_dup_count,
                                             const void *grad_output,
                                             int64_t grad_out_stride,
                                             void *grad_x,
                                             int64_t grad_x_stride,
                                             int64_t m,
                                             int64_t total_nz_element,
                                             int64_t k,
                                             int64_t embedding_dim,
                                             int64_t relation_num,
                                             int aggregator,
                                             cudaStream_t stream) {
  int block_count = m;
  // Now only support simple
  int thread_count = AlignUp(embedding_dim, 32);
  if (thread_count > 256) {
    thread_count = 256;
  }
  int dynamic_shared_memory_size = sizeof(float) * relation_num * thread_count;

  SpmmCsrNoWeightAtomicBackwardFillKernel<T><<<k, thread_count, 0, stream>>>(
      (T *) grad_x, grad_x_stride, sample_dup_count, embedding_dim);

  SpmmCsrRelationalNoWeightAtomicBackwardInputKernel<T><<<block_count, thread_count, dynamic_shared_memory_size,
                                                          stream>>>(
      csr_row_ptr,
      csr_col_ind,
      edge_type,
      sample_dup_count,
      (const T *) grad_output,
      grad_out_stride,
      (T *) grad_x,
      grad_x_stride,
      k,
      embedding_dim,
      relation_num,
      aggregator);
  WM_CUDA_CHECK(cudaGetLastError());
}

REGISTER_DISPATCH_ONE_TYPE(SpmmCsrRelationalNoWeightBackwardCommon, SpmmCsrRelationalNoWeightBackwardCommon, HALF_FLOAT)

void SpmmCsrRelationalNoWeightBackward(WMType data_type,
                                       const int *csr_row_ptr,
                                       const int *csr_col_ind,
                                       const int8_t *edge_type,
                                       const int *sample_dup_count,
                                       const void *grad_output,
                                       int grad_out_stride,
                                       void *grad_x,
                                       int grad_x_stride,
                                       int m,// total row count,
                                       int total_nz_element,
                                       int k,// total column count
                                       int embedding_dim,
                                       int num_relations,
                                       int aggregator,
                                       cudaStream_t stream) {
  WM_CHECK(num_relations < kMaxRelationCount);
  WM_CHECK(aggregator >= kRGCNAggeratorStart && aggregator < kRGCNAggeratorEnd);
  DISPATCH_ONE_TYPE(data_type, SpmmCsrRelationalNoWeightBackwardCommon,
                    csr_row_ptr,
                    csr_col_ind,
                    edge_type,
                    sample_dup_count,
                    grad_output,
                    (int64_t) grad_out_stride,
                    grad_x,
                    (int64_t) grad_x_stride,
                    (int64_t) m,
                    (int64_t) total_nz_element,
                    (int64_t) k,
                    (int64_t) embedding_dim,
                    (int64_t) num_relations,
                    aggregator,
                    stream);
}

}// namespace whole_graph