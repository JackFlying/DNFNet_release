#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <vector>
static void HandleError(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess)
  {
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
           file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void search_negative(float **score_mat, int *cidxs_fl, int *cidxs_starts, int **results, int mat_h, int mat_w, int n_clusters)
{
  int x = blockIdx.x; // x for index of cluster
  int y = blockIdx.y; // y for index of rows in the mat
  float *samp_scores_arr = score_mat[y];
  int *cidxs_arr = cidxs_fl + cidxs_starts[x];
  int cidxs_len = cidxs_starts[x + 1] - cidxs_starts[x];
  int min_idx = cidxs_arr[0];
  float min_score = samp_scores_arr[min_idx];
  for (int i = 1; i < cidxs_len; i++)
  {
    int cur_idx = cidxs_arr[i];
    float cur_score = samp_scores_arr[cur_idx];
    if (cur_score < min_score)
    {
      min_idx = cur_idx;
      min_score = cur_score;
    }
  }
  results[y][x] = min_idx;
}

void run_kernel_negative(float **gpu_mat_ptr, int *gpu_cidxs_fl_ptr, int *gpu_cidxs_starts_ptr, int **gpu_res, int mat_h, int mat_w, int n_clusters)
{
  // dim3 dimGrid(bb_h); //(w,h)
  // dim3 dimBlk(dl_h);
  // cost<<<dimGrid, dimBlk>>>(gpu_dl_ptr, gpu_bb_ptr, gpu_res, dl_w, bb_w, p, k);
  dim3 dimGrid(n_clusters, mat_h);
  // dim3 dimGrid(1, 1);
  search_negative<<<dimGrid, 1>>>(gpu_mat_ptr, gpu_cidxs_fl_ptr, gpu_cidxs_starts_ptr, gpu_res, mat_h, mat_w, n_clusters);
  // cudaError_t error = cudaGetLastError();
  // HANDLE_ERROR()
}

float **copy_mat_to_dev(pybind11::array_t<float> mat, float *gpu_fl_ptr, int *mat_h_ptr, int *mat_w_ptr)
{
  pybind11::buffer_info ha = mat.request();
  int h = ha.shape[0];
  int w = ha.shape[1];
  *mat_h_ptr = h;
  *mat_w_ptr = w;
  float **gpu_mat_ptr;
  float **gpu_mat_hptr;
  // int *gpu_data;
  float *host_data;
  // malloc
  HANDLE_ERROR(cudaMalloc((void **)(&gpu_mat_ptr), sizeof(float *) * h));
  HANDLE_ERROR(cudaMalloc((void **)(&gpu_fl_ptr), h * w * sizeof(float)));
  gpu_mat_hptr = (float **)malloc(sizeof(float *) * h);
  for (int i = 0; i < h; i++)
  {
    gpu_mat_hptr[i] = gpu_fl_ptr + i * w;
  }
  // copy

  host_data = (float *)malloc(sizeof(float) * h * w);
  for (int i = 0; i < h; i++)
  {
    for (int j = 0; j < w; j++)
    {
      host_data[i * w + j] = mat.at(i, j);
    }
  }
  // data: cpu-> gpu
  HANDLE_ERROR(cudaMemcpy((void *)(gpu_fl_ptr), (void *)(host_data), sizeof(float) * w * h, cudaMemcpyHostToDevice));
  // addr ptrs: cpu->gpu
  HANDLE_ERROR(cudaMemcpy((void *)(gpu_mat_ptr), (void *)(gpu_mat_hptr), h * sizeof(float *), cudaMemcpyHostToDevice));
  free(gpu_mat_hptr);
  free(host_data);
  return gpu_mat_ptr;
}
int *copy_cidxs_to_dev(std::vector<std::vector<int>> cidxs, int **gpu_cidxs_starts_ptr)
{
  int *gpu_cidxs_fl_ptr;
  int all_samps = 0;
  int n_clusters = cidxs.size();
  int *cidxs_starts = (int *)malloc(sizeof(int) * (n_clusters + 1));
  cidxs_starts[0] = 0;
  for (int i = 0; i < n_clusters; i++)
  {
    int c_len = cidxs[i].size();
    all_samps += c_len;
    cidxs_starts[i + 1] = cidxs_starts[i] + c_len;
  }
  // int *gpu_data;
  int *host_data;
  // malloc
  HANDLE_ERROR(cudaMalloc((void **)(gpu_cidxs_starts_ptr), sizeof(int) * (n_clusters + 1)));
  HANDLE_ERROR(cudaMalloc((void **)(&gpu_cidxs_fl_ptr), all_samps * sizeof(int)));

  host_data = (int *)malloc(sizeof(int) * all_samps);
  int next_in_host_data = 0;
  for (int i = 0; i < n_clusters; i++)
  {
    int cidxs_len = cidxs_starts[i + 1] - cidxs_starts[i];
    for (int j = 0; j < cidxs_len; j++)
    {
      host_data[next_in_host_data] = cidxs[i][j];
      next_in_host_data++;
    }
  }
  // data: cpu-> gpu
  HANDLE_ERROR(cudaMemcpy((void *)(gpu_cidxs_fl_ptr), (void *)(host_data), sizeof(int) * all_samps, cudaMemcpyHostToDevice));
  // starts: cpu->gpu
  HANDLE_ERROR(cudaMemcpy((void *)(*gpu_cidxs_starts_ptr), (void *)(cidxs_starts), sizeof(int) * (n_clusters + 1), cudaMemcpyHostToDevice));
  free(host_data);
  free(cidxs_starts);
  return gpu_cidxs_fl_ptr;
}
pybind11::array_t<int> min_negtive_idxs(pybind11::array_t<float> score_mat, std::vector<std::vector<int>> cluster_idxs)
{
  int mat_h, mat_w;
  float **gpu_mat_ptr;
  float *gpu_mat_fl_ptr;
  int *gpu_cidxs_fl_ptr;
  int *gpu_cidxs_starts_ptr;
  int **gpu_res;
  int **res;
  int *gpu_res_data;
  int *res_data;
  //gpu_mat_ptr = copy_mat_to_dev(score_mat, gpu_mat_fl_ptr, &mat_h, &mat_w);
  //copy_mat_to_dev
  pybind11::buffer_info ha = score_mat.request();
  mat_h = ha.shape[0];
  mat_w = ha.shape[1];
  float **gpu_mat_hptr;
  // int *gpu_data;
  float *host_data;
  // malloc
  HANDLE_ERROR(cudaMalloc((void **)(&gpu_mat_ptr), sizeof(float *) * mat_h));
  HANDLE_ERROR(cudaMalloc((void **)(&gpu_mat_fl_ptr), mat_h * mat_w * sizeof(float)));
  gpu_mat_hptr = (float **)malloc(sizeof(float *) * mat_h);
  for (int i = 0; i < mat_h; i++)
  {
    gpu_mat_hptr[i] = gpu_mat_fl_ptr + i * mat_w;
  }
  // copy

  host_data = (float *)malloc(sizeof(float) * mat_h * mat_w);
  for (int i = 0; i < mat_h; i++)
  {
    for (int j = 0; j < mat_w; j++)
    {
      host_data[i * mat_w + j] = score_mat.at(i, j);
    }
  }
  // data: cpu-> gpu
  HANDLE_ERROR(cudaMemcpy((void *)(gpu_mat_fl_ptr), (void *)(host_data), sizeof(float) * mat_w * mat_h, cudaMemcpyHostToDevice));
  // addr ptrs: cpu->gpu
  HANDLE_ERROR(cudaMemcpy((void *)(gpu_mat_ptr), (void *)(gpu_mat_hptr), mat_h * sizeof(float *), cudaMemcpyHostToDevice));
  free(gpu_mat_hptr);
  free(host_data);
  // end of copy_mat_to_dev
  
  gpu_cidxs_fl_ptr = copy_cidxs_to_dev(cluster_idxs, &gpu_cidxs_starts_ptr);
  int n_clusters = cluster_idxs.size();

  // init res list on device
  HANDLE_ERROR(cudaMalloc((void **)(&gpu_res), sizeof(int *) * mat_h));
  HANDLE_ERROR(cudaMalloc((void **)(&gpu_res_data), mat_h * n_clusters * sizeof(int)));
  res = (int **)malloc(sizeof(int *) * mat_h);
  for (int i = 0; i < mat_h; i++)
  {
    res[i] = gpu_res_data + i * n_clusters;
  }
  HANDLE_ERROR(cudaMemcpy((void *)(gpu_res), (void *)(res), mat_h * sizeof(int *), cudaMemcpyHostToDevice));
  // min search
  run_kernel_negative(gpu_mat_ptr, gpu_cidxs_fl_ptr, gpu_cidxs_starts_ptr, gpu_res, mat_h, mat_w, n_clusters);
  // init res mat on host
  res_data = (int *)malloc(mat_h * n_clusters * sizeof(int));
  HANDLE_ERROR(cudaMemcpy((void *)(res_data), (void *)(gpu_res_data), mat_h * n_clusters * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < mat_h; i++)
  {
    res[i] = res_data + i * n_clusters;
  }
  pybind11::array_t<int> res_arr = pybind11::array_t<int>(mat_h * n_clusters);
  pybind11::buffer_info buf = res_arr.request();
  int *ptr_res = (int *)buf.ptr;
  for (int i = 0; i < mat_h; i++)
  {
    for (int j = 0; j < n_clusters; j++)
    {
      // printf("%d %d %f\n", i, j, res[i][j]);
      ptr_res[i * n_clusters + j] = res[i][j];
    }
  }
  res_arr.resize({mat_h, n_clusters});
  // free
  // gpu_mat_free(gpu_dl_ptr);
  // gpu_mat_free(gpu_bb_ptr);
  cudaFree((void *)(gpu_mat_fl_ptr));
  cudaFree((void *)(gpu_mat_ptr));
  cudaFree((void *)(gpu_cidxs_fl_ptr));
  cudaFree((void *)(gpu_cidxs_starts_ptr));
  cudaFree((void *)(gpu_res_data));
  cudaFree((void *)(gpu_res));
  free(res_data);
  free(res);
  return res_arr;
}

PYBIND11_MODULE(min_search_cuda, m)
{
  m.def("min_negative_search", min_negtive_idxs, pybind11::return_value_policy::copy);
}
