#include <cuda.h>
#include <cassert>
#include <cstdio>

__global__ void warpX(int *in, int *out, int N) {
  if(threadIdx.x > N) return;

  out[threadIdx.x] = in[threadIdx.x] > 128;
}

int main(void) {
  int in[256], out[256];

  int *d_in, *d_out;

  for(int i = 0; i < 256; i++){
    in[i] = random() % 256;
  }

  assert(cudaMalloc(&d_in, sizeof(int) * 1024) == cudaSuccess);
  assert(cudaMalloc(&d_out, sizeof(int) * 256) == cudaSuccess);
  assert(cudaMemcpy(d_in, in, sizeof(int) * 256, cudaMemcpyHostToDevice) == cudaSuccess);

  warpX<<<1, 256>>>(d_in, d_out, 256);

  assert(cudaMemcpy(out, d_out, sizeof(int) * 256, cudaMemcpyDeviceToHost) == cudaSuccess);

  for(int i = 0; i < 256; i++) {
    printf("%d: %d %d %d\n", i, in[i], in[i] > 128, out[i]);
  }
}
