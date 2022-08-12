#include "builtin_types.h"
#include "cudaProfiler.h"
#include "cuda_api.h"
#include "driver_types.h"
#include "host_defines.h"
// clang-format on
#if (CUDART_VERSION < 8000)
// #include "__cudaFatFormat.h"
#endif
#include "gpgpu_context.h"
// #include "cuda_api_object.h"
// #include "../src/gpgpu-sim/gpu-sim.h"
// #include "../src/cuda-sim/ptx_loader.h"
// #include "../src/cuda-sim/cuda-sim.h"
// #include "../src/cuda-sim/ptx_ir.h"
// #include "../src/cuda-sim/ptx_parser.h"
// #include "../src/gpgpusim_entrypoint.h"
// #include "../src/stream_manager.h"
// #include "../src/abstract_hardware_model.h"
// #include "cuda_runtime_api.hh"

cudaError_t cudaMallocInternal(void **devPtr, size_t size,
                               gpgpu_context *gpgpu_ctx = NULL);
cudaError_t cudaStreamCreateInternal(cudaStream_t *stream,
                                     gpgpu_context *gpgpu_ctx = NULL);
cudaError_t cudaMemcpyInternal(void *dst, const void *src, size_t count,
                               enum cudaMemcpyKind kind,
                               gpgpu_context *gpgpu_ctx = NULL);
cudaError_t cudaConfigureCallInternal(dim3 gridDim, dim3 blockDim,
                                      size_t sharedMem, cudaStream_t stream,
                                      gpgpu_context *gpgpu_ctx = NULL);
cudaError_t cudaSetupArgumentInternal(const void *arg, size_t size,
                                      size_t offset,
                                      gpgpu_context *gpgpu_ctx = NULL);
cudaError_t graphicsLaunchInternal(const char *hostFun, kernel_info_t **kernel, 
                               gpgpu_context *gpgpu_ctx = NULL);
cudaError_t cudaStreamDestroyInternal(cudaStream_t stream,
                                      gpgpu_context *gpgpu_ctx = NULL);
void **cudaRegisterFatBinaryInternal(void *fatCubin,
                                     gpgpu_context *gpgpu_ctx = NULL);
void cudaRegisterFunctionInternal(void **fatCubinHandle, const char *hostFun,
                                  char *deviceFun, const char *deviceName,
                                  int thread_limit, uint3 *tid, uint3 *bid,
                                  dim3 *bDim, dim3 *gDim,
                                  gpgpu_context *gpgpu_ctx = NULL);
cudaError_t cudaLaunchInternal(const char *hostFun,
                               gpgpu_context *gpgpu_ctx = NULL);
void cuobjdump_from_binary(std::string filename);
// TODO: This is not weird at all. rename this
void **weirdRegisterFuntion(void *fatCubin, const char *hostFun,
                            char *deviceFun, const char *ptxfile,
                            const char *ptxinfo, unsigned version,
                            gpgpu_context *gpgpu_ctx = NULL);


cudaError_t malloc_cpy(void **dptr, unsigned size, std::string file);
bool runbfs();