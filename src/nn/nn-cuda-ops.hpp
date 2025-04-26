#ifndef NN_CUDA_OPS_H
#define NN_CUDA_OPS_H

#include "nn-core.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef struct {
    const char *name;
    NnByte nBatches;
    NnByte **pipes;
    NnPipeConfig *pipeConfigs;
    NnByte **buffers;
    NnBufferConfig *bufferConfigs;
    void *opConfig;
    NnByte *input;
    NnSize2D inputSize;
    bool hasInputContinuousMemory;
    NnByte *output;
    NnSize2D outputSize;
    bool hasOutputContinuousMemory;
    NnByte *weight;
    NnSize2D weightSize;
    cudaStream_t stream;
    cublasHandle_t cublasHandle;
} NnCudaOpContext;

typedef void (*NnCudaOpForwardInit)(NnCudaOpContext *context);
typedef void (*NnCudaOpForward)(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCudaOpContext *context);

NnCudaOpForwardInit getCudaOpForwardInit(NnOpCode code, NnOpQuantType quantType);
NnCudaOpForward getCudaOpForward(NnOpCode code, NnOpQuantType quantType);

#endif
