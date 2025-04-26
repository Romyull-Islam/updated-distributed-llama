#ifndef NN_CUDA_HPP
#define NN_CUDA_HPP

#include "nn.hpp"
#include "nn-core.hpp"
#include "nn-executor.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

class NnCudaOpForward;
class NnCudaOpContext;

class NnCudaDeviceSegment : public NnDeviceSegment {
public:
    NnCudaDeviceSegment(NnDevice *device, NnUint segmentIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution);
    ~NnCudaDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
    void execute(NnNetExecution *execution, NnUint batchIndex);

private:
    cudaStream_t stream;
    cublasHandle_t cublasHandle;
    NnByte **deviceBuffers; // Per-operation weight buffers
    NnUint nOps;
    NnCudaOpForward **opForward;
    NnCudaOpContext **opContexts;
    NnDevice *device;
};

class NnCudaDevice : public NnDevice {
public:
    NnCudaDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution);
    ~NnCudaDevice() override;
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;
    NnUint maxNThreads() override { return 1; } // CUDA is single-threaded per stream
    void resolvePointer(NnByte **pntr, NnSize2D *pntrSize, NnPointerConfig *pointerConfig, NnUint batchIndex);

private:
    int gpuIndex;
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
    NnNetExecution *netExecution;
    NnByte **buffers; // Device buffers for pipes and node buffers
    NnUint nBuffers;
};

#endif
