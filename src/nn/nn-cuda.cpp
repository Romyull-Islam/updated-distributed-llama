#include "nn-cuda.hpp"
#include "nn-cuda-ops.hpp"
#include "llm.hpp"
#include "nn-core.hpp"
#include "nn-executor.hpp"
#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

// CUDA kernels
__global__ void siluKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = x / (1.0f + expf(-x));
    }
}

__global__ void invRmsKernel(float *input, float *output, int n, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1) { // Single thread computes RMS
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += input[i] * input[i];
        }
        output[0] = 1.0f / sqrtf(sum / n + epsilon);
    }
}

__global__ void rmsNormKernel(float *input, float *output, float *invRms, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * invRms[0];
    }
}

__global__ void embeddingKernel(float *input, float *weight, float *output, int vocabSize, int dim, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        int token = (int)input[idx];
        if (token >= 0 && token < vocabSize) {
            for (int d = 0; d < dim; ++d) {
                output[idx * dim + d] = weight[token * dim + d];
            }
        }
    }
}

__global__ void mergeAddKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] += input[idx];
    }
}

__global__ void castKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx]; // Placeholder: Handle quantization if needed
    }
}

__global__ void shiftKernel(float *input, float *output, int n, int position) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx]; // Placeholder: Implement KV cache shift
    }
}

__global__ void mulKernel(float *input, float *output, float *other, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * other[idx];
    }
}

__global__ void ropeLlamaKernel(float *data, float *ropeCache, int seqLen, int dim, int batchSize, int position) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * dim / 2) {
        int b = idx / (dim / 2);
        int i = idx % (dim / 2) * 2;
        float x0 = data[b * dim + i];
        float x1 = data[b * dim + i + 1];
        float cosTheta = ropeCache[position * dim + i];
        float sinTheta = ropeCache[position * dim + i + 1];
        data[b * dim + i] = x0 * cosTheta - x1 * sinTheta;
        data[b * dim + i + 1] = x0 * sinTheta + x1 * cosTheta;
    }
}

__global__ void multiHeadAttKernel(float *q, float *k, float *v, float *att, float *output,
                                   int nHeads, int headSize, int seqLen, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * nHeads * seqLen) {
        int b = idx / (nHeads * seqLen);
        int h = (idx % (nHeads * seqLen)) / seqLen;
        int t = idx % seqLen;
        float sum = 0.0f;
        for (int s = 0; s <= t; ++s) {
            float score = 0.0f;
            for (int d = 0; d < headSize; ++d) {
                score += q[b * nHeads * seqLen * headSize + h * seqLen * headSize + t * headSize + d] *
                         k[b * nHeads * seqLen * headSize + h * seqLen * headSize + s * headSize + d];
            }
            score /= sqrtf((float)headSize);
            sum += expf(score);
            att[idx] = sum > 0 ? expf(score) / sum : 0;
        }
        for (int d = 0; d < headSize; ++d) {
            float val = 0.0f;
            for (int s = 0; s <= t; ++s) {
                val += att[idx] * v[b * nHeads * seqLen * headSize + h * seqLen * headSize + s * headSize + d];
            }
            output[b * nHeads * seqLen * headSize + h * seqLen * headSize + t * headSize + d] = val;
        }
    }
}

// Operation forward functions
void matmulForwardInit(NnCudaOpContext *context) {
    context->hasInputContinuousMemory = true;
    context->hasOutputContinuousMemory = true;
}

void matmulForward(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCudaOpContext *context) {
    if (threadIndex != 0) return;
    float alpha = 1.0f, beta = 0.0f;
    int m = context->outputSize.y;
    int n = context->outputSize.x;
    int k = context->inputSize.x;
    cublasStatus_t status = cublasSgemm(context->cublasHandle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        n, m, k,
                                        &alpha,
                                        (float*)context->weight, n,
                                        (float*)context->input, k,
                                        &beta,
                                        (float*)context->output, n);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS SGEMM failed");
    }
    cudaStreamSynchronize(context->stream);
}

void siluForwardInit(NnCudaOpContext *context) {
    context->hasInputContinuousMemory = true;
    context->hasOutputContinuousMemory = true;
}

void siluForward(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCudaOpContext *context) {
    if (threadIndex != 0) return;
    int n = context->inputSize.y * context->inputSize.x * batchSize;
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    siluKernel<<<blocks, threadsPerBlock, 0, context->stream>>>((float*)context->output, n);
    cudaStreamSynchronize(context->stream);
}

void invRmsForwardInit(NnCudaOpContext *context) {
    context->hasInputContinuousMemory = true;
    context->hasOutputContinuousMemory = true;
}

void invRmsForward(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCudaOpContext *context) {
    if (threadIndex != 0) return;
    NnInvRmsOpConfig *config = (NnInvRmsOpConfig*)context->opConfig;
    int n = context->inputSize.x;
    int threadsPerBlock = 256;
    int blocks = 1; // Single block for RMS computation
    invRmsKernel<<<blocks, threadsPerBlock, 0, context->stream>>>(
        (float*)context->input, (float*)context->output, n, config->epsilon);
    cudaStreamSynchronize(context->stream);
}

void rmsNormForwardInit(NnCudaOpContext *context) {
    context->hasInputContinuousMemory = true;
    context->hasOutputContinuousMemory = true;
}

void rmsNormForward(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCudaOpContext *context) {
    if (threadIndex != 0) return;
    NnRmsNormOpConfig *config = (NnRmsNormOpConfig*)context->opConfig;
    float *invRms = nullptr;
    NnSize2D invRmsSize;
    resolvePipeOrBuffer(&invRms, &invRmsSize, context->pipes, context->pipeConfigs,
                       context->buffers, context->bufferConfigs, config->invRmsBufferIndex);
    int n = context->inputSize.x * batchSize;
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    rmsNormKernel<<<blocks, threadsPerBlock, 0, context->stream>>>(
        (float*)context->input, (float*)context->output, invRms, n);
    cudaStreamSynchronize(context->stream);
}

void embeddingForwardInit(NnCudaOpContext *context) {
    context->hasInputContinuousMemory = true;
    context->hasOutputContinuousMemory = true;
}

void embeddingForward(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCudaOpContext *context) {
    if (threadIndex != 0) return;
    int vocabSize = context->weightSize.y;
    int dim = context->weightSize.x;
    int threadsPerBlock = 256;
    int blocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    embeddingKernel<<<blocks, threadsPerBlock, 0, context->stream>>>(
        (float*)context->input, (float*)context->weight, (float*)context->output, vocabSize, dim, batchSize);
    cudaStreamSynchronize(context->stream);
}

void mergeAddForwardInit(NnCudaOpContext *context) {
    context->hasInputContinuousMemory = true;
    context->hasOutputContinuousMemory = true;
}

void mergeAddForward(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCudaOpContext *context) {
    if (threadIndex != 0) return;
    int n = context->inputSize.y * context->inputSize.x * batchSize;
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    mergeAddKernel<<<blocks, threadsPerBlock, 0, context->stream>>>(
        (float*)context->input, (float*)context->output, n);
    cudaStreamSynchronize(context->stream);
}

void castForwardInit(NnCudaOpContext *context) {
    context->hasInputContinuousMemory = true;
    context->hasOutputContinuousMemory = true;
}

void castForward(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCudaOpContext *context) {
    if (threadIndex != 0) return;
    int n = context->inputSize.y * context->inputSize.x * batchSize;
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    castKernel<<<blocks, threadsPerBlock, 0, context->stream>>>(
        (float*)context->input, (float*)context->output, n);
    cudaStreamSynchronize(context->stream);
}

void shiftForwardInit(NnCudaOpContext *context) {
    context->hasInputContinuousMemory = true;
    context->hasOutputContinuousMemory = true;
}

void shiftForward(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCudaOpContext *context) {
    if (threadIndex != 0) return;
    NnShiftOpCodeConfig *config = (NnShiftOpCodeConfig*)context->opConfig;
    float *position = nullptr;
    NnSize2D posSize;
    resolvePipeOrBuffer(&position, &posSize, context->pipes, context->pipeConfigs,
                       context->buffers, context->bufferConfigs, config->positionPipeIndex);
    int n = context->inputSize.y * context->inputSize.x * batchSize;
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    shiftKernel<<<blocks, threadsPerBlock, 0, context->stream>>>(
        (float*)context->input, (float*)context->output, n, (int)position[0]);
    cudaStreamSynchronize(context->stream);
}

void mulForwardInit(NnCudaOpContext *context) {
    context->hasInputContinuousMemory = true;
    context->hasOutputContinuousMemory = true;
}

void mulForward(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCudaOpContext *context) {
    if (threadIndex != 0) return;
    NnMulOpCodeConfig *config = (NnMulOpCodeConfig*)context->opConfig;
    float *other = nullptr;
    NnSize2D otherSize;
    resolvePipeOrBuffer(&other, &otherSize, context->pipes, context->pipeConfigs,
                       context->buffers, context->bufferConfigs, config->otherBufferIndex);
    int n = context->inputSize.y * context->inputSize.x * batchSize;
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    mulKernel<<<blocks, threadsPerBlock, 0, context->stream>>>(
        (float*)context->input, (float*)context->output, other, n);
    cudaStreamSynchronize(context->stream);
}

void ropeLlamaForwardInit(NnCudaOpContext *context) {
    context->hasInputContinuousMemory = true;
    context->hasOutputContinuousMemory = true;
}

void ropeLlamaForward(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCudaOpContext *context) {
    if (threadIndex != 0) return;
    NnRopeLlamaOpConfig *config = (NnRopeLlamaOpConfig*)context->opConfig;
    float *ropeCache = nullptr;
    NnSize2D cacheSize;
    resolvePipeOrBuffer(&ropeCache, &cacheSize, context->pipes, context->pipeConfigs,
                       context->buffers, context->bufferConfigs, config->ropeCacheBufferIndex);
    float *position = nullptr;
    NnSize2D posSize;
    resolvePipeOrBuffer(&position, &posSize, context->pipes, context->pipeConfigs,
                       context->buffers, context->bufferConfigs, config->positionPipeIndex);
    int dim = config->slice.sliceDim;
    int seqLen = config->slice.seqLen;
    int n = batchSize * dim / 2;
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    ropeLlamaKernel<<<blocks, threadsPerBlock, 0, context->stream>>>(
        (float*)context->output, ropeCache, seqLen, dim, batchSize, (int)position[0]);
    cudaStreamSynchronize(context->stream);
}

void multiHeadAttForwardInit(NnCudaOpContext *context) {
    context->hasInputContinuousMemory = true;
    context->hasOutputContinuousMemory = true;
}

void multiHeadAttForward(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCudaOpContext *context) {
    if (threadIndex != 0) return;
    NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig*)context->opConfig;
    float *q = nullptr, *k = nullptr, *v = nullptr, *att = nullptr;
    NnSize2D qSize, kSize, vSize, attSize;
    resolvePipeOrBuffer(&q, &qSize, context->pipes, context->pipeConfigs,
                       context->buffers, context->bufferConfigs, config->qBufferIndex);
    resolvePipeOrBuffer(&k, &kSize, context->pipes, context->pipeConfigs,
                       context->buffers, context->bufferConfigs, config->kBufferIndex);
    resolvePipeOrBuffer(&v, &vSize, context->pipes, context->pipeConfigs,
                       context->buffers, context->bufferConfigs, config->vBufferIndex);
    resolvePipeOrBuffer(&att, &attSize, context->pipes, context->pipeConfigs,
                       context->buffers, context->bufferConfigs, config->attBufferIndex);
    float *position = nullptr;
    NnSize2D posSize;
    resolvePipeOrBuffer(&position, &posSize, context->pipes, context->pipeConfigs,
                       context->buffers, context->bufferConfigs, config->positionPipeIndex);
    int n = batchSize * config->nHeads0 * config->seqLen;
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    multiHeadAttKernel<<<blocks, threadsPerBlock, 0, context->stream>>>(
        q, k, v, att, (float*)context->output, config->nHeads0, config->headSize, config->seqLen, batchSize);
    cudaStreamSynchronize(context->stream);
}

// Operation dispatch
NnCudaOpForwardInit getCudaOpForwardInit(NnOpCode code, NnOpQuantType quantType) {
    if (quantType != Q_NONE) {
        throw std::runtime_error("Quantization not supported in CUDA");
    }
    switch (code) {
        case OP_MATMUL: return matmulForwardInit;
        case OP_SILU: return siluForwardInit;
        case OP_INV_RMS: return invRmsForwardInit;
        case OP_RMS_NORM: return rmsNormForwardInit;
        case OP_EMBEDDING: return embeddingForwardInit;
        case OP_MERGE_ADD: return mergeAddForwardInit;
        case OP_CAST: return castForwardInit;
        case OP_SHIFT: return shiftForwardInit;
        case OP_MUL: return mulForwardInit;
        case OP_ROPE_LLAMA: return ropeLlamaForwardInit;
        case OP_MULTIHEAD_ATT: return multiHeadAttForwardInit;
        default:
            throw std::runtime_error("Unsupported CUDA operation: " + std::string(opCodeToString(code)));
    }
}

NnCudaOpForward getCudaOpForward(NnOpCode code, NnOpQuantType quantType) {
    if (quantType != Q_NONE) {
        throw std::runtime_error("Quantization not supported in CUDA");
    }
    switch (code) {
        case OP_MATMUL: return matmulForward;
        case OP_SILU: return siluForward;
        case OP_INV_RMS: return invRmsForward;
        case OP_RMS_NORM: return rmsNormForward;
        case OP_EMBEDDING: return embeddingForward;
        case OP_MERGE_ADD: return mergeAddForward;
        case OP_CAST: return castForward;
        case OP_SHIFT: return shiftForward;
        case OP_MUL: return mulForward;
        case OP_ROPE_LLAMA: return ropeLlamaForward;
        case OP_MULTIHEAD_ATT: return multiHeadAttForward;
        default:
            throw std::runtime_error("Unsupported CUDA operation: " + std::string(opCodeToString(code)));
    }
}

static void resolvePipeOrBuffer(float **pntr, NnSize2D *pntrSize, NnByte **pipes, NnPipeConfig *pipeConfigs,
                               NnByte **buffers, NnBufferConfig *bufferConfigs, NnUint index) {
    if (index < 1000) { // Arbitrary threshold to distinguish pipes from buffers
        *pntr = (float*)pipes[index];
        *pntrSize = pipeConfigs[index].size;
    } else {
        *pntr = (float*)buffers[index - 1000];
        *pntrSize = bufferConfigs[index - 1000].size;
    }
}

// NnCudaDevice implementation
NnCudaDevice::NnCudaDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) {
    this->gpuIndex = gpuIndex;
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
    this->netExecution = netExecution;
    cudaError_t err = cudaSetDevice(gpuIndex);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device " + std::to_string(gpuIndex) + ": " + cudaGetErrorString(err));
    }
    nBuffers = netConfig->nPipes + nodeConfig->nBuffers;
    buffers = new NnByte*[nBuffers];
    for (NnUint i = 0; i < nBuffers; ++i) {
        NnSize2D size = i < netConfig->nPipes ? netConfig->pipes[i].size : nodeConfig->buffers[i - netConfig->nPipes].size;
        cudaMalloc(&buffers[i], size.nBytes * netConfig->nBatches);
    }
}

NnCudaDevice::~NnCudaDevice() {
    cudaSetDevice(gpuIndex);
    for (NnUint i = 0; i < nBuffers; ++i) {
        cudaFree(buffers[i]);
    }
    delete[] buffers;
    cudaDeviceReset();
}

void NnCudaDevice::resolvePointer(NnByte **pntr, NnSize2D *pntrSize, NnPointerConfig *pointerConfig, NnUint batchIndex) {
    if (pointerConfig->source == SRC_PIPE) {
        *pntr = buffers[pointerConfig->pointerIndex];
        *pntrSize = netConfig->pipes[pointerConfig->pointerIndex].size;
    } else if (pointerConfig->source == SRC_BUFFER) {
        *pntr = buffers[netConfig->nPipes + pointerConfig->pointerIndex];
        *pntrSize = nodeConfig->buffers[pointerConfig->pointerIndex].size;
    } else {
        throw std::runtime_error("Invalid pointer source");
    }
    if (pointerConfig->type == PNTR_BATCH) {
        *pntr += batchIndex * pntrSize->nBytes;
    } else if (pointerConfig->type == PNTR_BATCHED_SLICE) {
        *pntr += (batchIndex * pntrSize->nBytes / netConfig->nNodes) * nodeConfig->nodeIndex;
        pntrSize->x /= netConfig->nNodes;
        pntrSize->length = pntrSize->y * pntrSize->x;
        pntrSize->nBytes = getBytes(pntrSize->floatType, pntrSize->length);
    } else if (pointerConfig->type == PNTR_RAW) {
        // No offset for raw pointers
    }
}

NnDeviceSegment *NnCudaDevice::createSegment(NnUint segmentIndex) {
    return new NnCudaDeviceSegment(this, segmentIndex, netConfig, nodeConfig, netExecution);
}

// NnCudaDeviceSegment implementation
NnCudaDeviceSegment::NnCudaDeviceSegment(NnDevice *device, NnUint segmentIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) {
    this->device = device;
    this->segmentConfig = &nodeConfig->segments[segmentIndex];
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
    this->netExecution = netExecution;
    cudaSetDevice(((NnCudaDevice*)device)->gpuIndex);
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(err)));
    }
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle");
    }
    cublasSetStream(cublasHandle, stream);
    nOps = segmentConfig->nOps;
    deviceBuffers = new NnByte*[nOps];
    opForward = new NnCudaOpForward*[nOps];
    opContexts = new NnCudaOpContext*[nOps];
    for (NnUint i = 0; i < nOps; ++i) {
        deviceBuffers[i] = nullptr;
        NnOpConfig *opConfig = &segmentConfig->ops[i];
        opContexts[i] = new NnCudaOpContext();
        opContexts[i]->name = opConfig->name;
        opContexts[i]->nBatches = netConfig->nBatches;
        opContexts[i]->pipes = netExecution->pipes;
        opContexts[i]->pipeConfigs = netConfig->pipes;
        opContexts[i]->buffers = ((NnCudaDevice*)device)->buffers + netConfig->nPipes;
        opContexts[i]->bufferConfigs = nodeConfig->buffers;
        opContexts[i]->opConfig = opConfig->config;
        opContexts[i]->weightSize = opConfig->weightSize;
        opContexts[i]->stream = stream;
        opContexts[i]->cublasHandle = cublasHandle;
        NnCudaOpForwardInit init = getCudaOpForwardInit(opConfig->code, opConfig->quantType);
        init(opContexts[i]);
        opForward[i] = getCudaOpForward(opConfig->code, opConfig->quantType);
    }
}

NnCudaDeviceSegment::~NnCudaDeviceSegment() {
    cudaSetDevice(((NnCudaDevice*)device)->gpuIndex);
    for (NnUint i = 0; i < nOps; ++i) {
        cudaFree(deviceBuffers[i]);
        delete opContexts[i];
        delete opForward[i];
    }
    delete[] deviceBuffers;
    delete[] opForward;
    delete[] opContexts;
    cublasDestroy(cublasHandle);
    cudaStreamDestroy(stream);
}

void NnCudaDeviceSegment::loadWeight(NnUint opIndex, NnSize nBytes, NnByte *weight) {
    assert(opIndex < nOps);
    assert(segmentConfig->ops[opIndex].weightSize.nBytes == nBytes);
    if (!deviceBuffers[opIndex]) {
        cudaMalloc(&deviceBuffers[opIndex], nBytes);
    }
    std::string opName = segmentConfig->ops[opIndex].name;
    if (WeightCache::hasWeights(opName, opIndex)) {
        std::vector<NnByte> cachedWeights(nBytes);
        WeightCache::loadWeights(opName, opIndex, nBytes, cachedWeights.data());
        cudaMemcpyAsync(deviceBuffers[opIndex], cachedWeights.data(), nBytes, cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpyAsync(deviceBuffers[opIndex], weight, nBytes, cudaMemcpyHostToDevice, stream);
        WeightCache::saveWeights(opName, opIndex, nBytes, weight);
    }
    opContexts[opIndex]->weight = deviceBuffers[opIndex];
    cudaStreamSynchronize(stream);
}

void NnCudaDeviceSegment::execute(NnNetExecution *execution, NnUint batchIndex) {
    netExecution->batchSize = batchIndex + 1;
    for (NnUint opIndex = 0; opIndex < nOps; ++opIndex) {
        forward(opIndex, 1, 0, batchIndex + 1);
    }
}

void NnCudaDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) {
    NnCudaOpContext *context = opContexts[opIndex];
    NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
    ((NnCudaDevice*)device)->resolvePointer(&context->input, &context->inputSize, &opConfig->input, batchIndex);
    ((NnCudaDevice*)device)->resolvePointer(&context->output, &context->outputSize, &opConfig->output, batchIndex);
    opForward[opIndex](nThreads, threadIndex, batchSize, context);
}
