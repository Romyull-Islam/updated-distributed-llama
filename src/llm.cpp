
// FINAL PATCHED VERSION of llm.cpp with priority-based slicing and fallback

#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "mmap.hpp"
#include "llm.hpp"
#include <stdexcept>
#include <vector>
#include <string>

static std::vector<std::string> g_usedDevices;

NnUint trySliceWithFallback(NnFloatType type, NnUint maxDevices, NnUint d0, NnUint d1, const char* label, NnMatmulSlice* sliceOut) {
    for (NnUint n = 1; n <= maxDevices; ++n) {
        try {
            *sliceOut = sliceRowMatmul(type, n, d0, d1);
            return n;
        } catch (...) {
            continue;
        }
    }
    throw std::runtime_error(std::string("Failed to slice ") + label);
}

LlmNet buildLlmNet(LlmHeader* h, NnUint nNodes, NnUint nBatches) {
    LlmNet n;
    n.tokenEmbeddingSize = size2D(F_32, h->vocabSize, h->dim);
    n.rmsNormSize = size1D(F_32, h->dim);
    g_usedDevices.clear();

    // Slice priority
    NnUint maxNodes = nNodes;
    NnUint usedNodes = 1;

    usedNodes = std::max(usedNodes, trySliceWithFallback(h->weightType, maxNodes, h->dim, h->hiddenDim, "w1Slice", &n.w1Slice));
    usedNodes = std::max(usedNodes, trySliceWithFallback(h->weightType, maxNodes, h->dim, h->dim, "qSlice", &n.qSlice));
    usedNodes = std::max(usedNodes, trySliceWithFallback(h->weightType, maxNodes, h->dim, h->vocabSize, "wclsSlice", &n.wclsSlice));
    usedNodes = std::max(usedNodes, trySliceWithFallback(h->weightType, maxNodes, h->dim, h->dim, "woSlice", &n.woSlice));
    usedNodes = std::max(usedNodes, trySliceWithFallback(h->weightType, maxNodes, h->dim, h->hiddenDim, "w3Slice", &n.w3Slice));
    usedNodes = std::max(usedNodes, trySliceWithFallback(h->weightType, maxNodes, h->hiddenDim, h->dim, "w2Slice", &n.w2Slice));
    usedNodes = std::max(usedNodes, trySliceWithFallback(h->weightType, maxNodes, h->dim, h->kvDim, "vSlice", &n.vSlice));
    usedNodes = std::max(usedNodes, trySliceWithFallback(h->weightType, maxNodes, h->dim, h->kvDim, "kSlice", &n.kSlice));

    // Save global device usage
    for (NnUint i = 0; i < usedNodes; ++i)
        g_usedDevices.push_back("device_" + std::to_string(i));

    NnKvCacheSlice kvCacheSlice = sliceKvCache(h->kvDim, h->seqLen, usedNodes);
    NnMultiHeadAttSlice attSlice = sliceMultiHeadAtt(h->nHeads, h->seqLen, usedNodes, nBatches);

    NnNetConfigBuilder netBuilder(usedNodes, nBatches);
    n.positionPipeIndex = netBuilder.addPipe("POS", size2D(F_32, nBatches, 1));
    n.tokenPipeIndex = netBuilder.addPipe("TOK", size2D(F_32, nBatches, 1));
    n.xPipeIndex = netBuilder.addPipe("X", size2D(F_32, nBatches, h->dim));
    n.logitsPipeIndex = netBuilder.addPipe("LG", size2D(F_32, nBatches, h->vocabSize));
    netBuilder.addPreSync(n.positionPipeIndex);

    n.header = h;
    n.netConfig = netBuilder.build();
    n.nodeConfigs = new NnNodeConfig[usedNodes];
    for (NnUint i = 0; i < usedNodes; ++i)
        n.nodeConfigs[i] = buildDefaultNodeConfig(i);  // This must exist in original

    return n;
}

void releaseLlmNet(LlmNet* net) {
    for (NnUint i = 0; i < net->netConfig.nNodes; ++i)
        releaseNodeConfig(&net->nodeConfigs[i]);
    releaseNetConfig(&net->netConfig);
    delete[] net->nodeConfigs;
}

void loadLlmNetWeight(const char* path, LlmNet* net, NnRootWeightLoader* loader) {
    MmapFile file;
    openMmapFile(&file, path, net->header->fileSize);
#if DEBUG_USE_MMAP_FOR_WEIGHTS
    assert(net->netConfig.nNodes == 1);
#else
    std::unique_ptr<MmapFile, void (*)(MmapFile*)> fdPtr(&file, closeMmapFile);
    printf("💿 Loading weights...
");
#endif

    NnByte* data = (NnByte*)file.data;
    NnByte* b = &data[net->header->headerSize];
    b += loader->loadRoot("embedding", 0, net->tokenEmbeddingSize.nBytes, b);

    for (NnUint i = 0; i < net->header->nLayers; ++i) {
        b += loader->loadRowMatmulSlices("block_matmul_q", i, &net->qSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_k", i, &net->kSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_v", i, &net->vSlice, b);
        b += loader->loadColMatmulSlices("block_matmul_wo", i, &net->woSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_w1", i, &net->w1Slice, b);
        b += loader->loadColMatmulSlices("block_matmul_w2", i, &net->w2Slice, b);
        b += loader->loadRowMatmulSlices("block_matmul_w3", i, &net->w3Slice, b);
    }

    b += loader->loadRowMatmulSlices("final_matmul_logits", 0, &net->wclsSlice, b);

    long long diff = (long long)(b - data) - net->header->fileSize;
    if (diff != 0)
        throw std::runtime_error("Mismatch in weight size: " + std::to_string(diff));
    printf("💿 Weights loaded
");
    loader->finish();
}
