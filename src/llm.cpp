#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "mmap.hpp"
#include "llm.hpp"
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Track device assignments globally
static std::vector<std::string> g_usedDevices;

json getGlobalDeviceSummaryJson() {
    return json{{"used_devices", g_usedDevices}};
}

static NnUint trySliceWithFallback(NnFloatType type, NnUint maxDevices, NnUint d0, NnUint d1, const std::string &label, NnMatmulSlice &outSlice) {
    for (NnUint n = 1; n <= maxDevices; ++n) {
        try {
            outSlice = sliceRowMatmul(type, n, d0, d1);
            return n;
        } catch (...) {
            continue;
        }
    }
    throw std::runtime_error("Failed to slice " + label + " with available devices");
}

LlmNet buildLlmNet(LlmHeader *h, NnUint availableNodes, NnUint nBatches) {
    LlmNet net = {};
    net.header = h;

    g_usedDevices.clear();

    NnUint maxDevices = availableNodes;
    NnUint usedDevices = 1;

    // Explicit layer priority based on computational complexity
    std::vector<std::pair<std::string, NnMatmulSlice*>> layers = {
        {"w1Slice", &net.w1Slice},     // largest: FFN first projection
        {"qSlice", &net.qSlice},       // attention query
        {"wclsSlice", &net.wclsSlice}, // final projection
        {"woSlice", &net.woSlice},     // attention output projection
        {"w3Slice", &net.w3Slice},
        {"w2Slice", &net.w2Slice},
        {"vSlice", &net.vSlice},
        {"kSlice", &net.kSlice}
    };

    usedDevices = 1;
    for (auto &[label, slice] : layers) {
        NnUint d0 = (label == "w2Slice" ? h->hiddenDim : h->dim);
        NnUint d1 = (label == "w2Slice" ? h->dim :
                     label == "kSlice" || label == "vSlice" ? h->kvDim :
                     label == "wclsSlice" ? h->vocabSize : h->hiddenDim);
        NnUint devicesForThis = trySliceWithFallback(h->weightType, maxDevices, d0, d1, label, *slice);
        usedDevices = std::max(usedDevices, devicesForThis);
    }

    for (NnUint i = 0; i < usedDevices; ++i) {
        g_usedDevices.push_back("device_" + std::to_string(i));
    }

    net.tokenEmbeddingSize = size2D(F_32, h->vocabSize, h->dim);
    net.rmsNormSize = size1D(F_32, h->dim);

    NnKvCacheSlice kv = sliceKvCache(h->kvDim, h->seqLen, usedDevices);
    NnMultiHeadAttSlice att = sliceMultiHeadAtt(h->nHeads, h->seqLen, usedDevices, nBatches);

    NnNetConfigBuilder config(usedDevices, nBatches);
    net.positionPipeIndex = config.addPipe("POS", size2D(F_32, nBatches, 1));
    net.tokenPipeIndex = config.addPipe("TOK", size2D(F_32, nBatches, 1));
    net.xPipeIndex = config.addPipe("X", size2D(F_32, nBatches, h->dim));
    net.logitsPipeIndex = config.addPipe("LG", size2D(F_32, nBatches, h->vocabSize));
    config.addPreSync(net.positionPipeIndex);

    net.netConfig = config.build();
    net.nodeConfigs = new NnNodeConfig[usedDevices];
    for (NnUint i = 0; i < usedDevices; ++i) {
        net.nodeConfigs[i] = buildNodeConfig(i);
    }

    return net;
}
