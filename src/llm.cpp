#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "mmap.hpp"
#include "llm.hpp"
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Forward declare the Matmul slice struct
struct NnMatmulSlice;

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

static NnNodeConfig buildNodeConfig(NnUint deviceIndex, LlmHeader *h, LlmNet *net, NnUint totalDevices) {
    NnNodeConfigBuilder builder(deviceIndex);

    const NnUint dim = h->dim;
    const NnUint kvDim = h->kvDim;
    const NnUint vocab = h->vocabSize;

    const NnSize2D dimVec = size2D(F_32, 1, dim);
    const NnSize2D kvVec = size2D(F_32, 1, kvDim);
    const NnSize2D logitsVec = size2D(F_32, 1, vocab / totalDevices);

    const NnUint bufferX = builder.addBuffer("x", dimVec);
    const NnUint bufferY = builder.addBuffer("y", dimVec);
    const NnUint bufferLG = builder.addBuffer("logits", logitsVec);
    const NnUint invRms = builder.addBuffer("inv_rms", size2D(F_32, 1, 1));

    NnSegmentConfigBuilder segment;

    // Add real ops: cast → rms_norm → matmul → cast → matmul → cast
    segment.addOp(OP_INV_RMS, "inv_rms", 0,
        pointerBatchConfig(SRC_BUFFER, bufferX),
        pointerBatchConfig(SRC_BUFFER, invRms),
        size0(),
        NnInvRmsOpConfig{h->normEpsilon});

    segment.addOp(OP_RMS_NORM, "rms_norm", 0,
        pointerBatchConfig(SRC_BUFFER, bufferX),
        pointerBatchConfig(SRC_BUFFER, bufferY),
        size1D(F_32, dim),
        NnRmsNormOpConfig{invRms});

    segment.addOp(OP_MATMUL, "matmul_proj", 0,
        pointerBatchConfig(SRC_BUFFER, bufferY),
        pointerBatchConfig(SRC_BUFFER, bufferLG),
        size2D(h->weightType, net->wclsSlice.n, net->wclsSlice.d0),
        NnMatmulOpConfig{});

    segment.addSync(net->logitsPipeIndex, SYNC_NODE_SLICES_EXCEPT_ROOT);

    builder.addSegment(segment.build());
    return builder.build();
}

LlmNet buildLlmNet(LlmHeader *h, NnUint availableNodes, NnUint nBatches) {
    LlmNet net = {};
    net.header = h;

    g_usedDevices.clear();

    NnUint usedDevices = 1;
    const NnFloatType type = h->weightType;

    std::vector<std::pair<std::string, NnMatmulSlice*>> layers = {
        {"w1Slice", &net.w1Slice},
        {"qSlice", &net.qSlice},
        {"wclsSlice", &net.wclsSlice},
        {"woSlice", &net.woSlice},
        {"w3Slice", &net.w3Slice},
        {"w2Slice", &net.w2Slice},
        {"vSlice", &net.vSlice},
        {"kSlice", &net.kSlice}
    };

    for (auto &[label, slice] : layers) {
        NnUint d0 = (label == "w2Slice" ? h->hiddenDim : h->dim);
        NnUint d1 = (label == "w2Slice" ? h->dim :
                     label == "kSlice" || label == "vSlice" ? h->kvDim :
                     label == "wclsSlice" ? h->vocabSize : h->hiddenDim);
        NnUint n = trySliceWithFallback(type, availableNodes, d0, d1, label, *slice);
        usedDevices = std::max(usedDevices, n);
    }

    for (NnUint i = 0; i < usedDevices; ++i)
        g_usedDevices.push_back("device_" + std::to_string(i));

    net.tokenEmbeddingSize = size2D(F_32, h->vocabSize, h->dim);
    net.rmsNormSize = size1D(F_32, h->dim);

    sliceKvCache(h->kvDim, h->seqLen, usedDevices);  // memory-aware
    sliceMultiHeadAtt(h->nHeads, h->seqLen, usedDevices, nBatches);

    NnNetConfigBuilder config(usedDevices, nBatches);
    net.positionPipeIndex = config.addPipe("POS", size2D(F_32, nBatches, 1));
    net.tokenPipeIndex = config.addPipe("TOK", size2D(F_32, nBatches, 1));
    net.xPipeIndex = config.addPipe("X", size2D(F_32, nBatches, h->dim));
    net.logitsPipeIndex = config.addPipe("LG", size2D(F_32, nBatches, h->vocabSize));
    config.addPreSync(net.positionPipeIndex);

    net.netConfig = config.build();
    net.nodeConfigs = new NnNodeConfig[usedDevices];
    for (NnUint i = 0; i < usedDevices; ++i) {
        net.nodeConfigs[i] = buildNodeConfig(i, h, &net, usedDevices);
    }

    return net;
}
