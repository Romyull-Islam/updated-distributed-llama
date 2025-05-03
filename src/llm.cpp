#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "mmap.hpp"
#include "llm.hpp"
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

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

// Corrected inline implementation
static NnNodeConfig buildNodeConfig(NnUint deviceIndex) {
    NnNodeConfigBuilder builder(deviceIndex);

    NnSize2D dimVec = size2D(F_32, 1, 4096);
    NnSize2D logitsVec = size2D(F_32, 1, 32000);

    const NnUint xBuf = builder.addBuffer("X", dimVec);
    const NnUint posBuf = builder.addBuffer("POS", size2D(F_32, 1, 1));
    const NnUint tokBuf = builder.addBuffer("TOK", size2D(F_32, 1, 1));
    const NnUint lgBuf = builder.addBuffer("LG", logitsVec);

    NnSegmentConfigBuilder segment;

    // Token embedding
    NnPointerConfig inputTok = pointerConfig(SRC_BUFFER, tokBuf, 0, 0);
    NnPointerConfig outputTok = pointerConfig(SRC_BUFFER, xBuf, 0, 0);
    segment.addOp(OP_EMBEDDING, "tok_embed", 0, inputTok, outputTok, dimVec, NnEmbeddingOpConfig{});

    // RMSNorm
    segment.addOp(OP_RMS_NORM, "rms", 0, outputTok, outputTok, size2D(F_32, 1, 4096), NnRmsNormOpConfig{0});

    // MatMul
    segment.addOp(OP_MATMUL, "att_q", 0, outputTok, outputTok, dimVec, NnMatmulOpConfig{});

    segment.addSync(0, SYNC_NODE_SLICES);

    builder.addSegment(segment.build());
    return builder.build();
}

LlmNet buildLlmNet(LlmHeader *h, NnUint availableNodes, NnUint nBatches) {
    LlmNet net = {};
    net.header = h;

    g_usedDevices.clear();
    NnUint maxDevices = availableNodes;
    NnUint usedDevices = 1;

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
