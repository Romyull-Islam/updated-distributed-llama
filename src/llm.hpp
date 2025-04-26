#ifndef LLM_HPP
#define LLM_HPP

#include "nn/nn-core.hpp"
#include "nn/nn-executor.hpp"
#include "nn/nn-network.hpp"
#include <string>
#include <vector>
#include <map>
#include <mutex>

enum LlmHeaderKey {
    VERSION = 0,
    ARCH_TYPE = 1,
    DIM = 2,
    HIDDEN_DIM = 3,
    N_LAYERS = 4,
    N_HEADS = 5,
    N_KV_HEADS = 6,
    N_EXPERTS = 7,
    N_ACTIVE_EXPERTS = 8,
    VOCAB_SIZE = 9,
    SEQ_LEN = 10,
    HIDDEN_ACT = 11,
    ROPE_THETA = 12,
    WEIGHT_FLOAT_TYPE = 13,
    ROPE_SCALING_FACTOR = 14,
    ROPE_SCALING_LOW_FREQ_FACTOR = 15,
    ROPE_SCALING_HIGH_FREQ_FACTOR = 16,
    ROPE_SCALING_ORIG_MAX_SEQ_LEN = 17,
    ROPE_TYPE = 18,
};

enum LlmHiddenAct {
    HIDDEN_ACT_GELU,
    HIDDEN_ACT_SILU,
};

enum LlmArchType {
    LLAMA = 0xABCD00,
};

typedef struct {
    NnSize headerSize;
    NnSize fileSize;
    int version;
    LlmArchType archType;
    NnUint dim;
    NnUint nLayers;
    NnUint nHeads;
    NnUint headSize;
    NnUint nKvHeads;
    NnUint nExperts;
    NnUint nActiveExperts;
    NnUint origSeqLen;
    NnUint seqLen;
    NnUint hiddenDim;
    LlmHiddenAct hiddenAct;
    NnUint kvDim;
    NnUint vocabSize;
    float ropeTheta;
    NnRopeType ropeType;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactor;
    NnUint ropeScalingOrigMaxSeqLen;
    float normEpsilon;
    NnFloatType weightType;
    NnFloatType syncType;
} LlmHeader;

typedef struct {
    LlmHeader *header;
    NnNetConfig netConfig;
    NnNodeConfig *nodeConfigs;
    NnRowMatmulSlice qSlice;
    NnRowMatmulSlice kSlice;
    NnRowMatmulSlice vSlice;
    NnColMatmulSlice woSlice;
    NnRowMatmulSlice w1Slice;
    NnColMatmulSlice w2Slice;
    NnRowMatmulSlice w3Slice;
    NnRowMatmulSlice wclsSlice;
    NnUint positionPipeIndex;
    NnUint tokenPipeIndex;
    NnUint xPipeIndex;
    NnUint logitsPipeIndex;
    NnSize2D tokenEmbeddingSize;
    NnSize2D rmsNormSize;
} LlmNet;

class WeightCache {
public:
    static bool hasWeights(const std::string& opName, NnUint opIndex);
    static void saveWeights(const std::string& opName, NnUint opIndex, NnSize nBytes, const NnByte* data);
    static void loadWeights(const std::string& opName, NnUint opIndex, NnSize nBytes, NnByte* data);
    static NnSize getWeightSize(const std::string& opName, NnUint opIndex);
private:
    static std::map<std::string, std::vector<NnByte>> cache;
    static std::map<std::string, NnSize> sizes;
    static std::mutex mutex;
};

LlmHeader loadLlmHeader(const char* path, const unsigned int maxSeqLen, NnFloatType syncType);
void printLlmHeader(LlmHeader *header);
LlmNet buildLlmNet(LlmHeader *h, NnUint nNodes, NnUint nBatches);
void releaseLlmNet(LlmNet *net);
void loadLlmNetWeight(const char* path, LlmNet *net, NnRootWeightLoader *loader);

#endif
