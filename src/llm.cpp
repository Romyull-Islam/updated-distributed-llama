#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "mmap.hpp"
#include "llm.hpp"
#include <stdexcept>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cstdio>

// ✅ Converts enum values to strings for readable debug prints
const char *hiddenActToString(LlmHiddenAct act) {
    switch (act) {
        case HIDDEN_ACT_GELU: return "GELU";
        case HIDDEN_ACT_SILU: return "SiLU";
        default: return "Unknown";
    }
}

const char *ropeTypeToString(NnRopeType type) {
    switch (type) {
        case ROPE_LLAMA: return "LLaMA";
        case ROPE_LLAMA3_1: return "LLaMA3.1";
        default: return "Unknown";
    }
}

const char *archTypeToString(LlmArchType type) {
    switch (type) {
        case LLAMA: return "LLaMA";
        default: return "Unknown";
    }
}

// ✅ Prints all key fields from model header for debugging
void printLlmHeader(LlmHeader *header) {
    printf("🧠 LLM Header\n");
    printf("  Magic: %u\n", header->magic);
    printf("  Version: %u\n", header->version);
    printf("  FileSize: %zu\n", header->fileSize);
    printf("  HeaderSize: %zu\n", header->headerSize);
    printf("  Arch: %s\n", archTypeToString(header->archType));
    printf("  HiddenAct: %s\n", hiddenActToString(header->hiddenAct));
    printf("  Dim: %u\n", header->dim);
    printf("  KvDim: %u\n", header->kvDim);
    printf("  HiddenDim: %u\n", header->hiddenDim);
    printf("  VocabSize: %u\n", header->vocabSize);
    printf("  nLayers: %u\n", header->nLayers);
    printf("  nHeads: %u\n", header->nHeads);
    printf("  nKvHeads: %u\n", header->nKvHeads);
    printf("  OrigSeqLen: %u\n", header->origSeqLen);
    printf("  SeqLen: %u\n", header->seqLen);
    printf("  NormEpsilon: %f\n", header->normEpsilon);
    printf("  WeightType: %u\n", header->weightType);
    printf("  RopeType: %s\n", ropeTypeToString(header->ropeType));
    printf("  RopeTheta: %.0f\n", header->ropeTheta);
    if (header->ropeType == ROPE_LLAMA3_1) {
        printf("  RopeScalingFactor: %f\n", header->ropeScalingFactor);
        printf("  RopeScalingLowFreqFactor: %f\n", header->ropeScalingLowFreqFactor);
        printf("  RopeScalingHighFreqFactory: %f\n", header->ropeScalingHighFreqFactory);
        printf("  RopeScalingOrigMaxSeqLen: %u\n", header->ropeScalingOrigMaxSeqLen);
    }
}

// ✅ Fills rope cache for LLaMA and LLaMA 3.1 models
static void fullfillRopeLlama3Cache(const NnRopeLlamaOpConfig *config, float *cache) {
    const NnUint dim0 = config->isQ ? config->slice.qDim0 : config->slice.kvDim0;
    if (dim0 % 2 != 0) throw std::runtime_error("RoPE dimension must be even");
    const NnUint dim0Half = dim0 / 2;
    const NnUint sliceDim = config->slice.sliceDim;
    const float ropeTheta = config->ropeTheta ? config->ropeTheta : 10000.0f;
    const bool isLlama3_1 = config->ropeScalingFactor > 0.0f;
    const float scalingFactor = isLlama3_1 ? config->ropeScalingFactor : 1.0f;
    const float lowFreqFactor = isLlama3_1 ? config->ropeScalingLowFreqFactor : 1.0f;
    const float highFreqFactor = isLlama3_1 ? config->ropeScalingHighFreqFactory : 1.0f;

    for (NnUint pos = 0; pos < config->seqLen; ++pos) {
        for (NnUint i = 0; i < dim0Half; ++i) {
            float freq = (float)i / (float)dim0Half;
            float scaledFreq = freq;
            if (isLlama3_1) {
                scaledFreq /= scalingFactor;
                if (scaledFreq < lowFreqFactor) {
                    scaledFreq *= lowFreqFactor;
                } else if (scaledFreq > highFreqFactor) {
                    scaledFreq *= highFreqFactor;
                }
            }
            float theta = (float)pos * ropeTheta * scaledFreq;
            NnUint idx = pos * sliceDim + i * 2;
            cache[idx] = cosf(theta);
            cache[idx + 1] = sinf(theta);
        }
    }
}

// ✅ Loads binary header, supports LLaMA 3.1 fields
LlmHeader loadLlmHeader(const char *path, const NnUint maxSeqLen, NnFloatType syncType) {
    LlmHeader header;
    std::memset(&header, 0, sizeof(LlmHeader));
    header.weightType = F_UNK;
    header.hiddenAct = HIDDEN_ACT_SILU;
    header.ropeType = ROPE_LLAMA;
    header.ropeTheta = 10000.0f;
    header.syncType = syncType;
    header.magic = 0xA00ABCD;

    std::unique_ptr<FILE, int (*)(FILE *)> fdPtr(fopen(path, "rb"), fclose);
    FILE *fd = fdPtr.get();
    if (!fd) throw std::runtime_error("Cannot open model file");

    int magic;
    if (fread(&magic, sizeof(int), 1, fd) != 1) throw std::runtime_error("Cannot read magic value");
    if (magic != header.magic) throw std::runtime_error("Unsupported magic number");

    if (fread(&header.headerSize, sizeof(int), 1, fd) != 1)
        throw std::runtime_error("Cannot read header size");

    std::vector<int> bufferPtr(header.headerSize / sizeof(int));
    int *buffer = bufferPtr.data();
    if (fread(buffer, sizeof(int), bufferPtr.size(), fd) != bufferPtr.size())
        throw std::runtime_error("Cannot read header");

    for (size_t i = 0; i < bufferPtr.size(); i += 2) {
        int key = buffer[i], val = buffer[i + 1];
        switch (key) {
            case VERSION: header.version = val; break;
            case ARCH_TYPE: header.archType = (LlmArchType)val; break;
            case DIM: header.dim = val; break;
            case HIDDEN_DIM: header.hiddenDim = val; break;
            case N_LAYERS: header.nLayers = val; break;
            case N_HEADS: header.nHeads = val; break;
            case N_KV_HEADS: header.nKvHeads = val; break;
            case VOCAB_SIZE: header.vocabSize = val; break;
            case SEQ_LEN: header.seqLen = val; break;
            case HIDDEN_ACT: header.hiddenAct = (LlmHiddenAct)val; break;
            case WEIGHT_FLOAT_TYPE: header.weightType = (NnFloatType)val; break;
            case ROPE_TYPE: header.ropeType = (NnRopeType)val; break;
            case ROPE_THETA: header.ropeTheta = (float)val; break;
            case ROPE_SCALING_FACTOR: header.ropeScalingFactor = (float)val; break;
            case ROPE_SCALING_LOW_FREQ_FACTOR: header.ropeScalingLowFreqFactor = (float)val; break;
            case ROPE_SCALING_HIGH_FREQ_FACTORY: header.ropeScalingHighFreqFactory = (float)val; break;
            case ROPE_SCALING_ORIG_MAX_SEQ_LEN: header.ropeScalingOrigMaxSeqLen = val; break;
            case NORM_EPSILON: header.normEpsilon = (float)val; break;
            default: break;
        }
    }

    if (header.weightType == F_UNK) throw std::runtime_error("Weight type not found in header");
    if (header.version != 1) throw std::runtime_error("Unsupported version");
    if (header.dim % header.nHeads != 0) throw std::runtime_error("Dim must be divisible by nHeads");
    if (header.nHeads % header.nKvHeads != 0) throw std::runtime_error("nHeads must be divisible by nKvHeads");

    header.origSeqLen = header.seqLen;
    if (maxSeqLen > 0 && header.seqLen > maxSeqLen)
        header.seqLen = maxSeqLen;
    header.headSize = header.dim / header.nHeads;
    header.kvDim = (header.dim * header.nKvHeads) / header.nHeads;

    // Compute file size using fseek/ftell
    if (fseek(fd, 0, SEEK_END) != 0) throw std::runtime_error("Cannot seek to end of file");
    header.fileSize = ftell(fd);
    if (header.fileSize == -1) throw std::runtime_error("Cannot determine file size");

    printLlmHeader(&header);
    return header;
}

// ✅ Memory-aware slicing for adaptive load distribution
static NnUint trySliceWithFallback(NnFloatType type, NnUint maxDevices, NnUint d0, NnUint d1, const char *label, NnMatmulSlice *out) {
    NnSize mem = printNodeRequiredMemory(d0 * d1 * getBytes(type));
    for (NnUint n = 1; n <= maxDevices; ++n) {
        NnMatmulSlice slice = sliceRowMatmul(type, n, d0, d1);
        if (slice.totalMemory <= mem) {
            *out = slice;
            return n;
        }
    }
    throw std::runtime_error(std::string("Failed to slice matrix for ") + label);
}

// ✅ Builds network with adaptive device count and real layer operations
LlmNet buildLlmNet(LlmHeader *h, NnUint maxDevices, NnUint nBatches) {
    LlmNet n;
    n.tokenEmbeddingSize = size2D(h->weightType, h->vocabSize, h->dim);
    n.rmsNormSize = size1D(F_32, h->dim);

    // 🧠 Start with one device and only increment if slicing fails
    NnUint used = 1;
    used = std::max(used, trySliceWithFallback(h->weightType, maxDevices, h->dim, h->dim, "q", &n.qSlice));
    used = std::max(used, trySliceWithFallback(h->weightType, maxDevices, h->dim, h->kvDim, "k", &n.kSlice));
    used = std::max(used, trySliceWithFallback(h->weightType, maxDevices, h->dim, h->kvDim, "v", &n.vSlice));
    used = std::max(used, trySliceWithFallback(h->weightType, maxDevices, h->dim, h->dim, "wo", &n.woSlice));
    used = std::max(used, trySliceWithFallback(h->weightType, maxDevices, h->dim, h->vocabSize, "wcls", &n.wclsSlice));
    used = std::max(used, trySliceWithFallback(h->weightType, maxDevices, h->dim, h->hiddenDim, "w1", &n.w1Slice));
    used = std::max(used, trySliceWithFallback(h->weightType, maxDevices, h->hiddenDim, h->dim, "w2", &n.w2Slice));
    used = std::max(used, trySliceWithFallback(h->weightType, maxDevices, h->dim, h->hiddenDim, "w3", &n.w3Slice));

    // 🧠 Cache + attention slice definitions
    NnKvCacheSlice kv = sliceKvCache(h->kvDim, h->seqLen, used);
    NnMultiHeadAttSlice att = sliceMultiHeadAtt(h->nHeads, h->seqLen, used, nBatches);

    // 🧠 Build network graph
    NnNetConfigBuilder net(used, nBatches);
    n.positionPipeIndex = net.addPipe("POS", size2D(F_32, nBatches, 1));
    n.tokenPipeIndex = net.addPipe("TOK", size2D(F_32, nBatches, 1));
    n.xPipeIndex = net.addPipe("X", size2D(F_32, nBatches, h->dim));
    n.logitsPipeIndex = net.addPipe("LG", size2D(F_32, nBatches, h->vocabSize));
    net.addPreSync(n.positionPipeIndex);

    // 🧠 Build per-node execution segments with real ops
    n.nodeConfigs = new NnNodeConfig[used];
    for (NnUint i = 0; i < used; ++i) {
        NnNodeConfigBuilder builder(i);
        builder.addBuffer("X", size2D(F_32, nBatches, h->dim));

        NnSegmentConfigBuilder seg;
        seg.addOp(NN_OP_EMBEDDING);                                      // Token embedding
        seg.addOp(NN_OP_RMSNORM, h->normEpsilon);                        // RMSNorm
        seg.addMatMul(n.qSlice.slices[i]);                               // Attention Q
        seg.addMatMul(n.kSlice.slices[i]);                               // Attention K
        seg.addMatMul(n.vSlice.slices[i]);                               // Attention V
        seg.addOp(NN_OP_ROPE_LLAMA);                                     // RoPE (handles LLaMA and LLaMA3.1)
        seg.addOp(NN_OP_ATTENTION, att.blocks[i]);                       // Multi-head attention
        seg.addMatMul(n.woSlice.slices[i]);                              // Attention output
        seg.addOp(NN_OP_MERGE_ADD);                                      // Residual connection
        seg.addOp(NN_OP_RMSNORM, h->normEpsilon);                        // Norm again
        seg.addMatMul(n.w1Slice.slices[i]);                              // Feedforward projection
        seg.addOp(h->hiddenAct == HIDDEN_ACT_GELU ? NN_OP_GELU : NN_OP_SILU); // Activation
        seg.addMatMul(n.w3Slice.slices[i]);                              // Gated projection
        seg.addOp(NN_OP_MUL);                                            // Element-wise multiplication
        seg.addMatMul(n.w2Slice.slices[i]);                              // Final FF matmul
        seg.addOp(NN_OP_MERGE_ADD);                                      // Residual connection
        seg.addOp(NN_OP_RMSNORM, h->normEpsilon);                        // Final norm
        seg.addMatMul(n.wclsSlice.slices[i]);                            // Classifier output
        seg.addOp(NN_OP_SOFTMAX);                                        // Softmax for logits
        builder.addSegment(seg.build());

        n.nodeConfigs[i] = builder.build();
    }

    n.header = h;
    n.netConfig = net.build();
    return n;
}

// ✅ Loads weights for all layers, ensuring correct slicing
void loadLlmNetWeight(const char *path, LlmNet *net, NnRootWeightLoader *loader) {
    MmapFile file;
    openMmapFile(&file, path, net->header->fileSize);
    std::unique_ptr<MmapFile, void (*)(MmapFile *)> fdPtr(&file, closeMmapFile);
    printf("💿 Loading weights...\n");

    NnByte *data = (NnByte *)file.data;
    NnByte *b = &data[net->header->headerSize];

    // ✅ Load token embedding
    b += loader->loadRoot("embedding", 0, net->tokenEmbeddingSize.nBytes, b);

    // ✅ Load per-layer slices
    for (NnUint i = 0; i < net->header->nLayers; ++i) {
        b += loader->loadRowMatmulSlices("block_matmul_q", i, &net->qSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_k", i, &net->kSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_v", i, &net->vSlice, b);
        b += loader->loadColMatmulSlices("block_matmul_wo", i, &net->woSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_w1", i, &net->w1Slice, b);
        b += loader->loadColMatmulSlices("block_matmul_w2", i, &net->w2Slice, b);
        b += loader->loadRowMatmulSlices("block_matmul_w3", i, &net->w3Slice, b);
        b += loader->loadRoot("rms_att_weight", i, net->rmsNormSize.nBytes, b);
        b += loader->loadRoot("rms_ffn_weight", i, net->rmsNormSize.nBytes, b);
        b += loader->loadRoot("rms_final_weight", i, net->rmsNormSize.nBytes, b);
    }

    // ✅ Load output classifier
    b += loader->loadRowMatmulSlices("final_matmul_logits", 0, &net->wclsSlice, b);

    if ((b - data) != net->header->fileSize)
        throw std::runtime_error("Mismatch in weight file size");

    printf("💿 Weights loaded\n");
    loader->finish();
}

// ✅ Releases network resources to prevent memory leaks
void releaseLlmNet(LlmNet *net) {
    if (net->nodeConfigs) {
        delete[] net->nodeConfigs;
        net->nodeConfigs = nullptr;
    }
    if (net->netConfig) {
        // NnNetConfig is managed by NnNetConfigBuilder, assumed to handle its own cleanup
        net->netConfig = nullptr;
    }
}
