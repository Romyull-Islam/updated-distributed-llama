#ifndef APP_HPP
#define APP_HPP

#include <chrono>
#include <vector>
#include <string>
#include "nn/nn-core.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "llm.hpp"             // for LlmHeader, loadLlmHeader, etc.
#include "tokenizer.hpp"       // for Tokenizer and Sampler
#include "device_selector.hpp" // for DeviceInfo, discover/sort/select
 // for create_inference_engine

class AppCliArgs {
public:
    char *mode;
    NnUint nThreads;
    NnUint nBatches;
    bool help;

    // inference
    char *modelPath;
    char *tokenizerPath;
    char *prompt;
    NnFloatType syncType;
    NnUint nWorkers;
    char **workerHosts;
    NnUint *workerPorts;
    float temperature;
    float topp;
    NnUint steps;
    bool benchmark;
    unsigned long long seed;
    ChatTemplateType chatTemplateType;
    NnUint maxSeqLen;
    bool netTurbo;
    int gpuIndex;

    // worker
    NnUint port;

    // new hybrid inference flags
    bool prioritizeByMemory = false;
    std::vector<std::string> priorityList;

    static AppCliArgs parse(int argc, char **argv, bool hasMode);
    ~AppCliArgs();
};

typedef struct {
    NnUint position;
    NnUint batchSize; // 0 = stop signal
} LlmControlPacket;

class RootLlmInference {
public:
    float *logitsPipe;
private:
    float *tokenPipe;
    float *positionPipe;
    LlmHeader *header;
    NnDevice *device;
    NnNetExecution *execution;
    NnExecutor *executor;
    NnNetwork *network;
    LlmControlPacket controlPacket;
public:
    RootLlmInference(LlmNet *net, NnDevice *device, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network);
    void setBatchSize(NnUint batchSize);
    void setPosition(NnUint position);
    void setToken(NnUint batchIndex, NnUint token);
    void forward();
    void finish();
};

class WorkerLlmInference {
public:
    bool isFinished;
private:
    float *positionPipe;
    NnNetExecution *execution;
    NnNetwork *network;
    LlmControlPacket controlPacket;
public:
    WorkerLlmInference(NnNetExecution *execution, NnNetwork *network);
    bool tryReadControlPacket();
};

typedef struct {
    AppCliArgs *args;
    LlmHeader *header;
    Inference *inference;  // ✅ now accepts both RootLlmInference and future types
    Tokenizer *tokenizer;
    Sampler *sampler;
    NnNetwork *network;
    NnExecutor *executor;
} AppInferenceContext;


void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context));
void runWorkerApp(AppCliArgs *args);

#endif // APP_HPP
