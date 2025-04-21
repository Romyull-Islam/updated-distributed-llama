// ===================== FILE: src/app.cpp =====================
// Usage Notes:
// --prioritize-by-memory      → sort devices by available memory
// --priority node1,node2,... → explicit node order
// If neither is given, fallback order is based on --workers list as: node1, node2, ...

#include "app.hpp"
#include "device_selector.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <memory>
#include "llm.hpp"
#include "tokenizer.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"

#if defined(DLLAMA_VULKAN)
#include "nn/nn-vulkan.hpp"
#endif

// ===================== AppCliArgs Parsing and Hybrid Inference =====================



static NnFloatType parseFloatType(char *val) {
    if (std::strcmp(val, "f32") == 0) return F_32;
    if (std::strcmp(val, "f16") == 0) return F_16;
    if (std::strcmp(val, "q40") == 0) return F_Q40;
    if (std::strcmp(val, "q80") == 0) return F_Q80;
    throw std::runtime_error("Invalid float type: " + std::string(val));
}

static ChatTemplateType parseChatTemplateType(char *val) {
    if (std::strcmp(val, "llama2") == 0) return TEMPLATE_LLAMA2;
    if (std::strcmp(val, "llama3") == 0) return TEMPLATE_LLAMA3;
    if (std::strcmp(val, "deepSeek3") == 0) return TEMPLATE_DEEP_SEEK3;
    throw std::runtime_error("Invalid chat template type: " + std::string(val));
}

AppCliArgs AppCliArgs::parse(int argc, char* *argv, bool requireMode) {
    AppCliArgs args;
    args.help = false;
    args.mode = nullptr;
    args.nBatches = 32;
    args.nThreads = 1;
    args.modelPath = nullptr;
    args.tokenizerPath = nullptr;
    args.prompt = nullptr;
    args.syncType = F_32;
    args.nWorkers = 0;
    args.workerHosts = nullptr;
    args.workerPorts = nullptr;
    args.port = 9990;
    args.temperature = 0.8f;
    args.topp = 0.9f;
    args.steps = 0;
    args.seed = (unsigned long long)time(nullptr);
    args.chatTemplateType = TEMPLATE_UNKNOWN;
    args.maxSeqLen = 0;
    args.netTurbo = true;
    args.gpuIndex = -1;
    int i = 1;
    if (requireMode && argc > 1) {
        args.mode = argv[1];
        i++;
    }
    for (int x = 0; x < argc; x++) {
        if ((std::strcmp(argv[x], "--usage") == 0) ||
            (std::strcmp(argv[x], "--help") == 0) ||
            (std::strcmp(argv[x], "-h") == 0)) {
            args.help = true;
            return args;
        }
    }
    for (; i + 1 < argc; i += 2) {
        char *name = argv[i];
        char *value = argv[i + 1];
        if (std::strcmp(name, "--model") == 0) {
            args.modelPath = value;
        } else if (std::strcmp(name, "--tokenizer") == 0) {
            args.tokenizerPath = value;
        } else if (std::strcmp(name, "--prompt") == 0) {
            args.prompt = value;
        } else if (std::strcmp(name, "--buffer-float-type") == 0) {
            args.syncType = parseFloatType(value);
        } else if (std::strcmp(name, "--workers") == 0) {
            int j = i + 1;
            for (; j < argc && argv[j][0] != '-'; j++);
            int count = j - i - 1;

            args.nWorkers = count;
            args.workerHosts = new char*[count];
            args.workerPorts = new NnUint[count];

            for (int s = 0; s < count; s++) {
                char *v = argv[i + 1 + s];
                char *sep = std::strstr(v, ":");
                if (sep == NULL) {
                    throw std::runtime_error("Invalid worker address: " + std::string(v));
                }
                int hostLen = sep - v;
                args.workerHosts[s] = new char[hostLen + 1];
                std::memcpy(args->workerHosts[s], v, hostLen);
                args.workerHosts[s][hostLen] = '\0';
                args.workerPorts[s] = std::atoi(sep + 1);
            }

            i += count - 1;
        } else if (std::strcmp(name, "--port") == 0) {
            args.port = atoi(value);
        } else if (std::strcmp(name, "--nthreads") == 0) {
            args.nThreads = atoi(value);
        } else if (std::strcmp(name, "--steps") == 0) {
            args.steps = atoi(value);
        } else if (std::strcmp(name, "--temperature") == 0) {
            args.temperature = atof(value);
        } else if (std::strcmp(name, "--topp") == 0) {
            args.topp = atof(value);
        } else if (std::strcmp(name, "--seed") == 0) {
            args.seed = atoll(value);
        } else if (std::strcmp(name, "--chat-template") == 0) {
            args.chatTemplateType = parseChatTemplateType(value);
        } else if (std::strcmp(name, "--max-seq-len") == 0) {
            args.maxSeqLen = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--gpu-index") == 0) {
            args.gpuIndex = atoi(value);
        } else if (std::strcmp(name, "--net-turbo") == 0) {
            args.netTurbo = atoi(value) == 1;
        } else {
            throw std::runtime_error("Unknown option: " + std::string(name));
        }
    }
    return args;
}

AppCliArgs::~AppCliArgs() {
    if (workerHosts != nullptr) {
        for (NnUint i = 0; i < nWorkers; i++)
            delete[] workerHosts[i];
        delete[] workerHosts;
    }
    if (workerPorts != nullptr)
        delete[] workerPorts;
}

void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context)) {
    AppInferenceContext* context = new AppInferenceContext();

    context->args = args;
    context->tokenizer = loadTokenizer(args->tokenizerPath);
    context->sampler = new Sampler(args->temperature, args->topp, args->seed);
    context->header = loadLlmHeader(args->modelPath);

    std::vector<DeviceInfo> allDevices = discover_devices(args);

    std::vector<DeviceInfo> sortedDevices = args->prioritizeByMemory
        ? sort_devices_by_memory(allDevices)
        : sort_devices_by_priority_list(allDevices, args->priorityList);

    double requiredGB = estimate_required_memory(args->modelPath);
    std::vector<DeviceInfo> selectedDevices = select_devices_incrementally(sortedDevices, requiredGB);

    context->inference = create_inference_engine(args, selectedDevices);

    handler(context);

    delete context->sampler;
    delete context->inference;
    delete context->tokenizer;
    delete context->header;
    delete context;
}

void runWorkerApp(AppCliArgs *args) {
    while (true) {
        std::unique_ptr<NnNetwork> networkPtr = NnNetwork::serve(args->port);
        NnNetwork *network = networkPtr.get();

        NnWorkerConfigReader configReader(network);
        NnNetConfig netConfig = configReader.readNet();
        NnNodeConfig nodeConfig = configReader.readNode();
        std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
        std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

        printNodeRequiredMemory(&netConfig, &nodeConfig);

        NnNetExecution execution(args->nThreads, &netConfig);
        std::unique_ptr<NnDevice> device(createDevice(args, &netConfig, &nodeConfig, &execution));

        NnNetworkNodeSynchronizer synchronizer(network, &execution, &netConfig, &nodeConfig);
        NnExecutor executor(&netConfig, &nodeConfig, device.get(), &execution, &synchronizer, false);

        NnWorkerWeightReader weightReader(&executor, network);
        weightReader.read();

        WorkerLlmInference inference(&execution, network);
        bool isFirstAttempt = true;
        bool isTurboEnabled = false;
        clock_t startTime;
        while (true) {
            try {
                if (isFirstAttempt)
                    startTime = clock();

                if (!inference.tryReadControlPacket()) {
                    if (isTurboEnabled && !isFirstAttempt && clock() - startTime > CLOCKS_PER_SEC) {
                        network->setTurbo(false);
                        isTurboEnabled = false;
                        printf("🚁 Network is in blocking mode\n");
                    }
                    isFirstAttempt = false;
                    continue;
                }
                if (inference.isFinished)
                    break;

                if (args->netTurbo && !isTurboEnabled) {
                    network->setTurbo(true);
                    isTurboEnabled = true;
                    printf("🚁 Network is in non-blocking mode\n");
                }
                executor.forward();
                isFirstAttempt = true;
            } catch (const NnReadNetworkException &e) {
                printf("Read network exception: %s\n", e.message);
                break;
            } catch (const NnWriteNetworkException &e) {
                printf("Write network exception: %s\n", e.message);
                break;
            }
        }
    }
}
