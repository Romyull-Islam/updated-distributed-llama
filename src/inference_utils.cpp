// ===================== FILE: src/inference_utils.cpp =====================
#include "inference_utils.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-cpu.hpp"
#include "llm.hpp"
#include <cstring>

std::vector<char*> extract_hosts(const std::vector<DeviceInfo>& devices) {
    std::vector<char*> hosts;
    for (const auto& d : devices) {
        if (!d.host.empty()) {
            char* host = new char[d.host.size() + 1];
            std::strcpy(host, d.host.c_str());
            hosts.push_back(host);
        }
    }
    return hosts;
}

std::vector<NnUint> extract_ports(const std::vector<DeviceInfo>& devices) {
    std::vector<NnUint> ports;
    for (const auto& d : devices) {
        if (!d.host.empty()) {
            ports.push_back(9999); // Assumes default port
        }
    }
    return ports;
}

Inference* create_inference_engine(AppCliArgs* args, const std::vector<DeviceInfo>& selectedDevices) {
    if (selectedDevices.size() > 1) {
        std::vector<std::string> hosts;
        std::vector<int> ports;
        for (const auto& d : selectedDevices) {
            if (!d.host.empty()) {
                hosts.push_back(d.host);
                ports.push_back(9999);
            }
        }
        return createHybridRootInference(
            args->modelPath,
            hosts,
            ports,
            args->nThreads
        );
    } else {
        return createLocalInference(args->modelPath, args->nThreads);
    }
}

Inference* createLocalInference(const char* modelPath, int nThreads) {
    LlmHeader* header = new LlmHeader(loadLlmHeader(modelPath, 0, F_Q40));
    LlmNet* net = new LlmNet(buildLlmNet(header, 1, 32));
    NnNodeConfig* rootNodeConfig = &net->nodeConfigs[0];
    NnNetExecution* execution = new NnNetExecution(nThreads, &net->netConfig);
    NnDevice* device = new NnCpuDevice(&net->netConfig, rootNodeConfig, execution);
    NnFakeNodeSynchronizer* synchronizer = new NnFakeNodeSynchronizer();
    NnExecutor* executor = new NnExecutor(&net->netConfig, rootNodeConfig, device, execution, synchronizer, false);

    NnRootWeightLoader loader(executor, nullptr, 1);
    loadLlmNetWeight(modelPath, net, &loader);

    return new RootLlmInference(net, device, execution, executor, nullptr);
}

Inference* createHybridRootInference(
    const char* modelPath,
    const std::vector<std::string>& hosts,
    const std::vector<int>& ports,
    int nThreads
) {
    NnUint nNodes = hosts.size() + 1;

    LlmHeader* header = new LlmHeader(loadLlmHeader(modelPath, 0, F_Q40));
    LlmNet* net = new LlmNet(buildLlmNet(header, nNodes, 32));
    NnNodeConfig* rootNodeConfig = &net->nodeConfigs[0];
    NnNetExecution* execution = new NnNetExecution(nThreads, &net->netConfig);

    NnNetwork* network = NnNetwork::connect(nNodes - 1, hosts, ports).release();
    NnNodeSynchronizer* synchronizer = new NnNetworkNodeSynchronizer(network, execution, &net->netConfig, rootNodeConfig);
    NnRootConfigWriter(network).writeToWorkers(&net->netConfig, net->nodeConfigs);

    NnDevice* device = new NnCpuDevice(&net->netConfig, rootNodeConfig, execution);
    NnExecutor* executor = new NnExecutor(&net->netConfig, rootNodeConfig, device, execution, synchronizer, false);

    NnRootWeightLoader loader(executor, network, nNodes);
    loadLlmNetWeight(modelPath, net, &loader);

    return new RootLlmInference(net, device, execution, executor, network);
}
