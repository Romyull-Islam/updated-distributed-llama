// ===================== FILE: src/inference_utils.hpp =====================
#ifndef INFERENCE_UTILS_HPP
#define INFERENCE_UTILS_HPP

#include "app.hpp"
#include "device_selector.hpp"
#include "llm.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-cpu.hpp"

Inference* create_inference_engine(AppCliArgs* args, const std::vector<DeviceInfo>& selectedDevices);
Inference* createLocalInference(const char* modelPath, int nThreads);
Inference* createHybridRootInference(const char* modelPath, const std::vector<std::string>& hosts, const std::vector<int>& ports, int nThreads);
NnDevice* createDevice(AppCliArgs* args, const NnNetConfig* netConfig, const NnNodeConfig* nodeConfig, NnNetExecution* execution);

#endif
