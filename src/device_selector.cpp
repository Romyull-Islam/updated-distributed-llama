#include "device_selector.hpp"
#include "llm.hpp"
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <string>
#include <sys/stat.h>

float getLocalMemoryGB() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    long memTotalKB = 0;

    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal") != std::string::npos) {
            std::sscanf(line.c_str(), "MemTotal: %ld kB", &memTotalKB);
            break;
        }
    }
    return memTotalKB / 1024.0f / 1024.0f; // Convert to GB
}

float getRemoteMemoryGB(const std::string& ip) {
    std::string command = "ssh -o ConnectTimeout=2 " + ip + " cat /proc/meminfo | grep MemTotal";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return 0.0f;

    char buffer[256];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);

    long memTotalKB = 0;
    if (sscanf(result.c_str(), "MemTotal: %ld kB", &memTotalKB) == 1) {
        return memTotalKB / 1024.0f / 1024.0f; // Convert to GB
    } else {
        return 0.0f;
    }
}

std::vector<DeviceInfo> discover_devices(AppCliArgs* args) {
    std::vector<DeviceInfo> devices;

    float localMem = getLocalMemoryGB();
    devices.push_back(DeviceInfo{"root", localMem, ""});

    for (int i = 0; i < args->nWorkers; i++) {
        std::string name = "worker-" + std::to_string(i);
        float remoteMem = getRemoteMemoryGB(args->workerHosts[i]);
        devices.push_back(DeviceInfo{name, remoteMem, args->workerHosts[i]});
    }
    return devices;
}

std::vector<DeviceInfo> sort_devices_by_memory(const std::vector<DeviceInfo>& devices) {
    std::vector<DeviceInfo> sorted = devices;
    std::sort(sorted.begin(), sorted.end(), [](const DeviceInfo& a, const DeviceInfo& b) {
        return a.memoryGB > b.memoryGB;
    });
    return sorted;
}

std::vector<DeviceInfo> sort_devices_by_priority_list(const std::vector<DeviceInfo>& devices, const std::vector<std::string>& priorityList) {
    std::unordered_map<std::string, DeviceInfo> map;
    for (const auto& d : devices) map[d.name] = d;

    std::vector<DeviceInfo> sorted;
    for (const auto& name : priorityList) {
        if (map.count(name)) sorted.push_back(map[name]);
    }
    return sorted;
}

std::vector<DeviceInfo> select_devices_incrementally(const std::vector<DeviceInfo>& devices, double requiredMemoryGB) {
    std::vector<DeviceInfo> selected;
    double accumulated = 0.0;
    for (const auto& d : devices) {
        selected.push_back(d);
        accumulated += d.memoryGB;
        if (accumulated >= requiredMemoryGB) break;
    }
    return selected;
}

float estimate_required_memory(const char* modelPath) {
    try {
        LlmHeader header = loadLlmHeader(modelPath, 0, F_Q40);  // fallback to default float type

        // Fallback estimate using known structure without relying on nParams
        float estimatedParams = static_cast<float>(
            (header.dim * header.nLayers * 12) +
            (header.vocabSize * header.dim) +
            (header.hiddenDim * header.dim * 2)
        );

        float bytesPerParam = (header.weightType == F_Q80) ? 4.0f : 2.0f;
        float totalBytes = estimatedParams * bytesPerParam;
        float overheadFactor = 1.5f;
        return (totalBytes * overheadFactor) / (1024.0f * 1024.0f * 1024.0f);
    } catch (...) {
        struct stat st;
        if (stat(modelPath, &st) == 0) {
            float fallbackFactor = 2.0f;
            return (static_cast<float>(st.st_size) * fallbackFactor) / (1024.0f * 1024.0f * 1024.0f);
        }
        return 1.0f;  // Fallback to 1 GB if file check fails
    }
}
