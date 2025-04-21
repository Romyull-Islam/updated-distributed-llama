#ifndef DEVICE_SELECTOR_HPP
#define DEVICE_SELECTOR_HPP

#include <string>
#include <vector>
#include "llm.hpp"  // ✅ for LlmHeader and loadLlmHeader
#include "app.hpp"  // ✅ Needed for AppCliArgs


// Forward declare AppCliArgs
class AppCliArgs;

// Define DeviceInfo
struct DeviceInfo {
    std::string name;
    double memoryGB;
    std::string host;
};

// Declarations
std::vector<DeviceInfo> discover_devices(AppCliArgs* args);
std::vector<DeviceInfo> sort_devices_by_memory(const std::vector<DeviceInfo>& devices);
std::vector<DeviceInfo> sort_devices_by_priority_list(const std::vector<DeviceInfo>& devices, const std::vector<std::string>& priority);
std::vector<DeviceInfo> select_devices_incrementally(const std::vector<DeviceInfo>& devices, float required_memory);
float estimate_required_memory(const char* modelPath);

#endif // DEVICE_SELECTOR_HPP
