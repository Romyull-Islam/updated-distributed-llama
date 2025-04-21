#ifndef DEVICE_SELECTOR_HPP
#define DEVICE_SELECTOR_HPP

#include <vector>
#include <string>

class AppCliArgs; // ✅ Forward declaration

struct DeviceInfo {
    std::string host;
    double memoryGB;
};

// Your functions
std::vector<DeviceInfo> discover_devices(AppCliArgs* args);
std::vector<DeviceInfo> sort_devices_by_memory(const std::vector<DeviceInfo>& devices);
std::vector<DeviceInfo> sort_devices_by_priority_list(const std::vector<DeviceInfo>& devices, const std::vector<std::string>& priorityList);
std::vector<DeviceInfo> select_devices_incrementally(const std::vector<DeviceInfo>& sortedDevices, double requiredGB);
double estimate_required_memory(const char* modelPath);

#endif // DEVICE_SELECTOR_HPP
