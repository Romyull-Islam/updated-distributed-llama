// ===================== FILE: src/device_selector.hpp =====================
#pragma once
#include <string>
#include <vector>
#include "app.hpp"



struct DeviceInfo {
    std::string name;
    float memory; // in GB
    std::string ip;
};

// Detect total memory of the root node
float getLocalMemoryGB();

// Detect memory on remote worker using SSH
float getRemoteMemoryGB(const std::string& ip);

// Discover all devices (root + remote)
std::vector<DeviceInfo> discover_devices(AppCliArgs* args);

// Sort devices by largest memory first
std::vector<DeviceInfo> sort_devices_by_memory(const std::vector<DeviceInfo>& devices);

// Sort devices by user-specified priority list
std::vector<DeviceInfo> sort_devices_by_priority_list(const std::vector<DeviceInfo>& devices, const std::vector<std::string>& priority);

// Select top-N devices until cumulative memory meets requirement
std::vector<DeviceInfo> select_devices_incrementally(const std::vector<DeviceInfo>& devices, float required_memory);

// Estimate model size from header
float estimate_required_memory(const char* modelPath);
