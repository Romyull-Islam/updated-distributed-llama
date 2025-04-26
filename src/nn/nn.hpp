#ifndef NN_H
#define NN_H

#include "nn-core.hpp"

class NnExecutor; // Forward declaration
class NnNetExecution; // Forward declaration

class NnDeviceSegment {
public:
    NnSegmentConfig* segmentConfig;
    NnNetConfig* netConfig;
    NnNodeConfig* nodeConfig;
    NnNetExecution* netExecution;
    NnDeviceSegment() : segmentConfig(nullptr), netConfig(nullptr), nodeConfig(nullptr), netExecution(nullptr) {}
    virtual ~NnDeviceSegment() = default;
    virtual void loadWeight(NnUint opIndex, NnSize nBytes, NnByte* weight) = 0;
    virtual void execute(NnNetExecution* execution, NnUint batchIndex) = 0;
    virtual void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) {}
};

class NnDevice {
public:
    virtual ~NnDevice() = default;
    virtual NnDeviceSegment* createSegment(NnUint segmentIndex) = 0;
    virtual NnUint maxNThreads() { return std::thread::hardware_concurrency(); }
};

#endif
