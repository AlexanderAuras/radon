#include <iostream>

typedef float (*filterFunc)(float, float);

__device__ __noinline__ float ramLak(float x, float max) {
    return abs(x);
}

__device__ __noinline__ float hann(float x, float max) {
    return abs(x)*0.5*(1.0+cos(2.0*M_PI*x/max));
}

__device__ filterFunc _devRamLakPtr = ramLak;
__device__ filterFunc _devHannPtr = hann;
filterFunc ramLakPtr = nullptr;
filterFunc hannPtr = nullptr;

void initFilterPointers() {
    cudaGetSymbolAddress(reinterpret_cast<void**>(&ramLakPtr), _devRamLakPtr);
    cudaGetSymbolAddress(reinterpret_cast<void**>(&hannPtr), _devHannPtr);
}