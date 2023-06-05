#include <math.h>

template<typename T> struct DynTypeMath {
    static __device__ T sin(T x);
    static __device__ T cos(T x);
    static __device__ T mod(T x, T y);
    static __device__ T abs(T x);
    static __device__ T floor(T x);
    static __device__ T ceil(T x);
};

template<> struct DynTypeMath<float> {
    inline static __device__ float sin(float x) { return sinf(x); }
    inline static __device__ float cos(float x) { return cosf(x); }
    inline static __device__ float mod(float x, float y) { return fmodf(x, y); }
    inline static __device__ float abs(float x) { return fabsf(x); }
    inline static __device__ float floor(float x) { return floorf(x); }
    inline static __device__ float ceil(float x) { return ceilf(x); }
};

template<> struct DynTypeMath<double> {
    inline static __device__ double sin(double x) { return sin(x); }
    inline static __device__ double cos(double x) { return cos(x); }
    inline static __device__ double mod(double x, double y) { return fmod(x, y); }
    inline static __device__ double abs(double x) { return fabs(x); }
    inline static __device__ double floor(double x) { return floor(x); }
    inline static __device__ double ceil(double x) { return ceil(x); }
};