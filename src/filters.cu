template <typename T> __forceinline__ __device__ T ram_lak(T x, T max) {
    return abs(x);
}

template <typename T> __forceinline__ __device__ T hann(T x, T max) {
    return abs(x)*0.5*(1.0+cos(2.0*M_PI*x/max));
}