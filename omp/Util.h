#ifndef OMP_UTIL_H
#define OMP_UTIL_H

#include <cstddef>
#include <cassert>

#if _MSC_VER
#include <memory>
#include <limits>
#include <intrin.h>
#endif

// Detect x64.
#if defined(__amd64) || defined(_M_X64)
    #define OMP_X64 1
#endif

// Detect SSE2/SSE4.
#ifndef OMP_SSE2
    #if (__SSE2__ || (_MSC_VER && (_M_X64 || _M_IX86_FP >= 2)))
        #define OMP_SSE2 1
        // MSVC has no way to distuingish between SSE2/SSE4, so we only enable SSE4 with AVX.
		#define OMP_SSE4 (__SSE4_1__ || (_MSC_VER && __AVX__))
    #endif
#endif

namespace omp {

inline unsigned countTrailingZeros(unsigned x)
{
    #if _MSC_VER
    unsigned long bitIdx;
    _BitScanForward(&bitIdx, x);
    return bitIdx;
    #else
    return __builtin_ctz(x);
    #endif
}

inline unsigned countLeadingZeros(unsigned x)
{
    #if _MSC_VER
    unsigned long bitIdx;
    _BitScanReverse(&bitIdx, x);
    return 31 - bitIdx;
    #else
    return __builtin_clz(x);
    #endif
}

inline unsigned bitCount(unsigned x)
{
    #if _MSC_VER
    return __popcnt(x);
    #else
    return __builtin_popcount(x);
    #endif
}

inline unsigned bitCount(unsigned long x)
{
    #if _MSC_VER
    return bitCount((unsigned)x);
    #else
    return __builtin_popcountl(x);
    #endif
}

inline unsigned bitCount(unsigned long long x)
{
    #if _MSC_VER && _M_X64
    return (unsigned)__popcnt64(x);
    #elif _MSC_VER
    return __popcnt((unsigned)x) + __popcnt((unsigned)(x >> 32));
    #else
    return __builtin_popcountll(x);
    #endif
}

#if OMP_ASSERT
    #define omp_assert(x) assert(x)
#else
    #define omp_assert(x) do { } while(0)
#endif

#if _MSC_VER
    #define OMP_ALIGNOF(x) __alignof(x)
#else
    #define OMP_ALIGNOF(x) alignof(x)
#endif

// Allocates aligned memory.
static void* alignedNew(size_t size, unsigned alignment)
{
    // Alignment must be nonzero power of two.
    omp_assert(alignment && !(alignment & (alignment - 1)));

    if (alignment < OMP_ALIGNOF(void*))
        alignment = OMP_ALIGNOF(void*);

    // Allocate enough memory so that we can find an aligned block inside it.
    char* pwrapper = new char[size + alignment];
    if (!pwrapper)
        return nullptr;

    // Find next aligned pointer.
    char* p = pwrapper + alignment;
    p -= reinterpret_cast<uintptr_t>(p) & (alignment - 1);

    // Save the original pointer before the start of the aligned block. (Needed for dealloc.)
    reinterpret_cast<char**>(p)[-1] = pwrapper;

    return p;
}

// Deallocates memory allocated by alignedNew().
static void alignedDelete(void* p)
{
    delete[] static_cast<char**>(p)[-1];
}

// Custom allocator for standard library containers that guarantees correct aligment.
template<class T, size_t tAlignment = OMP_ALIGNOF(T)>
class AlignedAllocator
{
public:
    typedef T value_type;

    T* allocate(size_t n)
    {
        return static_cast<T*>(alignedNew(n * sizeof(T), tAlignment));
    }

    void deallocate(T* p, size_t n)
    {
        alignedDelete(p);
    }

    // MSVC2013 doesn't support allocator traits correctly.
    #if _MSC_VER

    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t different_type;

    static size_t max_size()
    {
        return std::numeric_limits<size_t>::max() / sizeof(T);
    }

    template< class U, class... Args >
    void construct(U* p, Args&&... args)
    {
        new(p)T(args...);
    }

    void destroy(pointer p)
    {
        p->~T();
    }

    std::allocator<T> select_on_container_copy_construction() const
    {
        return std::allocator<T>();
    }

    #endif
};

#if OMP_SSE2 && !OMP_X64
#define OMP_DEFINE_ALIGNED_ALLOCATOR(T) \
    namespace std { \
    template<> \
    class allocator<T> : public omp::AlignedAllocator<T, sizeof(T)> \
    { \
    }; \
    }
#else
    #define OMP_DEFINE_ALIGNED_ALLOCATOR(T)
#endif

}

#endif // OMP_UTIL_H
