#ifndef OMP_UTIL_H
#define OMP_UTIL_H

#include <memory>
#include <limits>
#include <cstddef>
#include <cassert>
#include <cstdint>
#if _MSC_VER
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
        // MSVC has no way to distuingish between SSE2/SSE4, so we only enable SSE4 there with AVX.
        #define OMP_SSE4 (__SSE4_1__ || (_MSC_VER && __AVX__))
    #endif
#endif

#if _MSC_VER
    #define OMP_FORCE_INLINE __forceinline
#else
    #define OMP_FORCE_INLINE
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
static inline void* alignedNew(size_t size, unsigned alignment)
{
    // Alignment must be nonzero power of two.
    omp_assert(alignment && !(alignment & (alignment - 1)));

    if (alignment < OMP_ALIGNOF(void*))
        alignment = OMP_ALIGNOF(void*);

    // Allocate enough memory so that we can find an aligned block inside it.
    char* pwrapper = (char*)::operator new(size + alignment);
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
static inline void alignedDelete(void* p)
{
    ::operator delete(static_cast<char**>(p)[-1]);
}

// Custom allocator that guarantees correct aligment inside standard containers for objects
// that have alignment greater than std::max_align_t.
template<class T>
class AlignedAllocator
{
public:
    typedef T value_type;

    AlignedAllocator() = default;

    template<class U>
    AlignedAllocator(const AlignedAllocator<U>&)
    {
    }

    T* allocate(size_t n)
    {
        return static_cast<T*>(alignedNew(n * sizeof(T), OMP_ALIGNOF(T)));
    }

    void deallocate(T* p, size_t n)
    {
        alignedDelete(p);
    }

    template<class U>
    struct rebind
    {
        typedef AlignedAllocator<U> other;
    };

    template <class U>
    bool operator==(const AlignedAllocator<U>&) const
    {
        return true;
    }

    template <class U>
    bool operator!=(const AlignedAllocator<U>&) const
    {
        return false;
    }

    // The following stuff should not be needed but some compilers aren't fully compatible.

    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t different_type;

    template<class U, class... Args>
    void construct(U* p, Args&&... args)
    {
        new(p) U(std::forward<Args>(args)...);
    }

    template<class U>
    void destroy(U* p)
    {
        p->~U();
    }

    static size_t max_size()
    {
        return std::numeric_limits<size_t>::max() / sizeof(T);
    }

    AlignedAllocator select_on_container_copy_construction() const
    {
        return AlignedAllocator();
    }
};

// Specializes std::allocator for TYPE and pair<T,TYPE>. This is for convenience so that it's
// not necessary to specify a custom allocator as a template argument for STL containers.
// It's a bit of a hack, because the specialization isn't fully conformant (rebind isn't
// symmetric). allocator_traits is specialized to fix some issues caused by that.
// On x64 the allocator isn't needed because default allocation is typically 16-byte aligned.
#if OMP_SSE2 && !OMP_X64
#define OMP_ALIGNED_STD_ALLOCATOR(TYPE) \
    namespace std { \
        template<> \
        class allocator<TYPE> : public omp::AlignedAllocator<TYPE> \
        { \
        public: \
            allocator() = default; \
            template<class U> allocator(const allocator<U>&) { } \
            template<class U> allocator(const omp::AlignedAllocator<U>&) { } \
        }; \
        template<typename T> \
        class allocator<pair<T,TYPE>> : public omp::AlignedAllocator<pair<T,TYPE>> \
        { \
        public: \
            allocator() = default; \
            template<class U> allocator(const allocator<U>&) { } \
            template<class U> allocator(const omp::AlignedAllocator<U>&) { } \
        }; \
        template <> \
        class allocator_traits<allocator<TYPE>> : public allocator_traits<omp::AlignedAllocator<TYPE>> { }; \
        template <typename T> \
        class allocator_traits<allocator<pair<T,TYPE>>> \
            : public allocator_traits<omp::AlignedAllocator<pair<T,TYPE>>> { }; \
    }
#else
    #define OMP_ALIGNED_STD_ALLOCATOR(TYPE)
#endif

}

#endif // OMP_UTIL_H
