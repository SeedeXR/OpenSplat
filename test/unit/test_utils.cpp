// Unit tests for src/common/utils.hpp (header-only — no Torch needed).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "utils.hpp"

#include <vector>
#include <atomic>
#include <numeric>
#include <algorithm>

TEST_CASE("InfiniteRandomIterator yields a full permutation each cycle") {
    std::vector<int> v{1, 2, 3, 4, 5};
    InfiniteRandomIterator<int> it(v);
    std::vector<int> got;
    for (int k = 0; k < 5; ++k) got.push_back(it.next());
    std::sort(got.begin(), got.end());
    CHECK(got == std::vector<int>({1, 2, 3, 4, 5}));

    // Continues indefinitely past the end (reshuffles), always in-range.
    for (int k = 0; k < 20; ++k) {
        int n = it.next();
        CHECK(n >= 1);
        CHECK(n <= 5);
    }
}

TEST_CASE("InfiniteRandomIterator is deterministic (fixed seed 42)") {
    std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> b = a;
    InfiniteRandomIterator<int> ia(a), ib(b);
    for (int k = 0; k < 16; ++k) CHECK(ia.next() == ib.next());
}

TEST_CASE("parallel_for visits every element exactly once") {
    std::vector<int> v(1000);
    std::iota(v.begin(), v.end(), 0);
    std::atomic<long> sum{0};
    parallel_for(v.begin(), v.end(), [&](int x) { sum += x; });
    CHECK(sum.load() == 999L * 1000 / 2);
}

TEST_CASE("parallel_for is a no-op on an empty range") {
    std::vector<int> v;
    std::atomic<int> cnt{0};
    parallel_for(v.begin(), v.end(), [&](int) { cnt++; });
    CHECK(cnt.load() == 0);
}

TEST_CASE("RELEASE_SAFELY deletes and nulls a pointer") {
    int *p = new int(7);
    RELEASE_SAFELY(p);
    CHECK(p == nullptr);
    RELEASE_SAFELY(p);  // safe to call again
    CHECK(p == nullptr);
}
