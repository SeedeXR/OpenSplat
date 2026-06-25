// Unit tests for src/model/kdtree_tensor.cpp (PointsTensor::scales)
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <torch/torch.h>
#include "kdtree_tensor.hpp"

// scales() = mean distance to the 3 nearest neighbours of each point.
TEST_CASE("scales: shape is [N,1] and all values are positive and finite") {
    torch::Tensor pts = torch::tensor({{0.f, 0.f, 0.f},
                                       {1.f, 0.f, 0.f},
                                       {0.f, 1.f, 0.f},
                                       {1.f, 1.f, 0.f},
                                       {0.f, 0.f, 1.f}}, torch::kFloat32);
    PointsTensor pt(pts);
    torch::Tensor s = pt.scales();
    CHECK(s.sizes() == torch::IntArrayRef({5, 1}));
    CHECK(s.min().item<float>() > 0.0f);
    CHECK(std::isfinite(s.max().item<float>()));
}

TEST_CASE("scales: an isolated point has a larger scale than a clustered one") {
    torch::Tensor pts = torch::tensor({{0.f, 0.f, 0.f},   // tight unit-square cluster
                                       {1.f, 0.f, 0.f},
                                       {0.f, 1.f, 0.f},
                                       {1.f, 1.f, 0.f},
                                       {100.f, 100.f, 100.f}},  // far away
                                      torch::kFloat32);
    PointsTensor pt(pts);
    torch::Tensor s = pt.scales();
    CHECK(s[4].item<float>() > s[0].item<float>());
}

TEST_CASE("scales: a uniformly translated cloud yields identical scales") {
    torch::Tensor base = torch::rand({16, 3});
    PointsTensor a(base);
    torch::Tensor offset = base + 50.0f;
    PointsTensor b(offset);
    CHECK(torch::allclose(a.scales(), b.scales(), 1e-3, 1e-3));
}
