// Unit tests for src/common/tensor_math.cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <torch/torch.h>
#include "tensor_math.hpp"

using namespace torch::indexing;

static bool close(const torch::Tensor &a, const torch::Tensor &b, double atol = 1e-4) {
    return torch::allclose(a, b, /*rtol=*/1e-4, atol);
}

TEST_CASE("quatToRotMat: identity quaternion -> identity rotation") {
    torch::Tensor q = torch::tensor({1.0f, 0.0f, 0.0f, 0.0f});  // (w,x,y,z)
    torch::Tensor R = quatToRotMat(q);
    CHECK(R.sizes() == torch::IntArrayRef({3, 3}));
    CHECK(close(R, torch::eye(3)));
}

TEST_CASE("quatToRotMat: 90deg about Z maps +X -> +Y") {
    const float s = 0.70710678f;
    torch::Tensor R = quatToRotMat(torch::tensor({s, 0.0f, 0.0f, s}));
    torch::Tensor x = torch::tensor({{1.0f}, {0.0f}, {0.0f}});
    torch::Tensor rotated = torch::matmul(R, x).squeeze();
    CHECK(close(rotated, torch::tensor({0.0f, 1.0f, 0.0f}), 1e-3));
}

TEST_CASE("quatToRotMat: output is a valid rotation (orthonormal, det=1)") {
    torch::Tensor R = quatToRotMat(torch::tensor({0.5f, 0.5f, 0.5f, 0.5f}));
    CHECK(close(torch::matmul(R, R.transpose(0, 1)), torch::eye(3), 1e-3));
    CHECK(torch::det(R).item<float>() == doctest::Approx(1.0).epsilon(0.01));
}

TEST_CASE("quatToRotMat: normalizes its input (scaled quat == unit quat)") {
    torch::Tensor R1 = quatToRotMat(torch::tensor({1.0f, 0.0f, 0.0f, 1.0f}));
    torch::Tensor R2 = quatToRotMat(torch::tensor({0.70710678f, 0.0f, 0.0f, 0.70710678f}));
    CHECK(close(R1, R2, 1e-3));
}

TEST_CASE("rodriguesToRotation: zero vector -> identity") {
    CHECK(close(rodriguesToRotation(torch::zeros({3})), torch::eye(3)));
}

TEST_CASE("rodriguesToRotation: pi/2 about Z matches the 90deg rotation") {
    const float halfPi = 1.57079633f;
    torch::Tensor R = rodriguesToRotation(torch::tensor({0.0f, 0.0f, halfPi}));
    torch::Tensor expected = torch::tensor({{0.0f, -1.0f, 0.0f},
                                            {1.0f, 0.0f, 0.0f},
                                            {0.0f, 0.0f, 1.0f}});
    CHECK(close(R, expected, 1e-3));
}

TEST_CASE("rotationMatrix: rotates a onto the direction of b") {
    torch::Tensor a = torch::tensor({1.0f, 0.0f, 0.0f});
    torch::Tensor b = torch::tensor({0.0f, 2.0f, 0.0f});  // non-unit on purpose
    torch::Tensor R = rotationMatrix(a, b);
    torch::Tensor mapped = torch::matmul(R, a / a.norm());
    CHECK(close(mapped, b / b.norm(), 1e-3));
}

TEST_CASE("autoScaleAndCenterPoses: centers origins at mean and scales to unit max") {
    // Two camera poses with translations on the X axis.
    torch::Tensor poses = torch::eye(4).unsqueeze(0).repeat({2, 1, 1});
    poses[0].index_put_({Slice(None, 3), 3}, torch::tensor({0.0f, 0.0f, 0.0f}));
    poses[1].index_put_({Slice(None, 3), 3}, torch::tensor({2.0f, 0.0f, 0.0f}));

    auto r = autoScaleAndCenterPoses(poses);
    torch::Tensor out = std::get<0>(r);
    torch::Tensor center = std::get<1>(r);
    float scale = std::get<2>(r);

    CHECK(close(center, torch::tensor({1.0f, 0.0f, 0.0f}), 1e-4));   // mean of origins
    // After centering, origins are +/-1 on X; max |origin| == 1 so scale == 1.
    CHECK(scale == doctest::Approx(1.0).epsilon(0.01));
    torch::Tensor origins = out.index({"...", Slice(None, 3), 3});
    CHECK(torch::abs(origins).max().item<float>() == doctest::Approx(1.0).epsilon(0.01));
}
