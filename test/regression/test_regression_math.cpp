// Regression / characterization tests.
//
// These lock in the CURRENT, verified-by-hand numeric behaviour of stable building blocks so
// that an unintended change (refactor, dependency bump, convention flip) is caught. If one of
// these fails, do NOT just update the expected value — confirm the change is intentional first
// and add a note here referencing why. See ../README.md and ../../docs/testing.md.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <torch/torch.h>
#include "tensor_math.hpp"
#include "optim_scheduler.hpp"

TEST_CASE("REG: quatToRotMat(0.7071,0,0,0.7071) == exact 90deg-about-Z matrix") {
    const float s = 0.70710678f;
    torch::Tensor R = quatToRotMat(torch::tensor({s, 0.0f, 0.0f, s}));
    torch::Tensor expected = torch::tensor({{0.0f, -1.0f, 0.0f},
                                            {1.0f, 0.0f, 0.0f},
                                            {0.0f, 0.0f, 1.0f}});
    CHECK(torch::allclose(R, expected, 1e-4, 1e-4));
}

TEST_CASE("REG: rodriguesToRotation([0,0,pi/2]) == exact 90deg-about-Z matrix") {
    torch::Tensor R = rodriguesToRotation(torch::tensor({0.0f, 0.0f, 1.57079633f}));
    torch::Tensor expected = torch::tensor({{0.0f, -1.0f, 0.0f},
                                            {1.0f, 0.0f, 0.0f},
                                            {0.0f, 0.0f, 1.0f}});
    CHECK(torch::allclose(R, expected, 1e-4, 1e-4));
}

TEST_CASE("REG: LR schedule 1e-2 -> 1e-4 over 1000 steps stays log-linear") {
    std::vector<torch::Tensor> params{torch::zeros({1}, torch::TensorOptions().requires_grad(true))};
    torch::optim::Adam opt(params, torch::optim::AdamOptions(1e-2));
    OptimScheduler sched(&opt, 1e-4f, 1000);
    CHECK(sched.getLearningRate(0)    == doctest::Approx(1e-2).epsilon(0.005));
    CHECK(sched.getLearningRate(250)  == doctest::Approx(3.1623e-3).epsilon(0.02)); // 10^-2.5
    CHECK(sched.getLearningRate(500)  == doctest::Approx(1e-3).epsilon(0.01));
    CHECK(sched.getLearningRate(750)  == doctest::Approx(3.1623e-4).epsilon(0.02)); // 10^-3.5
    CHECK(sched.getLearningRate(1000) == doctest::Approx(1e-4).epsilon(0.005));
}
