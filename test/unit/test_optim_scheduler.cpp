// Unit tests for src/model/optim_scheduler.cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <torch/torch.h>
#include "optim_scheduler.hpp"

static torch::optim::Adam makeAdam(float lr) {
    std::vector<torch::Tensor> params{torch::zeros({3}, torch::TensorOptions().requires_grad(true))};
    return torch::optim::Adam(params, torch::optim::AdamOptions(lr));
}

static float currentLr(torch::optim::Adam &opt) {
    return static_cast<torch::optim::AdamOptions &>(opt.param_groups()[0].options()).get_lr();
}

TEST_CASE("getLearningRate: endpoints equal lrInit and lrFinal") {
    auto opt = makeAdam(1e-2f);
    OptimScheduler sched(&opt, /*lrFinal=*/1e-4f, /*maxSteps=*/1000);
    CHECK(sched.getLearningRate(0) == doctest::Approx(1e-2).epsilon(0.01));
    CHECK(sched.getLearningRate(1000) == doctest::Approx(1e-4).epsilon(0.01));
}

TEST_CASE("getLearningRate: midpoint is the geometric mean (log-linear interp)") {
    auto opt = makeAdam(1e-2f);
    OptimScheduler sched(&opt, 1e-4f, 1000);
    // exp(0.5*(ln 1e-2 + ln 1e-4)) == sqrt(1e-2 * 1e-4) == 1e-3
    CHECK(sched.getLearningRate(500) == doctest::Approx(1e-3).epsilon(0.02));
}

TEST_CASE("getLearningRate: monotonically decreasing and clamped past maxSteps") {
    auto opt = makeAdam(1e-2f);
    OptimScheduler sched(&opt, 1e-4f, 1000);
    CHECK(sched.getLearningRate(100) > sched.getLearningRate(900));
    CHECK(sched.getLearningRate(5000) == doctest::Approx(sched.getLearningRate(1000)).epsilon(1e-4));
}

TEST_CASE("step() writes the scheduled learning rate into the optimizer") {
    auto opt = makeAdam(1e-2f);
    OptimScheduler sched(&opt, 1e-4f, 1000);
    sched.step(1000);
    CHECK(currentLr(opt) == doctest::Approx(1e-4).epsilon(0.01));
    sched.step(0);
    CHECK(currentLr(opt) == doctest::Approx(1e-2).epsilon(0.01));
}
