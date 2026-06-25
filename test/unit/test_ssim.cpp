// Unit tests for src/render/ssim.cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <torch/torch.h>
#include "ssim.hpp"

// SSIM expects HxWxC images (it permutes to NCHW internally).
TEST_CASE("SSIM of an image with itself is 1.0") {
    torch::manual_seed(0);
    SSIM ssim(11, 3);
    torch::Tensor img = torch::rand({32, 32, 3});
    float s = ssim.eval(img, img).item<float>();
    CHECK(s == doctest::Approx(1.0).epsilon(0.01));
}

TEST_CASE("SSIM of dissimilar images is well below 1.0") {
    torch::manual_seed(1);
    SSIM ssim(11, 3);
    torch::Tensor a = torch::zeros({32, 32, 3});
    torch::Tensor b = torch::ones({32, 32, 3});
    float s = ssim.eval(a, b).item<float>();
    CHECK(s < 0.5f);
}

TEST_CASE("SSIM is symmetric in its arguments") {
    torch::manual_seed(2);
    SSIM ssim(11, 3);
    torch::Tensor a = torch::rand({24, 24, 3});
    torch::Tensor b = torch::rand({24, 24, 3});
    float sab = ssim.eval(a, b).item<float>();
    float sba = ssim.eval(b, a).item<float>();
    CHECK(sab == doctest::Approx(sba).epsilon(0.001));
}
