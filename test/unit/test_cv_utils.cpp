// Unit tests for src/common/cv_utils.cpp (tensor <-> cv::Mat conversions)
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <torch/torch.h>
#include "cv_utils.hpp"

TEST_CASE("imageToTensor . tensorToImage roundtrips an HxWx3 image (within 1/255)") {
    torch::manual_seed(0);
    torch::Tensor t = torch::rand({16, 16, 3});  // float in [0,1], HWC
    cv::Mat img = tensorToImage(t);
    CHECK(img.rows == 16);
    CHECK(img.cols == 16);
    CHECK(img.channels() == 3);
    torch::Tensor back = imageToTensor(img);
    CHECK(back.sizes() == t.sizes());
    // tensorToImage truncates to uint8, so tolerance ~ 1/255.
    CHECK(torch::allclose(back, t, /*rtol=*/0.0, /*atol=*/0.02));
}

TEST_CASE("tensorToImage rejects non-3-channel input") {
    torch::Tensor gray = torch::rand({8, 8, 1});
    CHECK_THROWS(tensorToImage(gray));
}

TEST_CASE("floatNxNMatToTensor . floatNxNtensorToMat roundtrips a 2D float matrix") {
    torch::Tensor m = torch::rand({8, 8}).contiguous();
    cv::Mat mat = floatNxNtensorToMat(m);   // shares m's storage
    CHECK(mat.rows == 8);
    CHECK(mat.cols == 8);
    torch::Tensor back = floatNxNMatToTensor(mat);  // clones
    CHECK(torch::allclose(back, m, 0.0, 1e-6));
}
