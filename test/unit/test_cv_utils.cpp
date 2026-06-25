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

TEST_CASE("imageToU8Tensor: owning uint8 HWC, no normalization") {
    torch::Tensor t = torch::rand({12, 10, 3});
    cv::Mat img = tensorToImage(t);              // CV_8UC3
    torch::Tensor u8 = imageToU8Tensor(img);
    CHECK(u8.dtype() == torch::kUInt8);
    CHECK(u8.sizes() == torch::IntArrayRef({12, 10, 3}));
    // Must own its memory (survives the cv::Mat going away) — mutate the Mat, tensor unchanged.
    int64_t before = u8.sum().item<int64_t>();
    img.setTo(cv::Scalar(0, 0, 0));
    CHECK(u8.sum().item<int64_t>() == before);
}

TEST_CASE("u8 store is bit-identical to f32-after-/255 (zero quality loss)") {
    // The resource-aware u8 image store relies on this: source images are uint8,
    // so storing u8 and converting on demand == storing float32/255 exactly.
    torch::Tensor t = torch::rand({16, 16, 3});
    cv::Mat img = tensorToImage(t);
    torch::Tensor viaF32 = imageToTensor(img);                       // float path
    torch::Tensor viaU8  = imageToU8Tensor(img).toType(torch::kFloat32) / 255.0f;  // u8 path
    CHECK(torch::equal(viaF32, viaU8));   // exact, not approx
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
