#ifndef CV_UTILS
#define CV_UTILS

#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat imreadRGB(const std::string &filename);
void imwriteRGB(const std::string &filename, const cv::Mat &image);
cv::Mat floatNxNtensorToMat(const torch::Tensor &t);
torch::Tensor floatNxNMatToTensor(const cv::Mat &m);
cv::Mat tensorToImage(const torch::Tensor &t);
torch::Tensor imageToTensor(const cv::Mat &image);
// Owning uint8 HWC tensor (no normalization). Used by the resource-aware
// image store (4x less host RAM than the float32 path) — see src/common/resource.hpp.
torch::Tensor imageToU8Tensor(const cv::Mat &image);

#endif