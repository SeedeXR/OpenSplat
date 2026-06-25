// CUDA integration test: drives the real GPU training pipeline end-to-end
// (ProjectGaussians -> SphericalHarmonics -> RasterizeGaussians CUDA kernels +
// the gaussian model + the resource-aware --max-splats cap).
//
// Compiled with USE_CUDA and linked against the CUDA `gsplat` library, so it
// only exists in a CUDA build. It is DOUBLY gated at runtime: skips cleanly if
// no CUDA device is present (e.g. a GitHub-hosted runner) or no test scene is
// available (env OPENSPLAT_TEST_DATA, default ./data/db/drjohnson).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// Torch/model headers pull in glog, which #defines its own CHECK macro. Include
// them FIRST, then undef CHECK so doctest's CHECK (not glog's LOG(FATAL)) is used.
#include <torch/torch.h>
#include <filesystem>
#include <cstdlib>
#include <numeric>
#include "input_data.hpp"
#include "model.hpp"
#include "utils.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include <doctest/doctest.h>

namespace fs = std::filesystem;

static std::string testScene() {
    if (const char *e = std::getenv("OPENSPLAT_TEST_DATA")) return e;
    return "data/db/drjohnson";
}

// Run a short CUDA training loop, returning the final gaussian count.
// initialOut (optional) receives the gaussian count right after construction.
static long long trainShort(const std::string &scene, long long maxSplats,
                            bool storeU8, int numIters, float *finalLossOut,
                            long long *initialOut = nullptr) {
    InputData inputData = inputDataFromX(scene, "");
    // Cap images loaded so the test is quick and light.
    if (inputData.cameras.size() > 12) inputData.cameras.resize(12);
    parallel_for(inputData.cameras.begin(), inputData.cameras.end(),
                 [&](Camera &cam) { cam.loadImage(1.0f, storeU8); });

    torch::Device device(torch::kCUDA);
    Model model(inputData, inputData.cameras.size(),
                2 /*numDownscales*/, 3000 /*resSchedule*/, 3 /*shDegree*/, 1000 /*shInterval*/,
                100 /*refineEvery*/, 500 /*warmup*/, 30 /*resetAlphaEvery*/,
                0.0002f, 0.01f, 4000 /*stopScreenSizeAt*/, 0.05f,
                numIters, false, device, maxSplats);
    if (initialOut) *initialOut = model.means.size(0);

    std::vector<size_t> idx(inputData.cameras.size());
    std::iota(idx.begin(), idx.end(), 0);
    InfiniteRandomIterator<size_t> camsIter(idx);

    float loss = 0.0f;
    for (int step = 1; step <= numIters; step++) {
        Camera &cam = inputData.cameras[camsIter.next()];
        model.optimizersZeroGrad();
        torch::Tensor rgb = model.forward(cam, step);
        torch::Tensor gt = cam.getImage(model.getDownscaleFactor(step)).to(device);
        torch::Tensor l = model.mainLoss(rgb, gt, 0.2f);
        l.backward();
        model.optimizersStep();
        model.schedulersStep(step);
        model.afterTrain(step);
        loss = l.item<float>();
    }
    if (finalLossOut) *finalLossOut = loss;
    return model.means.size(0);
}

TEST_CASE("cuda: training runs on GPU and produces a finite loss (data-gated)") {
    if (!torch::cuda::is_available()) { MESSAGE("skip: no CUDA device"); return; }
    const std::string scene = testScene();
    if (!fs::exists(fs::path(scene) / "sparse")) { MESSAGE("skip: no scene at " << scene); return; }

    float loss = 0.0f;
    long long g = trainShort(scene, /*maxSplats=*/0, /*storeU8=*/false, /*numIters=*/300, &loss);
    CHECK(g > 0);
    CHECK(std::isfinite(loss));
    CHECK(loss >= 0.0f);
}

TEST_CASE("cuda: --max-splats bounds densification growth (data-gated)") {
    if (!torch::cuda::is_available()) { MESSAGE("skip: no CUDA device"); return; }
    const std::string scene = testScene();
    if (!fs::exists(fs::path(scene) / "sparse")) { MESSAGE("skip: no scene at " << scene); return; }

    // numIters=2200 => stopSplitAt=1100 > warmup 500, so densification is active.
    // All chunks initialize from the full sparse cloud, so derive the cap from the
    // measured initial count (cap above init => growth happens, then the cap engages).
    float lossUncapped = 0.0f, lossCapped = 0.0f;
    long long initCount = 0;
    long long uncapped = trainShort(scene, /*maxSplats=*/0, false, 2200, &lossUncapped, &initCount);
    const long long growth = uncapped - initCount;
    // Derive the cap from the OBSERVED growth so the test is meaningful by construction
    // (cap < uncapped always). Skip if densification barely grew on this scene/GPU.
    if (growth < 4000){
        MESSAGE("skipping cap check: uncapped growth " << growth << " too small to be meaningful");
        return;
    }
    const long long cap = initCount + growth / 2;
    long long capped = trainShort(scene, cap, false, 2200, &lossCapped);

    CHECK(std::isfinite(lossCapped));
    CHECK(std::isfinite(lossUncapped));
    CHECK(uncapped > cap);          // meaningful by construction (cap = init + growth/2)
    CHECK(capped < uncapped);       // the cap constrained growth vs the same uncapped run
    CHECK(capped <= cap + growth);  // bounded: soft-cap overshoot <= one densification batch
}

TEST_CASE("cuda: u8 store yields a bit-identical GT image on GPU (zero quality loss)") {
    if (!torch::cuda::is_available()) { MESSAGE("skip: no CUDA device"); return; }
    const std::string scene = testScene();
    if (!fs::exists(fs::path(scene) / "sparse")) { MESSAGE("skip: no scene at " << scene); return; }

    // The true "u8 store costs nothing" property is deterministic: source images
    // are uint8, so the GT tensor the trainer consumes must be bit-identical
    // whether stored as f32 or u8. (Post-training losses can't be compared to
    // 1e-6 on CUDA — the rasterizer backward uses non-deterministic atomicAdd.)
    InputData a = inputDataFromX(scene, "");
    InputData b = inputDataFromX(scene, "");
    a.cameras.resize(1);
    b.cameras.resize(1);
    a.cameras[0].loadImage(1.0f, /*storeU8=*/false);
    b.cameras[0].loadImage(1.0f, /*storeU8=*/true);

    torch::Device dev(torch::kCUDA);
    CHECK(torch::equal(a.cameras[0].getImage(1).to(dev), b.cameras[0].getImage(1).to(dev)));
    // ...and through the downscale-pyramid path too.
    CHECK(torch::equal(a.cameras[0].getImage(2).to(dev), b.cameras[0].getImage(2).to(dev)));
}
