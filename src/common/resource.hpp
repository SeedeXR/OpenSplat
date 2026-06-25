#ifndef OPENSPLAT_RESOURCE_H
#define OPENSPLAT_RESOURCE_H

// Resource-aware "process contract" for OpenSplat.
//
// OpenSplat is meant to run on machines from 2 GB desktops to 80 GB servers.
// This module makes it understand the host + the workload and pick a sane route,
// while letting the user tweak or fully override every decision.
//
// Precedence (highest first):  CLI flags  >  env vars (OPENSPLAT_*)  >  JSON
// contract file  >  named profile preset  >  auto-detection from hardware.
//
// The decision function resolvePolicy() is PURE (no I/O) so it is unit-tested
// directly; hardware detection and parsing are isolated in impure helpers.
//
// See linux_research/ for the measurements that calibrate the constants below.

#include <string>
#include <map>
#include <cstdint>

namespace rc {

enum class ImageStore { F32, U8 };
std::string imageStoreName(ImageStore s);

// Detected host capabilities.
struct HardwareInfo {
    long long ramTotalMB = 0;
    long long ramAvailMB = 0;
    int cpuCores = 0;
    bool hasCuda = false;
    long long vramTotalMB = 0;   // device 0
    long long vramFreeMB = 0;
};

// Partial overrides from a single source. A field is only honored when its
// matching *_set flag is true — this keeps the precedence merge explicit.
struct Overrides {
    std::string profile;       bool profile_set = false;
    long long ramBudgetMB = 0; bool ramBudgetMB_set = false;
    long long vramBudgetMB = 0;bool vramBudgetMB_set = false;
    long long maxGaussians = 0;bool maxGaussians_set = false;
    std::string imageStore;    bool imageStore_set = false;   // "f32" | "u8" | "auto"
    int minRenderPx = 0;       bool minRenderPx_set = false;
    float downscaleFactor = 0; bool downscaleFactor_set = false;
};

// What we know about the workload before loading images.
struct SceneInfo {
    int numImages = 0;
    int repWidth = 0;    // representative (first) camera native width
    int repHeight = 0;
};

// The resolved policy the trainer acts on.
struct Contract {
    std::string profile = "auto";
    long long ramBudgetMB = 0;
    long long vramBudgetMB = 0;
    long long maxGaussians = 0;            // 0 => unbounded
    ImageStore imageStore = ImageStore::F32;
    int minRenderPx = 400;                 // quality guard (see RESULTS.md Finding #2)
    float downscaleFactor = 1.0f;
    bool downscaleFromContract = false;    // true => contract dictates downscale
    // diagnostics
    long long estPreloadRamMB = 0;         // host RAM to hold all images in chosen store
    long long estVramAtCapMB = 0;          // VRAM the gaussian cap would occupy
    std::map<std::string, std::string> source;  // key -> winning source
};

// --- calibrated constants (see linux_research/measurements) ---
// GPU bytes per gaussian incl. params + Adam moments + gradients (deg-3 SH).
extern const double BYTES_PER_GAUSSIAN;
// Host RSS floor: libtorch + CUDA context + OpenCV, before any image is loaded.
extern const long long RUNTIME_BASELINE_MB;
// VRAM floor: CUDA context + cuDNN/cuBLAS handles + activation working set.
extern const long long VRAM_BASELINE_MB;

// --- impure helpers ---
HardwareInfo detectHardware(bool cudaAvailable);
Overrides overridesFromEnv();
Overrides overridesFromJsonFile(const std::string &path, std::string *err = nullptr);
Overrides overridesFromJsonString(const std::string &text, std::string *err = nullptr);

// --- pure decision ---
Contract resolvePolicy(const HardwareInfo &hw, const SceneInfo &scene,
                       const Overrides &cli, const Overrides &env,
                       const Overrides &json);

// Host MB to preload all images at the given bytes-per-channel-sample (1 or 4).
long long preloadRamMB(const SceneInfo &scene, int bytesPerSample);

void printContract(const Contract &c, const HardwareInfo &hw, const SceneInfo &scene);

} // namespace rc

#endif
