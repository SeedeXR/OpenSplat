#include "resource.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#include <nlohmann/json.hpp>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace rc {

// Calibrated on an NVIDIA A100-80GB / Colab box, OpenSplat 1.1.5, deg-3 SH,
// drjohnson @ 1332x876. See linux_research/measurements/.
//   BYTES_PER_GAUSSIAN: measured (954-696) MiB / (208929-66449) gaussians = ~1899
//     B/gaussian (params + Adam moments + grads); matches the README's "~2 GB per
//     million gaussians". We keep 2000 (conservative — over-budgets VRAM, which is safe).
//   VRAM_BASELINE_MB: measured ~576 MB fixed (CUDA ctx + handles + activations) -> 600.
//   RUNTIME_BASELINE_MB: host RSS regression over 2..128 images had intercept ~1423 MB
//     (libtorch + CUDA ctx + OpenCV), slope ~23 MB/image (f32 incl. pyramids) -> 1400.
const double BYTES_PER_GAUSSIAN = 2000.0;
const long long RUNTIME_BASELINE_MB = 1400;
const long long VRAM_BASELINE_MB = 600;

// Fraction of the budget we allow a fixed estimate to consume before switching
// strategy. Headroom absorbs estimate error + transient spikes.
static constexpr double RAM_SAFETY = 0.85;     // host RAM fit (image store / downscale)
static constexpr double VRAM_HEADROOM = 0.85;  // VRAM left for activations atop the gaussian cap

std::string imageStoreName(ImageStore s) {
    return s == ImageStore::U8 ? "u8" : "f32";
}

// ---- small helpers --------------------------------------------------------
static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

// profile name -> RAM budget in MB, or -1 if not a budget profile.
static long long profileRamMB(const std::string &p) {
    std::string lp = lower(p);
    if (lp == "2gb") return 2048;
    if (lp == "4gb") return 4096;
    if (lp == "6gb") return 6144;
    if (lp == "8gb") return 8192;
    return -1;
}
static bool isFullThrottle(const std::string &p) {
    std::string lp = lower(p);
    return lp == "full-throttle" || lp == "full" || lp == "max" || lp == "throttle";
}

long long preloadRamMB(const SceneInfo &scene, int bytesPerSample) {
    if (scene.numImages <= 0 || scene.repWidth <= 0 || scene.repHeight <= 0) return 0;
    // 3 channels. Add ~6% for the progressive downscale pyramid (1/4 + 1/16 ...).
    double bytes = (double)scene.numImages * scene.repWidth * scene.repHeight * 3.0
                   * bytesPerSample * 1.06;
    return (long long)(bytes / (1024.0 * 1024.0));
}

// ---- detection ------------------------------------------------------------
static void readMeminfo(long long &totalMB, long long &availMB) {
    totalMB = availMB = 0;
    std::ifstream f("/proc/meminfo");
    std::string key;
    long long val;
    std::string unit;
    while (f >> key >> val >> unit) {
        if (key == "MemTotal:") totalMB = val / 1024;
        else if (key == "MemAvailable:") availMB = val / 1024;
    }
}

static bool queryNvidiaSmi(long long &totalMB, long long &freeMB) {
    totalMB = freeMB = 0;
    FILE *p = popen("nvidia-smi --id=0 "
                    "--query-gpu=memory.total,memory.free "
                    "--format=csv,noheader,nounits 2>/dev/null", "r");
    if (!p) return false;
    char buf[256] = {0};
    bool ok = false;
    if (fgets(buf, sizeof(buf), p)) {
        long long t = 0, fr = 0;
        if (sscanf(buf, "%lld , %lld", &t, &fr) == 2 ||
            sscanf(buf, "%lld, %lld", &t, &fr) == 2 ||
            sscanf(buf, "%lld,%lld", &t, &fr) == 2) {
            totalMB = t; freeMB = fr; ok = (t > 0);
        }
    }
    pclose(p);
    return ok;
}

HardwareInfo detectHardware(bool cudaAvailable) {
    HardwareInfo hw;
    readMeminfo(hw.ramTotalMB, hw.ramAvailMB);
#if defined(_SC_NPROCESSORS_ONLN)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    hw.cpuCores = n > 0 ? (int)n : (int)std::thread::hardware_concurrency();
#else
    hw.cpuCores = (int)std::thread::hardware_concurrency();
#endif
    hw.hasCuda = cudaAvailable;
    if (cudaAvailable) queryNvidiaSmi(hw.vramTotalMB, hw.vramFreeMB);
    return hw;
}

// ---- parsing --------------------------------------------------------------
Overrides overridesFromEnv() {
    Overrides o;
    auto envs = [](const char *k) -> const char * { return std::getenv(k); };
    if (const char *v = envs("OPENSPLAT_PROFILE")) { o.profile = v; o.profile_set = true; }
    if (const char *v = envs("OPENSPLAT_RAM_BUDGET_MB")) { o.ramBudgetMB = atoll(v); o.ramBudgetMB_set = true; }
    if (const char *v = envs("OPENSPLAT_VRAM_BUDGET_MB")) { o.vramBudgetMB = atoll(v); o.vramBudgetMB_set = true; }
    if (const char *v = envs("OPENSPLAT_MAX_SPLATS")) { o.maxGaussians = atoll(v); o.maxGaussians_set = true; }
    if (const char *v = envs("OPENSPLAT_IMAGE_STORE")) { o.imageStore = v; o.imageStore_set = true; }
    if (const char *v = envs("OPENSPLAT_MIN_RENDER_PX")) { o.minRenderPx = atoi(v); o.minRenderPx_set = true; }
    if (const char *v = envs("OPENSPLAT_DOWNSCALE_FACTOR")) { o.downscaleFactor = (float)atof(v); o.downscaleFactor_set = true; }
    return o;
}

static Overrides parseJson(const nlohmann::json &j) {
    Overrides o;
    auto getstr = [&](const char *k, std::string &dst, bool &flag) {
        if (j.contains(k) && j[k].is_string()) { dst = j[k].get<std::string>(); flag = true; }
    };
    auto geti = [&](const char *k, long long &dst, bool &flag) {
        if (j.contains(k) && j[k].is_number()) { dst = (long long)j[k].get<double>(); flag = true; }
    };
    getstr("profile", o.profile, o.profile_set);
    geti("ram_budget_mb", o.ramBudgetMB, o.ramBudgetMB_set);
    geti("vram_budget_mb", o.vramBudgetMB, o.vramBudgetMB_set);
    geti("max_gaussians", o.maxGaussians, o.maxGaussians_set);
    getstr("image_store", o.imageStore, o.imageStore_set);
    if (j.contains("min_render_px") && j["min_render_px"].is_number()) {
        o.minRenderPx = j["min_render_px"].get<int>(); o.minRenderPx_set = true;
    }
    if (j.contains("downscale_factor") && j["downscale_factor"].is_number()) {
        o.downscaleFactor = j["downscale_factor"].get<float>(); o.downscaleFactor_set = true;
    }
    return o;
}

Overrides overridesFromJsonString(const std::string &text, std::string *err) {
    try {
        return parseJson(nlohmann::json::parse(text));
    } catch (const std::exception &e) {
        if (err) *err = e.what();
        return Overrides{};
    }
}

Overrides overridesFromJsonFile(const std::string &path, std::string *err) {
    std::ifstream f(path);
    if (!f) { if (err) *err = "could not open contract file: " + path; return Overrides{}; }
    std::stringstream ss; ss << f.rdbuf();
    return overridesFromJsonString(ss.str(), err);
}

// ---- pure decision --------------------------------------------------------
Contract resolvePolicy(const HardwareInfo &hw, const SceneInfo &scene,
                       const Overrides &cli, const Overrides &env,
                       const Overrides &json) {
    Contract c;

    // Generic "pick first source that set this field" with provenance. One
    // generic lambda serves string, integer (and any) fields — the return type
    // is deduced from the pointer-to-member, so there is a single precedence
    // ladder to maintain. (float downscale_factor is handled explicitly below
    // because it additionally toggles downscaleFromContract.)
    auto pick = [&](auto memberSet, auto member, const char *key,
                    auto autoVal, const char *autoSrc) {
        if (cli.*memberSet)  { c.source[key] = "cli";  return (decltype(autoVal))(cli.*member); }
        if (env.*memberSet)  { c.source[key] = "env";  return (decltype(autoVal))(env.*member); }
        if (json.*memberSet) { c.source[key] = "json"; return (decltype(autoVal))(json.*member); }
        c.source[key] = autoSrc; return autoVal;
    };
    auto pickStr = [&](auto ms, auto m, const char *k, const std::string &a, const char *s) {
        return pick(ms, m, k, a, s);
    };
    auto pickInt = [&](auto ms, auto m, const char *k, long long a, const char *s) {
        return pick(ms, m, k, a, s);
    };

    // 1. profile
    c.profile = pickStr(&Overrides::profile_set, &Overrides::profile, "profile", "auto", "default");
    const long long profRam = profileRamMB(c.profile);
    const bool full = isFullThrottle(c.profile);

    // 2. RAM budget
    long long autoRam;
    const char *ramSrc;
    if (profRam > 0)      { autoRam = profRam; ramSrc = "profile"; }
    else if (full)        { autoRam = hw.ramTotalMB; ramSrc = "profile"; }
    else                  { autoRam = hw.ramAvailMB > 0 ? hw.ramAvailMB : hw.ramTotalMB; ramSrc = "auto"; }
    c.ramBudgetMB = pickInt(&Overrides::ramBudgetMB_set, &Overrides::ramBudgetMB,
                            "ram_budget_mb", autoRam, ramSrc);

    // 3. VRAM budget
    long long autoVram;
    const char *vramSrc;
    if (full)                  { autoVram = hw.vramTotalMB; vramSrc = "profile"; }
    else if (hw.vramFreeMB > 0){ autoVram = hw.vramFreeMB; vramSrc = "auto"; }
    else                       { autoVram = hw.vramTotalMB; vramSrc = "auto"; }
    c.vramBudgetMB = pickInt(&Overrides::vramBudgetMB_set, &Overrides::vramBudgetMB,
                             "vram_budget_mb", autoVram, vramSrc);

    // 4. image store (host RAM strategy). U8 is 4x smaller, no quality loss.
    const long long estF32 = preloadRamMB(scene, 4);
    const long long estU8  = preloadRamMB(scene, 1);
    std::string storeStr = pickStr(&Overrides::imageStore_set, &Overrides::imageStore,
                                   "image_store", "auto", "auto");
    std::string ls = lower(storeStr);
    if (ls == "u8")        c.imageStore = ImageStore::U8;
    else if (ls == "f32")  c.imageStore = ImageStore::F32;
    else {  // auto
        // keep the fast F32 path only if it comfortably fits the budget
        long long f32Need = RUNTIME_BASELINE_MB + estF32;
        if (full || (c.ramBudgetMB > 0 && f32Need <= (long long)(c.ramBudgetMB * RAM_SAFETY)))
            c.imageStore = ImageStore::F32;
        else
            c.imageStore = ImageStore::U8;
    }
    c.estPreloadRamMB = (c.imageStore == ImageStore::U8) ? estU8 : estF32;

    // 5. max gaussians (VRAM cap). Reserve baseline + activation headroom.
    long long autoMax = 0;  // 0 == unbounded
    if (!full && c.vramBudgetMB > 0) {
        // leave 15% headroom for transient activations on top of the fixed baseline
        long long usable = (long long)((c.vramBudgetMB - VRAM_BASELINE_MB) * VRAM_HEADROOM);
        if (usable > 0)
            autoMax = (long long)(usable * 1024.0 * 1024.0 / BYTES_PER_GAUSSIAN);
    }
    c.maxGaussians = pickInt(&Overrides::maxGaussians_set, &Overrides::maxGaussians,
                             "max_gaussians", autoMax, full ? "profile" : (autoMax ? "auto" : "default"));
    if (c.maxGaussians < 0) c.maxGaussians = 0;
    c.estVramAtCapMB = c.maxGaussians > 0
        ? (long long)(c.maxGaussians * BYTES_PER_GAUSSIAN / (1024.0 * 1024.0)) + VRAM_BASELINE_MB
        : 0;

    // 6. min render px (quality guard)
    c.minRenderPx = (int)pickInt(&Overrides::minRenderPx_set, &Overrides::minRenderPx,
                                 "min_render_px", 400, "default");

    // 7. downscale factor. Explicit override wins; otherwise auto only steps it
    //    up when even U8 preload would not fit, and never past the min-render guard.
    //    (float field is handled explicitly to preserve CLI>env>JSON precedence.)
    if (cli.downscaleFactor_set)       { c.downscaleFactor = cli.downscaleFactor; c.source["downscale_factor"] = "cli"; c.downscaleFromContract = true; }
    else if (env.downscaleFactor_set)  { c.downscaleFactor = env.downscaleFactor; c.source["downscale_factor"] = "env"; c.downscaleFromContract = true; }
    else if (json.downscaleFactor_set) { c.downscaleFactor = json.downscaleFactor; c.source["downscale_factor"] = "json"; c.downscaleFromContract = true; }
    else {
        c.downscaleFactor = 1.0f;
        c.source["downscale_factor"] = "default";
        // auto-fit: only if U8 still busts the RAM budget, increase downscale by powers of 2.
        if (!full && c.ramBudgetMB > 0 && scene.repWidth > 0) {
            long long need = RUNTIME_BASELINE_MB + c.estPreloadRamMB;
            float ds = 1.0f;
            int longSide = std::max(scene.repWidth, scene.repHeight);
            while (need > (long long)(c.ramBudgetMB * RAM_SAFETY) && (longSide / (ds * 2.0f)) >= c.minRenderPx) {
                ds *= 2.0f;
                // preload scales ~1/ds^2
                need = RUNTIME_BASELINE_MB + (long long)(c.estPreloadRamMB / (ds * ds));
            }
            if (ds > 1.0f) {
                c.downscaleFactor = ds;
                c.downscaleFromContract = true;
                c.source["downscale_factor"] = "auto";
                c.estPreloadRamMB = (long long)(c.estPreloadRamMB / (ds * ds));
            }
        }
    }

    return c;
}

void printContract(const Contract &c, const HardwareInfo &hw, const SceneInfo &scene) {
    auto src = [&](const char *k) {
        auto it = c.source.find(k);
        return it == c.source.end() ? std::string("default") : it->second;
    };
    std::cout << "==================== OpenSplat resource contract ====================\n";
    std::cout << "host:   RAM " << hw.ramTotalMB << " MB total / " << hw.ramAvailMB
              << " MB avail | CPU " << hw.cpuCores << " cores | "
              << (hw.hasCuda ? "CUDA" : "CPU-only");
    if (hw.hasCuda) std::cout << " | VRAM " << hw.vramTotalMB << " MB total / " << hw.vramFreeMB << " MB free";
    std::cout << "\n";
    std::cout << "scene:  " << scene.numImages << " images @ " << scene.repWidth << "x" << scene.repHeight << "\n";
    std::cout << "profile:        " << c.profile << "  [" << src("profile") << "]\n";
    std::cout << "ram budget:     " << c.ramBudgetMB << " MB  [" << src("ram_budget_mb") << "]\n";
    std::cout << "vram budget:    " << c.vramBudgetMB << " MB  [" << src("vram_budget_mb") << "]\n";
    std::cout << "image store:    " << imageStoreName(c.imageStore) << "  [" << src("image_store")
              << "]  (est preload " << c.estPreloadRamMB << " MB)\n";
    std::cout << "max gaussians:  " << (c.maxGaussians ? std::to_string(c.maxGaussians) : std::string("unbounded"))
              << "  [" << src("max_gaussians") << "]";
    if (c.estVramAtCapMB) std::cout << "  (~" << c.estVramAtCapMB << " MB VRAM at cap)";
    std::cout << "\n";
    std::cout << "min render px:  " << c.minRenderPx << "  [" << src("min_render_px") << "]\n";
    if (c.downscaleFromContract)
        std::cout << "downscale:      " << c.downscaleFactor << "  [" << src("downscale_factor") << "]\n";
    std::cout << "=====================================================================\n";
}

} // namespace rc
