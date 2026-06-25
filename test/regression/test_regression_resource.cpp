// Regression / characterization tests for the resource-aware process contract.
//
// These lock in the CURRENT, measurement-calibrated behaviour of resolvePolicy()
// and its constants. If one fails, do NOT just update the number — confirm the
// change is intentional (re-measure on the reference box; see linux_research/)
// and note why here. See ../README.md.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "resource.hpp"

using namespace rc;

// Reference desktop-class hosts for the four simulated RAM budgets. VRAM is set
// to a plausible paired consumer GPU so the derived gaussian caps are realistic.
static HardwareInfo host(long long ramMB, long long vramMB) {
    HardwareInfo hw;
    hw.ramTotalMB = ramMB; hw.ramAvailMB = ramMB;
    hw.cpuCores = 4; hw.hasCuda = true;
    hw.vramTotalMB = vramMB; hw.vramFreeMB = vramMB;
    return hw;
}
static SceneInfo bigScene() { SceneInfo s; s.numImages = 128; s.repWidth = 1920; s.repHeight = 1080; return s; }

TEST_CASE("REG: calibrated constants are pinned (re-measure before changing)") {
    // Calibrated on A100-80GB / drjohnson @1332x876 — see linux_research/measurements.
    CHECK(BYTES_PER_GAUSSIAN == doctest::Approx(2000.0));
    CHECK(RUNTIME_BASELINE_MB == 1400);
    CHECK(VRAM_BASELINE_MB == 600);
}

TEST_CASE("REG: the four desktop RAM profiles pick the documented image store") {
    auto prof = [](const char *p) { Overrides o; o.profile = p; o.profile_set = true; return o; };
    // A 128x(1920x1080) scene: f32 preload ~ 3220 MB images (+1400 baseline).
    // 2gb/4gb budgets force u8; 6gb/8gb keep the fast f32 path
    // (u8 preload ~805 MB + 1400 baseline = 2205 MB fits within 0.85*6144).
    CHECK(resolvePolicy(host(2048, 4096), bigScene(), prof("2gb"), Overrides{}, Overrides{}).imageStore == ImageStore::U8);
    CHECK(resolvePolicy(host(4096, 4096), bigScene(), prof("4gb"), Overrides{}, Overrides{}).imageStore == ImageStore::U8);
    CHECK(resolvePolicy(host(6144, 6144), bigScene(), prof("6gb"), Overrides{}, Overrides{}).imageStore == ImageStore::F32);
    CHECK(resolvePolicy(host(8192, 8192), bigScene(), prof("8gb"), Overrides{}, Overrides{}).imageStore == ImageStore::F32);
}

TEST_CASE("REG: gaussian cap formula == (vram - baseline)*0.85 / bytes-per-gaussian") {
    // Mirror the code exactly: usable MB is truncated to an integer before the
    // bytes-per-gaussian division (6 GB GPU -> usable 4712 MB -> ~2.47M gaussians).
    Contract c = resolvePolicy(host(8192, 6144), bigScene(), Overrides{}, Overrides{}, Overrides{});
    long long usable = (long long)((6144 - VRAM_BASELINE_MB) * 0.85);
    long long expected = (long long)(usable * 1024.0 * 1024.0 / BYTES_PER_GAUSSIAN);
    CHECK(c.maxGaussians == expected);
    CHECK(c.maxGaussians > 2400000);
    CHECK(c.maxGaussians < 2500000);
}

TEST_CASE("REG: profile names are case-insensitive") {
    Overrides o; o.profile = "Full-Throttle"; o.profile_set = true;
    CHECK(resolvePolicy(host(8192, 8192), bigScene(), o, Overrides{}, Overrides{}).maxGaussians == 0);
    Overrides o2; o2.profile = "4GB"; o2.profile_set = true;
    CHECK(resolvePolicy(host(8192, 8192), bigScene(), o2, Overrides{}, Overrides{}).ramBudgetMB == 4096);
}

TEST_CASE("REG: env var overrides parse with OPENSPLAT_ prefix") {
    setenv("OPENSPLAT_MAX_SPLATS", "750000", 1);
    setenv("OPENSPLAT_IMAGE_STORE", "u8", 1);
    Overrides e = overridesFromEnv();
    CHECK(e.maxGaussians_set); CHECK(e.maxGaussians == 750000);
    CHECK(e.imageStore_set);   CHECK(e.imageStore == "u8");
    unsetenv("OPENSPLAT_MAX_SPLATS");
    unsetenv("OPENSPLAT_IMAGE_STORE");
}
