// Unit tests for src/common/resource.* — the resource-aware process contract.
// resolvePolicy() is a PURE function, so the "intelligence" (auto-detection ->
// decisions) and the override precedence (CLI > env > JSON > profile > auto) are
// tested directly here without any hardware or training.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "resource.hpp"

using namespace rc;

// A representative "8 GB desktop with a small GPU" host.
static HardwareInfo desktop8gb() {
    HardwareInfo hw;
    hw.ramTotalMB = 8192; hw.ramAvailMB = 7000;
    hw.cpuCores = 8; hw.hasCuda = true;
    hw.vramTotalMB = 6144; hw.vramFreeMB = 5800;
    return hw;
}
// ~1 MP scene.
static SceneInfo scene(int n, int w = 1332, int h = 876) {
    SceneInfo s; s.numImages = n; s.repWidth = w; s.repHeight = h; return s;
}
static Overrides none() { return Overrides{}; }

TEST_CASE("preloadRamMB scales with images x pixels x bytes-per-sample") {
    // 100 images @ 1000x1000x3 float = 100*3e6*4 = 1.2 GB, +6% pyramid ~ 1272 MB
    long long f32 = preloadRamMB(scene(100, 1000, 1000), 4);
    long long u8 = preloadRamMB(scene(100, 1000, 1000), 1);
    CHECK(f32 == doctest::Approx(1212.0).epsilon(0.05));
    CHECK(u8 == doctest::Approx(303.0).epsilon(0.05));
    CHECK(f32 > 3 * u8);  // ~4x
}

TEST_CASE("auto profile derives RAM budget from detected hardware") {
    Contract c = resolvePolicy(desktop8gb(), scene(16), none(), none(), none());
    CHECK(c.profile == "auto");
    CHECK(c.ramBudgetMB == 7000);          // == ramAvailMB
    CHECK(c.source["ram_budget_mb"] == "auto");
    CHECK(c.minRenderPx == 400);           // quality-guard default
}

TEST_CASE("named profile presets set the RAM budget and are tagged 'profile'") {
    Overrides cli = none(); cli.profile = "4gb"; cli.profile_set = true;
    Contract c = resolvePolicy(desktop8gb(), scene(16), cli, none(), none());
    CHECK(c.ramBudgetMB == 4096);
    CHECK(c.source["profile"] == "cli");
    CHECK(c.source["ram_budget_mb"] == "profile");
}

TEST_CASE("full-throttle removes the gaussian cap and keeps the fast f32 store") {
    Overrides cli = none(); cli.profile = "full-throttle"; cli.profile_set = true;
    Contract c = resolvePolicy(desktop8gb(), scene(128), cli, none(), none());
    CHECK(c.maxGaussians == 0);                 // unbounded
    CHECK(c.imageStore == ImageStore::F32);
}

TEST_CASE("image store auto: U8 chosen when f32 preload would bust the budget") {
    // 2 GB budget, 128 big images -> f32 (~1.4GB baseline + ~2.4GB images) busts it.
    Overrides cli = none(); cli.profile = "2gb"; cli.profile_set = true;
    Contract c = resolvePolicy(desktop8gb(), scene(128, 1920, 1080), cli, none(), none());
    CHECK(c.imageStore == ImageStore::U8);
}

TEST_CASE("image store auto: F32 kept when the scene fits comfortably") {
    Contract c = resolvePolicy(desktop8gb(), scene(8), none(), none(), none());
    CHECK(c.imageStore == ImageStore::F32);     // 8 imgs ~ 113 MB, fits 7 GB
}

TEST_CASE("max gaussians derived from VRAM budget with headroom") {
    // usable = (vramFree 5800 - 600 baseline) * 0.85 = 4420 MB; /2000 B = ~2.317M
    Contract c = resolvePolicy(desktop8gb(), scene(16), none(), none(), none());
    long long usable = (long long)((5800 - VRAM_BASELINE_MB) * 0.85);  // matches code's truncation
    long long expected = (long long)(usable * 1024.0 * 1024.0 / BYTES_PER_GAUSSIAN);
    CHECK(c.maxGaussians == expected);
    CHECK(c.maxGaussians > 2000000);
    CHECK(c.maxGaussians < 3000000);
}

TEST_CASE("override precedence: CLI > env > JSON > auto") {
    Overrides cli = none(); cli.maxGaussians = 111; cli.maxGaussians_set = true;
    Overrides env = none(); env.maxGaussians = 222; env.maxGaussians_set = true;
    Overrides js  = none(); js.maxGaussians = 333; js.maxGaussians_set = true;

    CHECK(resolvePolicy(desktop8gb(), scene(8), cli, env, js).maxGaussians == 111);
    CHECK(resolvePolicy(desktop8gb(), scene(8), none(), env, js).maxGaussians == 222);
    CHECK(resolvePolicy(desktop8gb(), scene(8), none(), none(), js).maxGaussians == 333);
    CHECK(resolvePolicy(desktop8gb(), scene(8), cli, env, js).source["max_gaussians"] == "cli");
    CHECK(resolvePolicy(desktop8gb(), scene(8), none(), env, js).source["max_gaussians"] == "env");
    CHECK(resolvePolicy(desktop8gb(), scene(8), none(), none(), js).source["max_gaussians"] == "json");
}

TEST_CASE("explicit image-store override beats the auto heuristic") {
    Overrides cli = none(); cli.imageStore = "u8"; cli.imageStore_set = true;
    Contract c = resolvePolicy(desktop8gb(), scene(8), cli, none(), none());  // would auto-pick f32
    CHECK(c.imageStore == ImageStore::U8);
    CHECK(c.source["image_store"] == "cli");
}

TEST_CASE("auto-downscale fits a huge scene into a tiny budget, clamped by min-render-px") {
    HardwareInfo hw; hw.ramTotalMB = 2048; hw.ramAvailMB = 2048; hw.cpuCores = 4;
    hw.hasCuda = false;
    // 400 images @ 4946x3286 (Mip-NeRF bicycle class): even u8 is enormous.
    Contract c = resolvePolicy(hw, scene(400, 4946, 3286), none(), none(), none());
    CHECK(c.downscaleFromContract);
    CHECK(c.downscaleFactor > 1.0f);
    // never downscale past the quality guard: render long side stays >= min_render_px
    int renderLong = (int)(4946 / c.downscaleFactor);
    CHECK(renderLong >= c.minRenderPx);
}

TEST_CASE("min-render-px is overridable") {
    Overrides cli = none(); cli.minRenderPx = 256; cli.minRenderPx_set = true;
    Contract c = resolvePolicy(desktop8gb(), scene(8), cli, none(), none());
    CHECK(c.minRenderPx == 256);
    CHECK(c.source["min_render_px"] == "cli");
}

TEST_CASE("JSON contract parses every field") {
    std::string err;
    Overrides o = overridesFromJsonString(R"({
        "profile": "6gb",
        "ram_budget_mb": 6000,
        "vram_budget_mb": 4000,
        "max_gaussians": 500000,
        "image_store": "u8",
        "min_render_px": 320,
        "downscale_factor": 2
    })", &err);
    CHECK(err.empty());
    CHECK(o.profile_set);          CHECK(o.profile == "6gb");
    CHECK(o.ramBudgetMB_set);      CHECK(o.ramBudgetMB == 6000);
    CHECK(o.vramBudgetMB_set);     CHECK(o.vramBudgetMB == 4000);
    CHECK(o.maxGaussians_set);     CHECK(o.maxGaussians == 500000);
    CHECK(o.imageStore_set);       CHECK(o.imageStore == "u8");
    CHECK(o.minRenderPx_set);      CHECK(o.minRenderPx == 320);
    CHECK(o.downscaleFactor_set);  CHECK(o.downscaleFactor == doctest::Approx(2.0f));
}

TEST_CASE("malformed JSON reports an error and yields no overrides") {
    std::string err;
    Overrides o = overridesFromJsonString("{ not json ", &err);
    CHECK_FALSE(err.empty());
    CHECK_FALSE(o.profile_set);
    CHECK_FALSE(o.maxGaussians_set);
}

TEST_CASE("a partial JSON contract only sets the fields it lists") {
    std::string err;
    Overrides o = overridesFromJsonString(R"({"max_gaussians": 250000})", &err);
    CHECK(err.empty());
    CHECK(o.maxGaussians_set);   CHECK(o.maxGaussians == 250000);
    CHECK_FALSE(o.profile_set);  CHECK_FALSE(o.imageStore_set);
}
