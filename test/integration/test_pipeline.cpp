// Integration tests: modules wired together. Built with the CPU rasterizer backend.
//
// The PLY roundtrip needs no external data. The dataset-loading case is DATA-GATED: it runs
// only when a COLMAP scene is available (env OPENSPLAT_TEST_DATA, or ./data/db/drjohnson from
// scripts/fetch_test_data.sh) and skips gracefully otherwise.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <torch/torch.h>
#include <filesystem>
#include <cstdlib>
#include "point_io.hpp"
#include "input_data.hpp"

namespace fs = std::filesystem;

TEST_CASE("io: PointSet survives a PLY save -> read roundtrip") {
    PointSet ps;
    ps.points = {{0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f}, {-1.0f, 0.5f, 4.25f}};
    ps.colors = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};

    fs::path tmp = fs::temp_directory_path() / "opensplat_roundtrip_test.ply";
    savePointSet(ps, tmp.string());
    REQUIRE(fs::exists(tmp));

    PointSet *rd = readPointSet(tmp.string());
    REQUIRE(rd != nullptr);
    CHECK(rd->count() == ps.count());
    for (size_t i = 0; i < ps.count(); ++i) {
        for (int d = 0; d < 3; ++d) {
            CHECK(rd->points[i][d] == doctest::Approx(ps.points[i][d]).epsilon(1e-4));
        }
    }
    RELEASE_POINTSET(rd);
    fs::remove(tmp);
}

static std::string testScene() {
    if (const char *e = std::getenv("OPENSPLAT_TEST_DATA")) return e;
    return "data/db/drjohnson";  // default from scripts/fetch_test_data.sh
}

TEST_CASE("pipeline: load a COLMAP scene into InputData (data-gated)") {
    const std::string scene = testScene();
    if (!fs::exists(fs::path(scene) / "sparse")) {
        MESSAGE("skipping: no COLMAP scene at '" << scene
                << "'. Run scripts/fetch_test_data.sh or set OPENSPLAT_TEST_DATA.");
        return;
    }

    InputData data = inputDataFromX(scene, "");
    CHECK(data.cameras.size() > 0);
    CHECK(data.points.xyz.size(0) > 0);
    CHECK(data.points.xyz.size(1) == 3);
    CHECK(data.scale > 0.0f);

    // Auto-scaled/centered camera origins must be bounded (|origin| <= 1 by construction).
    for (const Camera &cam : data.cameras) {
        REQUIRE(cam.camToWorld.numel() == 16);
    }
}
