# Regression tests

One test per fixed bug. The rule: **a bug fix is not done until a test exists that fails on the
old code and passes on the fix.** Name tests after the issue/symptom so the history is legible.

When adding a regression test:
1. Reproduce the bug as a minimal failing `doctest` case here.
2. Confirm it fails on the unfixed code.
3. Apply the fix; confirm it passes.
4. Wire the file into `../CMakeLists.txt` and reference the bug in a comment.

See [`../../docs/testing.md`](../../docs/testing.md) for the broader methodology.
