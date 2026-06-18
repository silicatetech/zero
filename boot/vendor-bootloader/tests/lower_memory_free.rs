// SPDX-License-Identifier: AGPL-3.0-or-later
use bootloader_test_runner::run_test_kernel;
#[test]
fn lower_memory_free() {
    run_test_kernel(env!(
        "CARGO_BIN_FILE_TEST_KERNEL_LOWER_MEMORY_FREE_lower_memory_free"
    ));
}
