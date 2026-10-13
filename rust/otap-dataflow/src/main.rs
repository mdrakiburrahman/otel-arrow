// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// These checks belong to the executable. Its library is also linked without
// cfg(test) by test binaries that intentionally enable every Cargo feature.
#[cfg(all(
    not(any(test, doc)),
    not(clippy),
    any(
        all(feature = "dhat-heap", feature = "mimalloc"),
        all(feature = "dhat-heap", feature = "jemalloc"),
        all(feature = "jemalloc", feature = "mimalloc"),
    )
))]
compile_error!(
    "Allocator features are mutually exclusive. Enable only one allocator: `dhat-heap`, `mimalloc`, `jemalloc`. \
    Example: \
        (mimalloc): cargo build --release --no-default-features --features mimalloc. \
        (jemalloc): cargo build --release --no-default-features --features jemalloc. \
        (dhat): cargo build --profile profiling --no-default-features --features dhat-heap."
);

#[cfg(all(not(any(test, doc)), not(clippy)))]
const _: () = {
    const SYMCRYPT: u8 = if cfg!(feature = "crypto-symcrypt") {
        1
    } else {
        0
    };
    const OPENSSL: u8 = if cfg!(feature = "crypto-openssl") {
        1
    } else {
        0
    };
    const AWS_LC: u8 = if cfg!(feature = "crypto-aws-lc") {
        1
    } else {
        0
    };
    const RING: u8 = if cfg!(feature = "crypto-ring") { 1 } else { 0 };
    const COUNT: u8 = SYMCRYPT + OPENSSL + AWS_LC + RING;
    if COUNT > 1 {
        panic!(
            "Crypto provider features are mutually exclusive. \
             Enable exactly one of: crypto-symcrypt, crypto-openssl, crypto-aws-lc, crypto-ring. \
             Use --no-default-features to disable the default crypto provider, then enable exactly one."
        );
    }
};

#[cfg(all(
    feature = "crypto-symcrypt",
    not(any(target_os = "linux", target_os = "windows")),
    not(any(test, doc)),
    not(clippy)
))]
compile_error!(
    "Feature `crypto-symcrypt` is only supported on Linux and Windows targets. \
     Use a different crypto provider on this platform (e.g., crypto-ring)."
);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    otel_arrow_dfe::run()
}
