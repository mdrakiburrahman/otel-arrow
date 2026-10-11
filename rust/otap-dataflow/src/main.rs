// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    otel_arrow_dfe::run()
}
