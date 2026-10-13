// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Verifies that downstream crates can register codecs through the public macro.

use bytes::Bytes;
use otel_arrow_dfe_config::SignalType;
use otel_arrow_dfe_pdata::{
    OtapArrowRecords, PdataArrowRecordSet, PdataRepresentation, PdataRepresentationId,
    RepresentationVersion,
};
use otel_arrow_dfe_pdata_codec::{
    ArrowCodecMetadata, ArrowCodecRegistration, CodecError, CodecMetadata, CodecRegistration,
    CodecRegistry, PdataArrowDecoder, PdataDecoder, PdataEncoding, register_pdata_arrow_codec,
    register_pdata_codec,
};

const EXTERNAL_ENCODING: PdataEncoding = PdataEncoding::new("external-fixture-v1-zstd");

static EXTERNAL_METADATA: CodecMetadata =
    CodecMetadata::new(EXTERNAL_ENCODING, &[SignalType::Logs])
        .with_format_version("1")
        .with_representation_version(RepresentationVersion::new(1, 0))
        .with_compression("zstd");

struct ExternalDecoder;

impl PdataDecoder for ExternalDecoder {
    fn decode(
        &mut self,
        _signal: SignalType,
        _bytes: &Bytes,
    ) -> Result<OtapArrowRecords, CodecError> {
        panic!("the registration hygiene test does not decode payloads")
    }
}

register_pdata_codec!(
    EXTERNAL_CODEC,
    CodecRegistration::new(&EXTERNAL_METADATA)
        .with_decoder(|_| Box::new(ExternalDecoder))
        .with_item_counter(|_, _| Some(0)),
);

static EXTERNAL_ARROW_METADATA: ArrowCodecMetadata = ArrowCodecMetadata::new(
    PdataRepresentation::new(
        PdataRepresentationId::new("external-arrow"),
        RepresentationVersion::new(1, 0),
    ),
    &[SignalType::Logs],
);

struct ExternalArrowDecoder;

impl PdataArrowDecoder for ExternalArrowDecoder {
    fn decode(&mut self, _records: &PdataArrowRecordSet) -> Result<OtapArrowRecords, CodecError> {
        panic!("the registration hygiene test does not decode payloads")
    }
}

register_pdata_arrow_codec!(
    EXTERNAL_ARROW_CODEC,
    ArrowCodecRegistration::new(&EXTERNAL_ARROW_METADATA)
        .with_decoder(|| Box::new(ExternalArrowDecoder)),
);

/// Scenario: A downstream crate registers byte and Arrow codecs without importing linkme.
/// Guarantees: Both public macros are hygienic and preserve identities and capabilities.
#[test]
fn downstream_registration_needs_no_linkme_import() {
    let registry = CodecRegistry::global().expect("linked codec registrations must be valid");
    let codec = registry
        .resolve(&EXTERNAL_ENCODING)
        .expect("the macro must add the downstream codec to the registry");

    assert!(codec.can_decode());
    assert!(!codec.can_encode());
    assert_eq!(codec.count_items(SignalType::Logs, &[]), Some(0));
    assert_eq!(codec.metadata().encoding(), &EXTERNAL_ENCODING);
    assert_eq!(codec.metadata().signals(), &[SignalType::Logs]);
    assert_eq!(codec.metadata().format_version(), Some("1"));
    assert_eq!(
        codec.metadata().representation_version(),
        Some(RepresentationVersion::new(1, 0))
    );
    assert_eq!(codec.metadata().compression(), Some("zstd"));

    let arrow_codec = registry
        .resolve_arrow(EXTERNAL_ARROW_METADATA.representation())
        .expect("the macro must add the downstream Arrow codec to the registry");
    assert!(arrow_codec.capabilities().can_pass_through());
    assert!(arrow_codec.capabilities().can_decode_to_otap());
    assert!(!arrow_codec.capabilities().can_encode_from_otap());
    assert_eq!(
        arrow_codec.metadata().representation(),
        EXTERNAL_ARROW_METADATA.representation()
    );
}
