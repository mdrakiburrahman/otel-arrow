// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Link-time codec discovery and immutable registry validation.
//!
//! Codec extension crates submit [`CodecRegistration`] values through
//! [`crate::register_pdata_codec!`]. A registration contains immutable metadata
//! and factories only; mutable decoder and encoder instances belong to the
//! pipeline-local runtime described by [`crate::CodecService`]. Registrations
//! may be one-sided, so a format can support decoding without encoding or the
//! reverse. Decoder registrations must also provide stateless item counting so
//! an operator request for item-count telemetry cannot be silently ignored.
//!
//! [`CodecRegistry`] validates the complete linked set before a pipeline uses
//! it. Encoding names identify byte-compatible representations, and duplicate
//! or invalid names are rejected independently of linker order. Once validated,
//! [`ResolvedCodec`] is a cheap handle used for admission, capability checks,
//! stateless item counting, and lazy factory access.

use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock};

use bytes::Bytes;
use otel_arrow_dfe_config::SignalType;
use otel_arrow_dfe_pdata::{PdataRepresentation, PdataRepresentationId, RepresentationVersion};

use crate::{
    ArrowCodecRegistration, CodecError, CodecOperation, DecodePolicy, EncodePolicy, EncodedPdata,
    PDATA_ARROW_CODEC_FACTORIES, PdataDecoder, PdataEncoder, PdataEncoding, RegistryError,
    ResolvedArrowCodec,
};

/// Conversion directions implemented by a registered representation codec.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CodecCapabilities {
    pass_through: bool,
    decode_to_otap: bool,
    encode_from_otap: bool,
}

impl CodecCapabilities {
    pub(crate) const fn new(decode_to_otap: bool, encode_from_otap: bool) -> Self {
        Self {
            pass_through: true,
            decode_to_otap,
            encode_from_otap,
        }
    }

    /// Whether the representation can be preserved without conversion.
    #[must_use]
    pub const fn can_pass_through(self) -> bool {
        self.pass_through
    }

    /// Whether the codec can convert its representation to native OTAP.
    #[must_use]
    pub const fn can_decode_to_otap(self) -> bool {
        self.decode_to_otap
    }

    /// Whether the codec can convert native OTAP to its representation.
    #[must_use]
    pub const fn can_encode_from_otap(self) -> bool {
        self.encode_from_otap
    }
}

/// Stateless item scan that needs no mutable codec instance.
///
/// Decoder registrations must provide this capability. `None` means the bytes
/// are malformed or otherwise cannot be counted; it is distinct from a valid
/// empty batch.
pub type ItemCounter = fn(SignalType, &[u8]) -> Option<usize>;

/// Creates independent decoder state for one pipeline runtime.
///
/// The policy is resolved once for the pipeline before the decoder is created.
/// `BestEffort` permits a codec-specific relaxed path; `Strict` requires full
/// malformed-input detection. Codecs may return the same strict decoder for
/// both modes.
pub type DecoderFactory = fn(DecodePolicy) -> Box<dyn PdataDecoder>;

/// Creates encoder state configured once for one output plan.
pub type EncoderFactory = fn(EncodePolicy) -> Result<Box<dyn PdataEncoder>, CodecError>;

/// Immutable metadata for one encoded representation.
#[derive(Debug)]
pub struct CodecMetadata {
    /// Globally stable representation name.
    encoding: PdataEncoding,
    /// Signals understood by this codec.
    signals: &'static [SignalType],
    /// Informational version. Incompatible versions require different identities.
    format_version: Option<&'static str>,
    /// Structured identity version used by the generalized representation model.
    representation_version: Option<RepresentationVersion>,
    /// Compression intrinsic to the representation, if any.
    compression: Option<&'static str>,
}

impl CodecMetadata {
    /// Declares an encoded representation and the signals it supports.
    #[must_use]
    pub const fn new(encoding: PdataEncoding, signals: &'static [SignalType]) -> Self {
        Self {
            encoding,
            signals,
            format_version: None,
            representation_version: None,
            compression: None,
        }
    }

    /// Adds an informational version for diagnostics and discovery.
    ///
    /// A version that changes byte compatibility belongs in [`PdataEncoding`]
    /// instead so incompatible formats cannot resolve to the same codec.
    #[must_use]
    pub const fn with_format_version(mut self, format_version: &'static str) -> Self {
        self.format_version = Some(format_version);
        self
    }

    /// Adds the structured representation version used by capability discovery.
    #[must_use]
    pub const fn with_representation_version(
        mut self,
        representation_version: RepresentationVersion,
    ) -> Self {
        self.representation_version = Some(representation_version);
        self
    }

    /// Declares compression intrinsic to the encoded representation.
    ///
    /// Transport compression, such as HTTP gzip, does not belong here.
    #[must_use]
    pub const fn with_compression(mut self, compression: &'static str) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Returns the stable representation identity.
    #[must_use]
    pub const fn encoding(&self) -> &PdataEncoding {
        &self.encoding
    }

    /// Returns the telemetry signals accepted by this representation.
    #[must_use]
    pub const fn signals(&self) -> &'static [SignalType] {
        self.signals
    }

    /// Returns the informational format version, when supplied.
    #[must_use]
    pub const fn format_version(&self) -> Option<&'static str> {
        self.format_version
    }

    /// Returns the structured representation version, when supplied.
    #[must_use]
    pub const fn representation_version(&self) -> Option<RepresentationVersion> {
        self.representation_version
    }

    /// Returns the generalized representation identity, when versioned metadata is supplied.
    #[must_use]
    pub fn representation(&self) -> Option<PdataRepresentation> {
        self.representation_version.map(|version| {
            PdataRepresentation::new(
                PdataRepresentationId::from(self.encoding.as_str().to_owned()),
                version,
            )
        })
    }

    /// Returns compression intrinsic to the representation, when supplied.
    #[must_use]
    pub const fn compression(&self) -> Option<&'static str> {
        self.compression
    }
}

/// Link-time codec registration containing only immutable factories.
#[derive(Debug)]
pub struct CodecRegistration {
    /// Representation identity and supported signals.
    metadata: &'static CodecMetadata,
    /// Decoder factory, absent for encode-only formats.
    decoder: Option<DecoderFactory>,
    /// Encoder factory, absent for decode-only formats.
    encoder: Option<EncoderFactory>,
    /// Stateless item scan required when a decoder is registered.
    count_items: Option<ItemCounter>,
}

impl CodecRegistration {
    /// Starts a registration with no decode or encode capabilities.
    ///
    /// Add at least one direction before submitting the registration to the
    /// registry. Registry validation rejects registrations with no factories.
    #[must_use]
    pub const fn new(metadata: &'static CodecMetadata) -> Self {
        Self {
            metadata,
            decoder: None,
            encoder: None,
            count_items: None,
        }
    }

    /// Adds a factory for pipeline-local decoder state.
    #[must_use]
    pub const fn with_decoder(mut self, decoder: DecoderFactory) -> Self {
        self.decoder = Some(decoder);
        self
    }

    /// Adds a factory for pipeline-local encoder state.
    #[must_use]
    pub const fn with_encoder(mut self, encoder: EncoderFactory) -> Self {
        self.encoder = Some(encoder);
        self
    }

    /// Adds a stateless item counter that needs no codec instance.
    ///
    /// Every decoder registration must provide this capability. Registry
    /// validation rejects a decoder without it.
    #[must_use]
    pub const fn with_item_counter(mut self, count_items: ItemCounter) -> Self {
        self.count_items = Some(count_items);
        self
    }
}

/// Trusted codec extensions compiled into this binary.
#[allow(unsafe_code)]
#[linkme::distributed_slice]
pub static PDATA_CODEC_FACTORIES: [CodecRegistration];

/// One immutable registration from a validated registry.
#[derive(Clone, Copy, Debug)]
pub struct ResolvedCodec(&'static CodecRegistration);

impl PartialEq for ResolvedCodec {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl Eq for ResolvedCodec {}

impl Hash for ResolvedCodec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::from_ref(self.0).hash(state);
    }
}

impl ResolvedCodec {
    /// Immutable metadata and canonical encoding name.
    #[must_use]
    pub const fn metadata(self) -> &'static CodecMetadata {
        self.0.metadata
    }

    /// Canonical encoding identity.
    #[must_use]
    pub const fn encoding(self) -> &'static PdataEncoding {
        &self.0.metadata.encoding
    }

    /// Whether a decoder factory is registered.
    #[must_use]
    pub const fn can_decode(self) -> bool {
        self.0.decoder.is_some()
    }

    /// Whether an encoder factory is registered.
    #[must_use]
    pub const fn can_encode(self) -> bool {
        self.0.encoder.is_some()
    }

    /// Registered conversion capabilities.
    #[must_use]
    pub const fn capabilities(self) -> CodecCapabilities {
        CodecCapabilities::new(self.can_decode(), self.can_encode())
    }

    /// Validates decoder support for a signal.
    pub fn require_decoder(self, signal: SignalType) -> Result<(), CodecError> {
        self.require(signal, CodecOperation::Decode, self.can_decode())
    }

    /// Validates encoder support for a signal.
    pub fn require_encoder(self, signal: SignalType) -> Result<(), CodecError> {
        self.require(signal, CodecOperation::Encode, self.can_encode())
    }

    fn require(
        self,
        signal: SignalType,
        operation: CodecOperation,
        factory_present: bool,
    ) -> Result<(), CodecError> {
        if factory_present && self.metadata().signals().contains(&signal) {
            Ok(())
        } else {
            Err(CodecError::Unsupported {
                encoding: self.encoding().clone(),
                operation,
                signal,
            })
        }
    }

    /// Counts items without instantiating mutable codec state when supported.
    #[must_use]
    pub fn count_items(self, signal: SignalType, bytes: &[u8]) -> Option<usize> {
        self.0
            .count_items
            .and_then(|counter| counter(signal, bytes))
    }

    /// Admits supported input without decoding or creating mutable codec state.
    pub fn admit(self, signal: SignalType, bytes: Bytes) -> Result<EncodedPdata, CodecError> {
        self.require_decoder(signal)?;
        Ok(EncodedPdata::from_resolved(self, signal, bytes))
    }

    pub(crate) fn create_decoder(
        self,
        policy: DecodePolicy,
    ) -> Result<Box<dyn PdataDecoder>, CodecError> {
        self.0
            .decoder
            .map(|factory| factory(policy))
            .ok_or_else(|| CodecError::Unsupported {
                encoding: self.encoding().clone(),
                operation: CodecOperation::Decode,
                signal: self.metadata().signals()[0],
            })
    }

    pub(crate) fn create_encoder(
        self,
        policy: EncodePolicy,
    ) -> Result<Box<dyn PdataEncoder>, CodecError> {
        self.0.encoder.ok_or_else(|| CodecError::Unsupported {
            encoding: self.encoding().clone(),
            operation: CodecOperation::Encode,
            signal: self.metadata().signals()[0],
        })?(policy)
    }
}

/// Validated immutable view of all codec extensions linked into a binary.
#[derive(Debug)]
pub struct CodecRegistry {
    codecs: Box<[ResolvedCodec]>,
    arrow_codecs: Box<[ResolvedArrowCodec]>,
}

static GLOBAL_REGISTRY: LazyLock<Result<Arc<CodecRegistry>, RegistryError>> = LazyLock::new(|| {
    CodecRegistry::validate_with_arrow(&PDATA_CODEC_FACTORIES, &PDATA_ARROW_CODEC_FACTORIES)
        .map(Arc::new)
});

impl CodecRegistry {
    /// Returns the process registry after validating all linked registrations.
    pub fn global() -> Result<Arc<Self>, RegistryError> {
        GLOBAL_REGISTRY
            .as_ref()
            .map(Arc::clone)
            .map_err(Clone::clone)
    }

    /// Validates an explicit registration set.
    pub fn validate(factories: &'static [CodecRegistration]) -> Result<Self, RegistryError> {
        Self::validate_with_arrow(factories, &[])
    }

    /// Validates explicit byte and Arrow registration sets.
    pub fn validate_with_arrow(
        factories: &'static [CodecRegistration],
        arrow_factories: &'static [ArrowCodecRegistration],
    ) -> Result<Self, RegistryError> {
        for (index, registration) in factories.iter().enumerate() {
            validate_registration(registration)?;
            if factories[..index]
                .iter()
                .any(|other| other.metadata.encoding == registration.metadata.encoding)
            {
                return Err(RegistryError::Duplicate {
                    encoding: registration.metadata.encoding.clone(),
                });
            }
        }
        for (index, registration) in arrow_factories.iter().enumerate() {
            validate_arrow_registration(registration)?;
            if arrow_factories[..index].iter().any(|other| {
                other.metadata.representation() == registration.metadata.representation()
            }) {
                return Err(RegistryError::DuplicateArrow {
                    representation: registration.metadata.representation().clone(),
                });
            }
        }
        Ok(Self {
            codecs: factories.iter().map(ResolvedCodec).collect(),
            arrow_codecs: arrow_factories.iter().map(ResolvedArrowCodec).collect(),
        })
    }

    /// Resolves a unique registered identity.
    pub fn resolve(&self, encoding: &PdataEncoding) -> Result<ResolvedCodec, RegistryError> {
        self.codecs
            .iter()
            .copied()
            .find(|codec| codec.encoding() == encoding)
            .ok_or_else(|| RegistryError::NotFound {
                encoding: encoding.clone(),
            })
    }

    /// Resolves and validates decoder support.
    pub fn resolve_decoder(
        &self,
        encoding: &PdataEncoding,
        signal: SignalType,
    ) -> Result<ResolvedCodec, CodecError> {
        let codec = self.resolve(encoding)?;
        codec.require_decoder(signal)?;
        Ok(codec)
    }

    /// Resolves and validates encoder support.
    pub fn resolve_encoder(
        &self,
        encoding: &PdataEncoding,
        signal: SignalType,
    ) -> Result<ResolvedCodec, CodecError> {
        let codec = self.resolve(encoding)?;
        codec.require_encoder(signal)?;
        Ok(codec)
    }

    /// Resolves a unique registered pluggable Arrow representation.
    pub fn resolve_arrow(
        &self,
        representation: &PdataRepresentation,
    ) -> Result<ResolvedArrowCodec, RegistryError> {
        self.arrow_codecs
            .iter()
            .copied()
            .find(|codec| codec.representation() == representation)
            .ok_or_else(|| RegistryError::ArrowNotFound {
                representation: representation.clone(),
            })
    }

    /// Resolves and validates Arrow pass-through support.
    pub fn resolve_arrow_passthrough(
        &self,
        representation: &PdataRepresentation,
        signal: SignalType,
    ) -> Result<ResolvedArrowCodec, CodecError> {
        let codec = self.resolve_arrow(representation)?;
        codec.require_passthrough(signal)?;
        Ok(codec)
    }

    /// Resolves and validates Arrow decoder support.
    pub fn resolve_arrow_decoder(
        &self,
        representation: &PdataRepresentation,
        signal: SignalType,
    ) -> Result<ResolvedArrowCodec, CodecError> {
        let codec = self.resolve_arrow(representation)?;
        codec.require_decoder(signal)?;
        Ok(codec)
    }

    /// Resolves and validates Arrow encoder support.
    pub fn resolve_arrow_encoder(
        &self,
        representation: &PdataRepresentation,
        signal: SignalType,
    ) -> Result<ResolvedArrowCodec, CodecError> {
        let codec = self.resolve_arrow(representation)?;
        codec.require_encoder(signal)?;
        Ok(codec)
    }

    /// Iterates all validated codec identities without creating mutable state.
    #[must_use]
    pub fn codecs(&self) -> impl ExactSizeIterator<Item = ResolvedCodec> + '_ {
        self.codecs.iter().copied()
    }

    /// Iterates all validated Arrow representation identities.
    #[must_use]
    pub fn arrow_codecs(&self) -> impl ExactSizeIterator<Item = ResolvedArrowCodec> + '_ {
        self.arrow_codecs.iter().copied()
    }
}

fn validate_registration(registration: &CodecRegistration) -> Result<(), RegistryError> {
    let metadata = registration.metadata;
    let encoding = &metadata.encoding;
    let name = encoding.as_str();
    if ["otap", "otlp", "preserve"].contains(&name) {
        return Err(RegistryError::InvalidIdentity {
            encoding: encoding.clone(),
            reason: "reserved format name",
        });
    }
    if name.is_empty()
        || !name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-:".contains(&byte)
        })
    {
        return Err(RegistryError::InvalidIdentity {
            encoding: encoding.clone(),
            reason: "identity must use lowercase ASCII letters, digits, '.', '_', '-', or ':'",
        });
    }
    if metadata.signals.is_empty() {
        return Err(RegistryError::EmptySignals {
            encoding: encoding.clone(),
        });
    }
    if registration.decoder.is_none() && registration.encoder.is_none() {
        return Err(RegistryError::EmptyCapabilities {
            encoding: encoding.clone(),
        });
    }
    if registration.decoder.is_some() && registration.count_items.is_none() {
        return Err(RegistryError::MissingItemCounter {
            encoding: encoding.clone(),
        });
    }
    Ok(())
}

fn validate_arrow_registration(registration: &ArrowCodecRegistration) -> Result<(), RegistryError> {
    let representation = registration.metadata.representation();
    let name = representation.id().as_str();
    if name.is_empty()
        || !name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-:".contains(&byte)
        })
    {
        return Err(RegistryError::InvalidArrowIdentity {
            representation: representation.clone(),
            reason: "identity must use lowercase ASCII letters, digits, '.', '_', '-', or ':'",
        });
    }
    if registration.metadata.signals().is_empty() {
        return Err(RegistryError::EmptyArrowSignals {
            representation: representation.clone(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArrowCodecMetadata;
    use otel_arrow_dfe_pdata::{PdataRepresentationId, RepresentationVersion};

    static EMPTY_METADATA: CodecMetadata =
        CodecMetadata::new(PdataEncoding::new("empty-v1"), &[SignalType::Logs])
            .with_format_version("1");
    static EMPTY_REGISTRATIONS: [CodecRegistration; 1] = [CodecRegistration::new(&EMPTY_METADATA)];

    static DECODER_WITHOUT_COUNTER_REGISTRATIONS: [CodecRegistration; 1] =
        [CodecRegistration::new(&EMPTY_METADATA).with_decoder(|_| Box::new(TestDecoder))];
    static NO_BYTE_REGISTRATIONS: [CodecRegistration; 0] = [];
    static PASSTHROUGH_ARROW_METADATA: ArrowCodecMetadata = ArrowCodecMetadata::new(
        PdataRepresentation::new(
            PdataRepresentationId::new("passthrough-arrow"),
            RepresentationVersion::new(1, 0),
        ),
        &[SignalType::Logs],
    );
    static PASSTHROUGH_ARROW_REGISTRATIONS: [ArrowCodecRegistration; 1] =
        [ArrowCodecRegistration::new(&PASSTHROUGH_ARROW_METADATA)];

    struct TestDecoder;

    impl PdataDecoder for TestDecoder {
        fn decode(
            &mut self,
            _signal: SignalType,
            _bytes: &Bytes,
        ) -> Result<otel_arrow_dfe_pdata::OtapArrowRecords, CodecError> {
            unreachable!("registry validation test never invokes the decoder")
        }
    }

    /// Scenario: A linked extension declares no decoder or encoder factory.
    /// Guarantees: Production registry construction rejects empty capabilities.
    #[test]
    fn rejects_empty_capabilities() {
        assert!(matches!(
            CodecRegistry::validate(&EMPTY_REGISTRATIONS),
            Err(RegistryError::EmptyCapabilities { .. })
        ));
    }

    /// Scenario: A linked extension registers a decoder without an item counter.
    /// Guarantees: Registry validation rejects a codec that could silently omit requested metrics.
    #[test]
    fn rejects_decoder_without_item_counter() {
        assert!(matches!(
            CodecRegistry::validate(&DECODER_WITHOUT_COUNTER_REGISTRATIONS),
            Err(RegistryError::MissingItemCounter { .. })
        ));
    }

    /// Scenario: An Arrow representation is registered only for in-memory pass-through.
    /// Guarantees: Registry validation accepts it while reporting both OTAP conversion directions as unavailable.
    #[test]
    fn accepts_pass_through_only_arrow_registration() {
        let registry = CodecRegistry::validate_with_arrow(
            &NO_BYTE_REGISTRATIONS,
            &PASSTHROUGH_ARROW_REGISTRATIONS,
        )
        .expect("pass-through-only Arrow registration is valid");
        let codec = registry
            .resolve_arrow(PASSTHROUGH_ARROW_METADATA.representation())
            .expect("registered Arrow representation");

        assert!(codec.capabilities().can_pass_through());
        assert!(!codec.capabilities().can_decode_to_otap());
        assert!(!codec.capabilities().can_encode_from_otap());
    }
}
