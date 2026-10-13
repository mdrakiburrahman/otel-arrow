// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline-local lifecycle and access for mutable codec implementations.
//!
//! [`CodecServiceBuilder`] combines an immutable validated registry and decode
//! policy with fresh runtime state. Cloned [`CodecService`] handles share that
//! state within one pipeline, while decoder and encoder instances are created
//! lazily and reused. The decode policy is applied once at decoder creation;
//! payload admission and matching-format forwarding therefore require neither
//! codec construction nor mutable runtime access.
//!
//! Codec trait calls are currently synchronous. The service holds its runtime
//! lock for the duration of each codec operation and, for prepared output, for
//! the synchronous consumer callback. Nested access from that callback returns
//! an error instead of blocking on the same lock. A panic while state is
//! borrowed poisons the service; later operations report that state as
//! unavailable instead of continuing with possibly corrupted codec state. No
//! codec borrow crosses an async suspension. [`EncodeOutput::into_bytes`]
//! detaches owned bytes before an asynchronous transport send, but it does not
//! make encoding asynchronous.
//! A codec that performs slow or blocking work would still block its caller;
//! bounded offloading is intentionally left to a future execution layer at the
//! engine service boundary.

use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use bytes::Bytes;
use otel_arrow_dfe_pdata::{
    OtapArrowRecords, OtapPayloadHelpers, PdataArrowRecordSet, PdataRepresentation,
};

use crate::{
    CodecError, CodecOperation, CodecRegistry, DecodePolicy, EncodeOutput, EncodedPdata,
    EncodingPlan, InspectionPlan, PdataArrowDecoder, PdataArrowEncoder, PdataDecoder, PdataEncoder,
    PdataView, RegistryError, ResolvedArrowCodec, ResolvedCodec,
};

struct DecoderInstance {
    codec: ResolvedCodec,
    decoder: Box<dyn PdataDecoder>,
}

struct EncoderInstance {
    plan: EncodingPlan,
    encoder: Box<dyn PdataEncoder>,
}

struct ArrowDecoderInstance {
    codec: ResolvedArrowCodec,
    decoder: Box<dyn PdataArrowDecoder>,
}

struct ArrowEncoderInstance {
    codec: ResolvedArrowCodec,
    encoder: Box<dyn PdataArrowEncoder>,
}

#[derive(Default)]
struct CodecRuntime {
    decoders: Vec<DecoderInstance>,
    encoders: Vec<EncoderInstance>,
    arrow_decoders: Vec<ArrowDecoderInstance>,
    arrow_encoders: Vec<ArrowEncoderInstance>,
}

/// Builds a pipeline-local codec service from a validated registry.
pub struct CodecServiceBuilder {
    registry: Arc<CodecRegistry>,
    decode_policy: DecodePolicy,
}

impl CodecServiceBuilder {
    /// Validates and selects the process-wide link-time registry.
    pub fn from_global_registry() -> Result<Self, RegistryError> {
        Ok(Self {
            registry: CodecRegistry::global()?,
            decode_policy: DecodePolicy::default(),
        })
    }

    /// Selects an already validated registry.
    #[must_use]
    pub fn from_registry(registry: Arc<CodecRegistry>) -> Self {
        Self {
            registry,
            decode_policy: DecodePolicy::default(),
        }
    }

    /// Selects the policy supplied to every decoder created by this service.
    #[must_use]
    pub const fn with_decode_policy(mut self, decode_policy: DecodePolicy) -> Self {
        self.decode_policy = decode_policy;
        self
    }

    /// Creates fresh lazy mutable state for one pipeline runtime.
    #[must_use]
    pub fn build(self) -> CodecService {
        CodecService {
            registry: self.registry,
            decode_policy: self.decode_policy,
            runtime: Arc::new(Mutex::new(CodecRuntime::default())),
        }
    }
}

/// Scoped access to mutable codec instances owned by one pipeline runtime.
///
/// Codec operations are synchronous and release the runtime lock before
/// returning. Callers must detach prepared output before awaiting asynchronous
/// work. The module-level documentation describes the execution boundary.
#[derive(Clone)]
pub struct CodecService {
    registry: Arc<CodecRegistry>,
    decode_policy: DecodePolicy,
    runtime: Arc<Mutex<CodecRuntime>>,
}

impl CodecService {
    /// Creates a service after validating the process registry.
    pub fn new() -> Result<Self, RegistryError> {
        Ok(CodecServiceBuilder::from_global_registry()?.build())
    }

    /// Validated immutable registry used by this pipeline.
    #[must_use]
    pub fn registry(&self) -> &CodecRegistry {
        &self.registry
    }

    /// Decode policy resolved for this pipeline-local service.
    #[must_use]
    pub const fn decode_policy(&self) -> DecodePolicy {
        self.decode_policy
    }

    /// Admits a known Arrow representation for pass-through without conversion.
    pub fn admit_arrow(
        &self,
        records: &PdataArrowRecordSet,
    ) -> Result<ResolvedArrowCodec, CodecError> {
        self.registry
            .resolve_arrow_passthrough(records.representation(), records.signal_type())
    }

    fn lock(&self) -> Result<MutexGuard<'_, CodecRuntime>, CodecError> {
        match self.runtime.try_lock() {
            Ok(runtime) => Ok(runtime),
            Err(TryLockError::WouldBlock) => Err(CodecError::ServiceBusy),
            Err(TryLockError::Poisoned(_)) => Err(CodecError::ServicePoisoned),
        }
    }

    /// Decodes admitted bytes through a lazily reused decoder instance.
    pub fn decode(&self, encoded: &EncodedPdata) -> Result<OtapArrowRecords, CodecError> {
        let mut runtime = self.lock()?;
        let records = runtime
            .decoder(encoded.codec(), self.decode_policy)?
            .decode(encoded.signal_type(), encoded.bytes())
            .map_err(|error| {
                error.with_operation_context(encoded.encoding(), CodecOperation::Decode)
            })?;
        if records.signal_type() != encoded.signal_type() {
            return Err(CodecError::SignalChanged {
                encoding: encoded.encoding().clone(),
                expected: encoded.signal_type(),
                actual: records.signal_type(),
            });
        }
        Ok(records)
    }

    /// Decodes a pluggable Arrow record set through a lazily reused codec.
    pub fn decode_arrow(
        &self,
        records: &PdataArrowRecordSet,
    ) -> Result<OtapArrowRecords, CodecError> {
        let codec = self
            .registry
            .resolve_arrow_decoder(records.representation(), records.signal_type())?;
        let mut runtime = self.lock()?;
        let decoded = runtime
            .arrow_decoder(codec)?
            .decode(records)
            .map_err(|error| {
                error.with_arrow_operation_context(records.representation(), CodecOperation::Decode)
            })?;
        if decoded.signal_type() != records.signal_type() {
            return Err(CodecError::ArrowSignalChanged {
                representation: records.representation().clone(),
                expected: records.signal_type(),
                actual: decoded.signal_type(),
            });
        }
        Ok(decoded)
    }

    /// Returns encoded bytes when accepted, otherwise decoded native records.
    pub fn view<'a>(
        &self,
        encoded: &'a EncodedPdata,
        plan: &InspectionPlan,
    ) -> Result<PdataView<'a>, CodecError> {
        if plan.accepts(encoded.codec()) {
            return Ok(PdataView::Encoded(crate::EncodedView::new(
                encoded.encoding(),
                encoded.signal_type(),
                encoded.bytes(),
            )));
        }
        self.decode(encoded)
            .map(|records| PdataView::Native(std::borrow::Cow::Owned(records)))
    }

    /// Runs a synchronous consumer while prepared output may borrow scratch.
    pub fn with_encoded_output<R>(
        &self,
        records: &mut OtapArrowRecords,
        plan: &EncodingPlan,
        consume: impl FnOnce(EncodeOutput<'_>) -> R,
    ) -> Result<R, CodecError> {
        let codec = plan.codec();
        codec.require_encoder(records.signal_type())?;
        let mut runtime = self.lock()?;
        let encoder = runtime.encoder(*plan).map_err(|error| {
            error.with_operation_context(codec.encoding(), CodecOperation::Encode)
        })?;
        let output = encoder.prepare_encode(records).map_err(|error| {
            error.with_operation_context(codec.encoding(), CodecOperation::Encode)
        })?;
        Ok(consume(output))
    }

    /// Detaches owned bytes before a caller performs an asynchronous send.
    pub fn encode_bytes(
        &self,
        records: &mut OtapArrowRecords,
        plan: &EncodingPlan,
    ) -> Result<Bytes, CodecError> {
        self.with_encoded_output(records, plan, |output| output.into_bytes())
    }

    /// Encodes native OTAP as a registered pluggable Arrow representation.
    pub fn encode_arrow(
        &self,
        records: OtapArrowRecords,
        representation: &PdataRepresentation,
    ) -> Result<PdataArrowRecordSet, CodecError> {
        let signal = records.signal_type();
        let codec = self
            .registry
            .resolve_arrow_encoder(representation, signal)?;
        let mut runtime = self.lock()?;
        let encoded = runtime
            .arrow_encoder(codec)?
            .encode(records)
            .map_err(|error| {
                error.with_arrow_operation_context(representation, CodecOperation::Encode)
            })?;
        if encoded.representation() != representation {
            return Err(CodecError::ArrowRepresentationChanged {
                expected: representation.clone(),
                actual: encoded.representation().clone(),
            });
        }
        if encoded.signal_type() != signal {
            return Err(CodecError::ArrowSignalChanged {
                representation: representation.clone(),
                expected: signal,
                actual: encoded.signal_type(),
            });
        }
        Ok(encoded)
    }

    /// Returns whether two handles address the same pipeline-owned state.
    #[must_use]
    pub fn shares_state_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.runtime, &other.runtime)
    }

    /// Number of lazily created mutable instances.
    #[cfg(any(test, feature = "testing"))]
    pub fn test_instance_count(&self) -> Result<usize, CodecError> {
        let runtime = self.lock()?;
        Ok(runtime.decoders.len()
            + runtime.encoders.len()
            + runtime.arrow_decoders.len()
            + runtime.arrow_encoders.len())
    }
}

impl CodecRuntime {
    fn decoder(
        &mut self,
        codec: ResolvedCodec,
        policy: DecodePolicy,
    ) -> Result<&mut dyn PdataDecoder, CodecError> {
        let index = match self
            .decoders
            .iter()
            .position(|instance| instance.codec == codec)
        {
            Some(index) => index,
            None => {
                let index = self.decoders.len();
                self.decoders.push(DecoderInstance {
                    codec,
                    decoder: codec.create_decoder(policy)?,
                });
                index
            }
        };
        Ok(self.decoders[index].decoder.as_mut())
    }

    fn encoder(&mut self, plan: EncodingPlan) -> Result<&mut dyn PdataEncoder, CodecError> {
        let index = match self
            .encoders
            .iter()
            .position(|instance| instance.plan == plan)
        {
            Some(index) => index,
            None => {
                let index = self.encoders.len();
                self.encoders.push(EncoderInstance {
                    plan,
                    encoder: plan.codec().create_encoder(plan.policy())?,
                });
                index
            }
        };
        Ok(self.encoders[index].encoder.as_mut())
    }

    fn arrow_decoder(
        &mut self,
        codec: ResolvedArrowCodec,
    ) -> Result<&mut dyn PdataArrowDecoder, CodecError> {
        let index = match self
            .arrow_decoders
            .iter()
            .position(|instance| instance.codec == codec)
        {
            Some(index) => index,
            None => {
                let index = self.arrow_decoders.len();
                self.arrow_decoders.push(ArrowDecoderInstance {
                    codec,
                    decoder: codec.create_decoder()?,
                });
                index
            }
        };
        Ok(self.arrow_decoders[index].decoder.as_mut())
    }

    fn arrow_encoder(
        &mut self,
        codec: ResolvedArrowCodec,
    ) -> Result<&mut dyn PdataArrowEncoder, CodecError> {
        let index = match self
            .arrow_encoders
            .iter()
            .position(|instance| instance.codec == codec)
        {
            Some(index) => index,
            None => {
                let index = self.arrow_encoders.len();
                self.arrow_encoders.push(ArrowEncoderInstance {
                    codec,
                    encoder: codec.create_encoder()?,
                });
                index
            }
        };
        Ok(self.arrow_encoders[index].encoder.as_mut())
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Arc;

    use prost::Message;

    use super::*;
    use crate::{
        ArrowCodecMetadata, ArrowCodecRegistration, CodecMetadata, CodecRegistration, EncodePolicy,
        PdataEncoding,
    };
    use otel_arrow_dfe_config::SignalType;
    use otel_arrow_dfe_pdata::{
        PdataRepresentationId, RepresentationVersion,
        testing::fixtures::{
            logs_with_full_resource_and_scope, metrics_sum_with_full_resource_and_scope,
        },
    };

    const LOGS_ONLY_ENCODING: PdataEncoding = PdataEncoding::new("logs-only-test-v1");
    static LOGS_ONLY_METADATA: CodecMetadata =
        CodecMetadata::new(LOGS_ONLY_ENCODING, &[SignalType::Logs]);
    static LOGS_ONLY_REGISTRATIONS: [CodecRegistration; 1] =
        [CodecRegistration::new(&LOGS_ONLY_METADATA)
            .with_decoder(|_| Box::new(FailingDecoder))
            .with_encoder(|_| Ok(Box::new(PanicEncoder)))
            .with_item_counter(|_, _| Some(0))];
    static NO_BYTE_REGISTRATIONS: [CodecRegistration; 0] = [];

    const TEST_ARROW_REPRESENTATION: PdataRepresentation = PdataRepresentation::new(
        PdataRepresentationId::new("test-arrow"),
        RepresentationVersion::new(1, 0),
    );
    static TEST_ARROW_METADATA: ArrowCodecMetadata =
        ArrowCodecMetadata::new(TEST_ARROW_REPRESENTATION, &[SignalType::Logs]);
    static TEST_ARROW_REGISTRATIONS: [ArrowCodecRegistration; 1] =
        [ArrowCodecRegistration::new(&TEST_ARROW_METADATA)
            .with_decoder(|| Box::new(EmptyArrowDecoder))
            .with_encoder(|| Box::new(EmptyArrowEncoder))];

    struct FailingDecoder;

    impl PdataDecoder for FailingDecoder {
        fn decode(
            &mut self,
            _signal: SignalType,
            _bytes: &Bytes,
        ) -> Result<OtapArrowRecords, CodecError> {
            Err(CodecError::ServiceBusy)
        }
    }

    struct PanicEncoder;

    impl PdataEncoder for PanicEncoder {
        fn encode(&mut self, _records: OtapArrowRecords) -> Result<Bytes, CodecError> {
            panic!("signal validation must run before this test encoder")
        }
    }

    struct EmptyArrowDecoder;

    impl PdataArrowDecoder for EmptyArrowDecoder {
        fn decode(
            &mut self,
            _records: &PdataArrowRecordSet,
        ) -> Result<OtapArrowRecords, CodecError> {
            Ok(OtapArrowRecords::Logs(Default::default()))
        }
    }

    struct EmptyArrowEncoder;

    impl PdataArrowEncoder for EmptyArrowEncoder {
        fn encode(&mut self, records: OtapArrowRecords) -> Result<PdataArrowRecordSet, CodecError> {
            PdataArrowRecordSet::new(
                TEST_ARROW_REPRESENTATION,
                records.signal_type(),
                records.num_items(),
                [],
            )
            .map_err(|error| {
                CodecError::arrow_operation(
                    &TEST_ARROW_REPRESENTATION,
                    CodecOperation::Encode,
                    error,
                )
            })
        }
    }

    fn service_for(registrations: &'static [CodecRegistration]) -> CodecService {
        let registry = CodecRegistry::validate(registrations).expect("valid test registry");
        CodecServiceBuilder::from_registry(Arc::new(registry)).build()
    }

    fn arrow_service() -> CodecService {
        let registry =
            CodecRegistry::validate_with_arrow(&NO_BYTE_REGISTRATIONS, &TEST_ARROW_REGISTRATIONS)
                .expect("valid Arrow test registry");
        CodecServiceBuilder::from_registry(Arc::new(registry)).build()
    }

    fn arrow_records(representation: PdataRepresentation) -> PdataArrowRecordSet {
        PdataArrowRecordSet::new(representation, SignalType::Logs, 0, [])
            .expect("valid empty Arrow record set")
    }

    fn records(signal: SignalType) -> OtapArrowRecords {
        let bytes: Bytes = match signal {
            SignalType::Logs => logs_with_full_resource_and_scope().encode_to_vec().into(),
            SignalType::Metrics => metrics_sum_with_full_resource_and_scope()
                .encode_to_vec()
                .into(),
            SignalType::Traces => unreachable!("runtime service tests do not request traces"),
        };
        let service = CodecService::new().expect("valid built-in registry");
        let codec = service
            .registry()
            .resolve_decoder(&PdataEncoding::OTLP, signal)
            .expect("built-in OTLP decoder");
        let encoded = codec.admit(signal, bytes).expect("supported OTLP signal");
        service.decode(&encoded).expect("valid OTLP fixture")
    }

    fn otlp_plan(service: &CodecService) -> EncodingPlan {
        EncodingPlan::resolve(
            service.registry(),
            &PdataEncoding::OTLP,
            EncodePolicy::default(),
        )
        .expect("built-in OTLP encoder")
    }

    /// Scenario: A logs-only output plan receives native metrics records.
    /// Guarantees: Runtime signal validation rejects the payload before creating or invoking the encoder.
    #[test]
    fn encoding_plan_validates_the_actual_records_signal() {
        let service = service_for(&LOGS_ONLY_REGISTRATIONS);
        let plan = EncodingPlan::resolve(
            service.registry(),
            &LOGS_ONLY_ENCODING,
            EncodePolicy::default(),
        )
        .unwrap();
        let mut metrics = records(SignalType::Metrics);

        assert!(matches!(
            service.encode_bytes(&mut metrics, &plan),
            Err(CodecError::Unsupported {
                operation: CodecOperation::Encode,
                signal: SignalType::Metrics,
                ..
            })
        ));
        assert_eq!(service.test_instance_count().unwrap(), 0);
    }

    /// Scenario: A decoder returns an error without the resolved codec identity.
    /// Guarantees: The codec service adds the actual identity and operation exactly once.
    #[test]
    fn decoder_failures_receive_service_owned_context() {
        let service = service_for(&LOGS_ONLY_REGISTRATIONS);
        let codec = service
            .registry()
            .resolve_decoder(&LOGS_ONLY_ENCODING, SignalType::Logs)
            .unwrap();
        let encoded = codec.admit(SignalType::Logs, Bytes::new()).unwrap();

        assert!(matches!(
            service.decode(&encoded),
            Err(CodecError::Operation {
                encoding,
                operation: CodecOperation::Decode,
                ..
            }) if encoding == LOGS_ONLY_ENCODING
        ));
    }

    /// Scenario: A prepared-output callback attempts another operation on the same service.
    /// Guarantees: Nested codec access returns an error instead of deadlocking the pipeline runtime.
    #[test]
    fn prepared_output_callback_cannot_reenter_the_service() {
        let service = CodecService::new().unwrap();
        let plan = otlp_plan(&service);
        let mut outer = records(SignalType::Logs);
        let mut nested = records(SignalType::Logs);

        let nested_result = service
            .with_encoded_output(&mut outer, &plan, |_| {
                service.encode_bytes(&mut nested, &plan)
            })
            .unwrap();

        assert!(matches!(nested_result, Err(CodecError::ServiceBusy)));
    }

    /// Scenario: A prepared-output consumer panics while holding codec runtime state.
    /// Guarantees: Later operations report poisoned state instead of reusing potentially corrupted buffers.
    #[test]
    fn poisoned_codec_state_is_not_reused() {
        let service = CodecService::new().unwrap();
        let plan = otlp_plan(&service);
        let mut records = records(SignalType::Logs);

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = service.with_encoded_output(&mut records, &plan, |_| -> () {
                panic!("poison codec service for the test")
            });
        }));
        assert!(panic.is_err());

        assert!(matches!(
            service.encode_bytes(&mut records, &plan),
            Err(CodecError::ServicePoisoned)
        ));
    }

    /// Scenario: The built-in OTLP byte codec is inspected through generalized representation metadata.
    /// Guarantees: Its stable identity, version, and bidirectional capabilities are discoverable without changing the optimized byte payload.
    #[test]
    fn built_in_otlp_exposes_generalized_representation_metadata() {
        let service = CodecService::new().expect("valid built-in registry");
        let codec = service
            .registry()
            .resolve(&PdataEncoding::OTLP)
            .expect("built-in OTLP codec");

        assert_eq!(
            codec.metadata().representation(),
            Some(PdataRepresentation::OTLP_BYTES)
        );
        assert!(codec.capabilities().can_decode_to_otap());
        assert!(codec.capabilities().can_encode_from_otap());
    }

    /// Scenario: A payload names an Arrow representation that is not linked into the binary.
    /// Guarantees: Admission returns a typed unavailable-codec error without inspecting its tables.
    #[test]
    fn missing_arrow_codec_is_reported_clearly() {
        let service = service_for(&NO_BYTE_REGISTRATIONS);
        let missing = PdataRepresentation::new(
            PdataRepresentationId::new("missing-arrow"),
            RepresentationVersion::new(1, 0),
        );
        let input = arrow_records(missing.clone());

        assert!(matches!(
            service.admit_arrow(&input),
            Err(CodecError::Registry(RegistryError::ArrowNotFound {
                representation
            })) if representation == missing
        ));
        assert_eq!(service.test_instance_count().unwrap(), 0);
    }

    /// Scenario: A registered Arrow representation supports conversion in both directions.
    /// Guarantees: Capabilities are discoverable and pipeline-local codec instances are reused.
    #[test]
    fn arrow_codec_capabilities_and_state_are_reused() {
        let service = arrow_service();
        let codec = service
            .registry()
            .resolve_arrow(&TEST_ARROW_REPRESENTATION)
            .expect("registered Arrow codec");
        assert!(codec.capabilities().can_decode_to_otap());
        assert!(codec.capabilities().can_encode_from_otap());

        let input = arrow_records(TEST_ARROW_REPRESENTATION);
        assert_eq!(
            service
                .admit_arrow(&input)
                .expect("admit registered Arrow records"),
            codec
        );
        assert_eq!(service.test_instance_count().unwrap(), 0);
        let decoded = service.decode_arrow(&input).expect("decode Arrow records");
        assert_eq!(decoded.signal_type(), SignalType::Logs);
        assert_eq!(service.test_instance_count().unwrap(), 1);
        let _ = service.decode_arrow(&input).expect("reuse Arrow decoder");
        assert_eq!(service.test_instance_count().unwrap(), 1);

        let encoded = service
            .encode_arrow(decoded, &TEST_ARROW_REPRESENTATION)
            .expect("encode Arrow records");
        assert_eq!(encoded.representation(), &TEST_ARROW_REPRESENTATION);
        assert_eq!(service.test_instance_count().unwrap(), 2);
    }
}
