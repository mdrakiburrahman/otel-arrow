// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

use std::error::Error as StdError;

use crate::PdataEncoding;
use otel_arrow_dfe_pdata::PdataRepresentation;

/// Operation being performed by a codec.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodecOperation {
    /// Preserve a registered representation without conversion.
    PassThrough,
    /// Convert encoded bytes to native OTAP.
    Decode,
    /// Convert native OTAP to encoded bytes.
    Encode,
}

impl std::fmt::Display for CodecOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::PassThrough => "pass through",
            Self::Decode => "decode",
            Self::Encode => "encode",
        })
    }
}

/// Invalid link-time codec registry.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum RegistryError {
    /// An encoding name does not satisfy the identity contract.
    #[error("invalid pdata encoding `{encoding}`: {reason}")]
    InvalidIdentity {
        /// Invalid identity.
        encoding: PdataEncoding,
        /// Validation failure.
        reason: &'static str,
    },
    /// More than one extension registered the same identity.
    #[error("duplicate pdata encoding registration `{encoding}`")]
    Duplicate {
        /// Duplicated identity.
        encoding: PdataEncoding,
    },
    /// A registration cannot perform any operation.
    #[error("pdata encoding `{encoding}` has no decoder or encoder")]
    EmptyCapabilities {
        /// Empty registration identity.
        encoding: PdataEncoding,
    },
    /// A registration has no supported signals.
    #[error("pdata encoding `{encoding}` has no supported signals")]
    EmptySignals {
        /// Invalid registration identity.
        encoding: PdataEncoding,
    },
    /// A decoder registration cannot report primary-signal item counts.
    #[error("pdata encoding `{encoding}` has a decoder but no item counter")]
    MissingItemCounter {
        /// Invalid registration identity.
        encoding: PdataEncoding,
    },
    /// The requested identity is not linked into this binary.
    #[error("no pdata codec registered for `{encoding}`")]
    NotFound {
        /// Missing identity.
        encoding: PdataEncoding,
    },
    /// A pluggable Arrow representation name or version is invalid.
    #[error("invalid pdata Arrow representation `{representation}`: {reason}")]
    InvalidArrowIdentity {
        /// Invalid representation identity.
        representation: PdataRepresentation,
        /// Validation failure.
        reason: &'static str,
    },
    /// More than one Arrow codec registered the same identity and version.
    #[error("duplicate pdata Arrow codec registration `{representation}`")]
    DuplicateArrow {
        /// Duplicated representation identity.
        representation: PdataRepresentation,
    },
    /// An Arrow registration has no supported signals.
    #[error("pdata Arrow representation `{representation}` has no supported signals")]
    EmptyArrowSignals {
        /// Invalid registration identity.
        representation: PdataRepresentation,
    },
    /// The requested Arrow representation is not linked into this binary.
    #[error("no pdata Arrow codec registered for `{representation}`")]
    ArrowNotFound {
        /// Missing representation identity.
        representation: PdataRepresentation,
    },
}

/// Codec resolution or execution failure.
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    /// Registry validation or lookup failed.
    #[error(transparent)]
    Registry(#[from] RegistryError),
    /// A registered codec does not implement the requested operation.
    #[error("pdata codec `{encoding}` cannot {operation}")]
    UnsupportedCodecOperation {
        /// Codec identity.
        encoding: PdataEncoding,
        /// Requested operation.
        operation: CodecOperation,
    },
    /// A registered codec does not support an operation for this signal.
    #[error("pdata codec `{encoding}` cannot {operation} {signal:?}")]
    Unsupported {
        /// Codec identity.
        encoding: PdataEncoding,
        /// Requested operation.
        operation: CodecOperation,
        /// Requested signal.
        signal: otel_arrow_dfe_config::SignalType,
    },
    /// A registered Arrow codec does not implement the requested operation.
    #[error("pdata Arrow codec `{representation}` cannot {operation}")]
    UnsupportedArrowCodecOperation {
        /// Representation identity.
        representation: PdataRepresentation,
        /// Requested operation.
        operation: CodecOperation,
    },
    /// A registered Arrow codec does not support an operation for this signal.
    #[error("pdata Arrow codec `{representation}` cannot {operation} {signal:?}")]
    UnsupportedArrow {
        /// Representation identity.
        representation: PdataRepresentation,
        /// Requested operation.
        operation: CodecOperation,
        /// Requested signal.
        signal: otel_arrow_dfe_config::SignalType,
    },
    /// A codec failed while processing telemetry.
    #[error("pdata codec `{encoding}` failed to {operation}: {source}")]
    Operation {
        /// Codec identity.
        encoding: PdataEncoding,
        /// Failed operation.
        operation: CodecOperation,
        /// Original error, retained for diagnostics.
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },
    /// An Arrow codec failed while processing telemetry.
    #[error("pdata Arrow codec `{representation}` failed to {operation}: {source}")]
    ArrowOperation {
        /// Representation identity.
        representation: PdataRepresentation,
        /// Failed operation.
        operation: CodecOperation,
        /// Original error, retained for diagnostics.
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },
    /// A decoder returned a different signal from the admitted envelope.
    #[error("pdata codec `{encoding}` decoded {actual:?}, expected {expected:?}")]
    SignalChanged {
        /// Codec identity.
        encoding: PdataEncoding,
        /// Signal admitted by the receiver.
        expected: otel_arrow_dfe_config::SignalType,
        /// Signal produced by the decoder.
        actual: otel_arrow_dfe_config::SignalType,
    },
    /// An Arrow decoder or encoder returned a different signal.
    #[error("pdata Arrow codec `{representation}` produced {actual:?}, expected {expected:?}")]
    ArrowSignalChanged {
        /// Representation identity.
        representation: PdataRepresentation,
        /// Signal supplied to the codec.
        expected: otel_arrow_dfe_config::SignalType,
        /// Signal produced by the codec.
        actual: otel_arrow_dfe_config::SignalType,
    },
    /// An Arrow encoder returned a different representation identity.
    #[error("pdata Arrow codec `{expected}` produced representation `{actual}`")]
    ArrowRepresentationChanged {
        /// Registered representation identity.
        expected: PdataRepresentation,
        /// Representation returned by the encoder.
        actual: PdataRepresentation,
    },
    /// Codec state is already borrowed by another synchronous operation.
    #[error(
        "pdata codec service is already in use; nested or concurrent codec access is unsupported"
    )]
    ServiceBusy,
    /// A panic occurred while codec state was borrowed.
    #[error("pdata codec service is unavailable after a panic while its state was borrowed")]
    ServicePoisoned,
}

impl CodecError {
    /// Wraps a codec implementation error without losing its source chain.
    pub fn operation(
        encoding: &PdataEncoding,
        operation: CodecOperation,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::Operation {
            encoding: encoding.clone(),
            operation,
            source: Box::new(source),
        }
    }

    /// Ensures an implementation failure carries service-owned diagnostics.
    pub(crate) fn with_operation_context(
        self,
        encoding: &PdataEncoding,
        operation: CodecOperation,
    ) -> Self {
        if matches!(
            &self,
            Self::Operation {
                encoding: error_encoding,
                operation: error_operation,
                ..
            } if error_encoding == encoding && *error_operation == operation
        ) {
            self
        } else {
            Self::operation(encoding, operation, self)
        }
    }

    /// Wraps an Arrow implementation error without losing its source chain.
    pub fn arrow_operation(
        representation: &PdataRepresentation,
        operation: CodecOperation,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::ArrowOperation {
            representation: representation.clone(),
            operation,
            source: Box::new(source),
        }
    }

    /// Ensures an Arrow implementation failure carries service-owned diagnostics.
    pub(crate) fn with_arrow_operation_context(
        self,
        representation: &PdataRepresentation,
        operation: CodecOperation,
    ) -> Self {
        if matches!(
            &self,
            Self::ArrowOperation {
                representation: error_representation,
                operation: error_operation,
                ..
            } if error_representation == representation && *error_operation == operation
        ) {
            self
        } else {
            Self::arrow_operation(representation, operation, self)
        }
    }
}
