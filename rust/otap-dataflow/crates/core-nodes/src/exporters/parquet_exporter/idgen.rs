// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//! This crate contains code for generating unique IDs for the various
//! record batches that are being exported to parquet

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, StructArray, UInt32Array};
use arrow::compute::kernels::cast;
use arrow::compute::kernels::numeric::add;
use arrow::compute::max;
use arrow::datatypes::{DataType, Field, Schema, UInt32Type};
use otel_arrow_dfe_pdata::proto::opentelemetry::arrow::v1::ArrowPayloadType;
use otel_arrow_dfe_pdata::schema::{consts, update_schema_metadata};
use uuid::Uuid;

use super::error::ParquetExporterError;
use super::records::OtapParquetRecords;

/// This is an ID generator that converts the batch IDs (which are only unique within a given
/// OTAP Batch) into IDs that can be treated as unique.
///
/// This particular implementation only generates IDs that are unique within some partition.
/// It expects the writer to partition the parquet file by the partition ID, which it sets on
/// the schema metadata.
///
/// Within the partition, the IDs increment more or less sequentially until they would overflow
/// the datatype (u32).
pub struct PartitionSequenceIdGenerator {
    part_id: Uuid,

    // current max ID. New batches will have their ID columns incremented by at least
    // this value to maintain unique IDs within the partition
    curr_max: u32,
}

pub const PARTITION_METADATA_KEY: &str = "_part_id";

impl Default for PartitionSequenceIdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl PartitionSequenceIdGenerator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            part_id: Uuid::new_v4(),
            curr_max: 0,
        }
    }

    fn init(&mut self) {
        self.part_id = Uuid::new_v4();
        self.curr_max = 0;
    }

    pub fn generate_unique_ids(
        &mut self,
        otap_batch: &mut OtapParquetRecords,
    ) -> Result<(), ParquetExporterError> {
        let payload_types = otap_batch.allowed_payload_types();

        // find the max ID in any of the record batches. We'll use this later on to increment
        // curr_max and also check for overflows before incrementing all the IDs
        let mut max_id = 0;
        for payload_type in payload_types {
            if let Some(batch) = otap_batch.get(*payload_type) {
                for name in [consts::ID, consts::PARENT_ID] {
                    if let Some(column) = batch.column_by_name(name) {
                        max_id = max_id.max(Self::max_id(column)?);
                    }
                }
                for name in [consts::RESOURCE, consts::SCOPE] {
                    if let Some(column) = batch.column_by_name(name)
                        && let Some(value) = column.as_any().downcast_ref::<StructArray>()
                        && let Some(id) = value.column_by_name(consts::ID)
                    {
                        max_id = max_id.max(Self::max_id(id)?);
                    }
                }
            }
        }

        // check if we'll overflow when we increment the IDs
        if max_id.checked_add(self.curr_max).is_none() {
            // if adding curr_max to some record batch would cause the id sequence to overflow,
            // re-init self to begin writing to new partition
            self.init();
        }

        // update IDs for all record batch types
        for payload_type in payload_types {
            if let Some(rb) = otap_batch.get(*payload_type) {
                let rb = self.add_offset_to_column(consts::ID, rb)?;
                let rb = self.add_offset_to_column(consts::PARENT_ID, &rb)?;
                let rb = self.add_offset_to_nested_id(consts::RESOURCE, &rb)?;
                let rb = self.add_offset_to_nested_id(consts::SCOPE, &rb)?;
                otap_batch.set(*payload_type, rb);
            }

            // update the schema metadata with the partition key on the main record batch
            match payload_type {
                ArrowPayloadType::Logs
                | ArrowPayloadType::UnivariateMetrics
                | ArrowPayloadType::Spans => {
                    if let Some(main_rb) = otap_batch.get(*payload_type) {
                        otap_batch.set(
                            *payload_type,
                            update_schema_metadata(
                                main_rb,
                                PARTITION_METADATA_KEY.into(),
                                format!("{}", self.part_id),
                            ),
                        );
                    }
                }
                _ => {}
            }
        }

        self.curr_max += max_id;
        // try incrementing curr_max by 1 so the next batch won't have
        // any overlapping IDs (batch IDs could start at 0)
        match self.curr_max.checked_add(1) {
            Some(new_curr_max) => {
                self.curr_max = new_curr_max;
            }
            None => {
                self.init();
            }
        }

        Ok(())
    }

    fn add_offset_to_column(
        &self,
        column_name: &str,
        record_batch: &RecordBatch,
    ) -> Result<RecordBatch, ParquetExporterError> {
        let schema = record_batch.schema_ref();
        let id_column_index = match schema.index_of(column_name) {
            Ok(index) => index,
            Err(_) => {
                // column doesn't exist, nothing to do
                return Ok(record_batch.clone());
            }
        };
        let new_id_col = self.offset_values(record_batch.column(id_column_index))?;

        // construct new record batch
        let new_columns = record_batch
            .columns()
            .iter()
            .enumerate()
            .map(|(i, col)| {
                if i == id_column_index {
                    new_id_col.clone()
                } else {
                    col.clone()
                }
            })
            .collect::<Vec<_>>();

        let new_fields = schema
            .fields()
            .into_iter()
            .enumerate()
            .map(|(i, field)| {
                if i == id_column_index {
                    Arc::new(Field::new(column_name, DataType::UInt32, true))
                } else {
                    field.clone()
                }
            })
            .collect::<Vec<_>>();

        // safety: this call should succeed based on how we're constructing the record batch
        // e.g. using the same fields/columns, except for replacing ID with the correct type
        Ok(
            RecordBatch::try_new(Arc::new(Schema::new(new_fields)), new_columns)
                .expect("can construct record batch"),
        )
    }

    fn max_id(column: &ArrayRef) -> Result<u32, ParquetExporterError> {
        let column = cast(column, &DataType::UInt32).map_err(|error| {
            ParquetExporterError::InvalidRecordBatch {
                error: format!("could not cast ID column to UInt32: {error}"),
            }
        })?;
        let values = column
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("cast produces UInt32Array");
        Ok(max::<UInt32Type>(values).unwrap_or(0))
    }

    fn offset_values(&self, column: &ArrayRef) -> Result<ArrayRef, ParquetExporterError> {
        let column = cast(column, &DataType::UInt32).map_err(|error| {
            ParquetExporterError::InvalidRecordBatch {
                error: format!("could not cast ID column to UInt32: {error}"),
            }
        })?;
        add(&UInt32Array::new_scalar(self.curr_max), &column).map_err(|error| {
            ParquetExporterError::UnknownError {
                error: error.to_string(),
            }
        })
    }

    fn add_offset_to_nested_id(
        &self,
        name: &str,
        batch: &RecordBatch,
    ) -> Result<RecordBatch, ParquetExporterError> {
        let Ok(index) = batch.schema_ref().index_of(name) else {
            return Ok(batch.clone());
        };
        let value = batch
            .column(index)
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| ParquetExporterError::InvalidRecordBatch {
                error: format!("{name} must be a struct"),
            })?;
        let Some((id_index, _)) = value.fields().find(consts::ID) else {
            return Ok(batch.clone());
        };
        let mut columns = value.columns().to_vec();
        columns[id_index] = self.offset_values(&columns[id_index])?;
        let mut fields = value.fields().to_vec();
        fields[id_index] = Arc::new(Field::new(consts::ID, DataType::UInt32, true));
        let value = StructArray::try_new(fields.into(), columns, value.nulls().cloned()).map_err(
            |error| ParquetExporterError::InvalidRecordBatch {
                error: error.to_string(),
            },
        )?;
        let mut columns = batch.columns().to_vec();
        let mut fields = batch.schema_ref().fields().to_vec();
        fields[index] = Arc::new(
            fields[index]
                .as_ref()
                .clone()
                .with_data_type(value.data_type().clone()),
        );
        columns[index] = Arc::new(value);
        RecordBatch::try_new(
            Arc::new(Schema::new(fields).with_metadata(batch.schema_ref().metadata().clone())),
            columns,
        )
        .map_err(|error| ParquetExporterError::InvalidRecordBatch {
            error: error.to_string(),
        })
    }
}

#[cfg(test)]
pub mod test {
    use std::collections::HashMap;

    use arrow::array::UInt16Array;
    use otel_arrow_dfe_pdata::Consumer;
    use otel_arrow_dfe_pdata::otap::{
        Metrics, OtapArrowRecords, Traces, from_record_messages, testing::complete_batch,
    };
    use otel_arrow_dfe_pdata::proto::opentelemetry::arrow::v1::ArrowPayloadType;
    use otel_arrow_dfe_pdata::schema::consts::metadata;
    use otel_arrow_dfe_pdata::schema::get_schema_metadata;

    use crate::exporters::parquet_exporter::fixtures::{
        SimpleDataGenOptions, create_simple_logs_arrow_record_batches,
    };

    use super::*;

    /// Scenario: repeated batches reuse resource and scope IDs while attributes reference those IDs.
    /// Guarantees: nested references advance with child parent IDs, include their maxima, and preserve struct null masks.
    #[test]
    fn resource_and_scope_references_follow_attribute_ids() {
        use arrow::buffer::NullBuffer;
        use otel_arrow_dfe_pdata::otap::raw_batch_store::RawLogsStore;
        let make = || {
            let nested = |id| -> ArrayRef {
                Arc::new(
                    StructArray::try_new(
                        vec![Field::new(consts::ID, DataType::UInt16, true)].into(),
                        vec![Arc::new(UInt16Array::from(vec![Some(id), None]))],
                        Some(NullBuffer::from(vec![true, false])),
                    )
                    .unwrap(),
                )
            };
            let root = RecordBatch::try_from_iter(vec![
                (
                    consts::ID,
                    Arc::new(UInt16Array::from(vec![0, 1])) as ArrayRef,
                ),
                (consts::RESOURCE, nested(7)),
                (consts::SCOPE, nested(9)),
            ])
            .unwrap();
            let mut store = RawLogsStore::new();
            store.set(ArrowPayloadType::Logs, root);
            for (payload, id) in [
                (ArrowPayloadType::ResourceAttrs, 7),
                (ArrowPayloadType::ScopeAttrs, 9),
            ] {
                store.set(
                    payload,
                    RecordBatch::try_from_iter(vec![(
                        consts::PARENT_ID,
                        Arc::new(UInt16Array::from(vec![id])) as ArrayRef,
                    )])
                    .unwrap(),
                );
            }
            OtapParquetRecords::Logs(store)
        };
        let mut generator = PartitionSequenceIdGenerator::new();
        for offset in [0, 10] {
            let mut records = make();
            generator.generate_unique_ids(&mut records).unwrap();
            let root = records.get(ArrowPayloadType::Logs).unwrap();
            for (name, payload, original) in [
                (consts::RESOURCE, ArrowPayloadType::ResourceAttrs, 7),
                (consts::SCOPE, ArrowPayloadType::ScopeAttrs, 9),
            ] {
                let nested = root
                    .column_by_name(name)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .unwrap();
                assert!(nested.is_null(1));
                let ids = nested
                    .column_by_name(consts::ID)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap();
                let children = records
                    .get(payload)
                    .unwrap()
                    .column_by_name(consts::PARENT_ID)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap();
                assert_eq!(ids.value(0), original + offset);
                assert_eq!(ids.value(0), children.value(0));
                assert!(ids.is_null(1));
            }
        }
    }

    #[test]
    fn test_partition_sequence_id_generator_logs() {
        let mut id_generator = PartitionSequenceIdGenerator::new();

        let mut log_batch = create_simple_logs_arrow_record_batches(SimpleDataGenOptions {
            id_offset: 0,
            num_rows: 2,
            ..Default::default()
        });
        let mut consumer = Consumer::default();
        let record_messages = consumer.consume_bar(&mut log_batch).unwrap();
        let mut otap_batch1: OtapParquetRecords =
            OtapArrowRecords::Logs(from_record_messages(record_messages).unwrap()).into();
        id_generator.generate_unique_ids(&mut otap_batch1).unwrap();
        let expected_ids = UInt32Array::from_iter_values(vec![0, 1]);

        let logs_rb = otap_batch1.get(ArrowPayloadType::Logs).unwrap();
        let id_col = logs_rb
            .column_by_name(consts::ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(id_col, &expected_ids);

        // check that the schema metadata was correctly applied
        let val = get_schema_metadata(logs_rb.schema_ref(), PARTITION_METADATA_KEY).unwrap();
        assert_eq!(val, &format!("{}", id_generator.part_id));

        // expect that the parent IDs are also updated
        for payload_type in [
            ArrowPayloadType::ResourceAttrs,
            ArrowPayloadType::ScopeAttrs,
            ArrowPayloadType::LogAttrs,
        ] {
            let rb = otap_batch1.get(payload_type).unwrap();
            let parent_id_col = rb
                .column_by_name(consts::PARENT_ID)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();
            assert_eq!(parent_id_col, &expected_ids);
        }

        // add a second batch
        let mut log_batch = create_simple_logs_arrow_record_batches(SimpleDataGenOptions {
            id_offset: 0,
            num_rows: 2,
            ..Default::default()
        });
        let mut consumer = Consumer::default();
        let record_messages = consumer.consume_bar(&mut log_batch).unwrap();
        let mut otap_batch2: OtapParquetRecords =
            OtapArrowRecords::Logs(from_record_messages(record_messages).unwrap()).into();
        id_generator.generate_unique_ids(&mut otap_batch2).unwrap();
        let expected_ids = UInt32Array::from_iter_values(vec![2, 3]);

        // validate the IDs are now offset by the max of the previous IDs
        let logs_rb = otap_batch2.get(ArrowPayloadType::Logs).unwrap();
        let id_col = logs_rb
            .column_by_name(consts::ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(id_col, &expected_ids);

        // expect that the parent IDs are also updated
        for payload_type in [
            ArrowPayloadType::ResourceAttrs,
            ArrowPayloadType::ScopeAttrs,
            ArrowPayloadType::LogAttrs,
        ] {
            let rb = otap_batch2.get(payload_type).unwrap();
            let parent_id_col = rb
                .column_by_name(consts::PARENT_ID)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();
            assert_eq!(parent_id_col, &expected_ids);
        }
    }

    #[test]
    fn test_partition_sequence_id_generator_logs_with_overflow() {
        let mut id_generator = PartitionSequenceIdGenerator::new();
        id_generator.curr_max = u32::MAX - 5;
        let curr_partition_id = id_generator.part_id;

        let mut log_batch = create_simple_logs_arrow_record_batches(SimpleDataGenOptions {
            id_offset: 0,
            num_rows: 10,
            ..Default::default()
        });
        let mut consumer = Consumer::default();
        let record_messages = consumer.consume_bar(&mut log_batch).unwrap();
        let mut otap_batch: OtapParquetRecords =
            OtapArrowRecords::Logs(from_record_messages(record_messages).unwrap()).into();

        id_generator.generate_unique_ids(&mut otap_batch).unwrap();
        let expected_ids = UInt32Array::from_iter_values(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let logs_rb = otap_batch.get(ArrowPayloadType::Logs).unwrap();
        let id_col = logs_rb
            .column_by_name(consts::ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(id_col, &expected_ids);

        // check that a new partition was created due to rollover
        assert_ne!(curr_partition_id, id_generator.part_id);
        let val = get_schema_metadata(logs_rb.schema_ref(), PARTITION_METADATA_KEY).unwrap();
        assert_eq!(val, &format!("{}", id_generator.part_id));
        assert_ne!(val, &format!("{curr_partition_id}"));
    }

    #[test]
    fn test_partition_sequence_id_generator_decodes_ids() {
        let mut id_generator = PartitionSequenceIdGenerator::new();

        let mut log_batch = create_simple_logs_arrow_record_batches(SimpleDataGenOptions {
            id_offset: 0,
            num_rows: 3,
            ids_decoded: false,
            ..Default::default()
        });
        let mut consumer = Consumer::default();
        let record_messages = consumer.consume_bar(&mut log_batch).unwrap();
        let mut otap_records =
            OtapArrowRecords::Logs(from_record_messages(record_messages).unwrap());
        // decode transport-optimized IDs before converting to parquet records
        otap_records.decode_transport_optimized_ids().unwrap();
        let mut otap_batch1: OtapParquetRecords = otap_records.into();
        id_generator.generate_unique_ids(&mut otap_batch1).unwrap();
        let expected_ids = UInt32Array::from_iter_values(vec![1, 2, 3]);

        let logs_rb = otap_batch1.get(ArrowPayloadType::Logs).unwrap();
        let id_col = logs_rb
            .column_by_name(consts::ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(id_col, &expected_ids);

        // check that the schema metadata was correctly applied
        let val = get_schema_metadata(logs_rb.schema_ref(), PARTITION_METADATA_KEY).unwrap();
        assert_eq!(val, &format!("{}", id_generator.part_id));

        // expect that the parent IDs are also updated
        for payload_type in [
            ArrowPayloadType::ResourceAttrs,
            ArrowPayloadType::ScopeAttrs,
            ArrowPayloadType::LogAttrs,
        ] {
            let rb = otap_batch1.get(payload_type).unwrap();
            let parent_id_col = rb
                .column_by_name(consts::PARENT_ID)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();
            assert_eq!(parent_id_col, &expected_ids);
        }
    }

    #[test]
    fn test_can_handle_u32_ids() {
        use otel_arrow_dfe_pdata::otap::raw_batch_store::RawTracesStore;

        let span_events = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(consts::PARENT_ID, DataType::UInt16, false).with_metadata(
                    HashMap::from_iter(vec![(
                        metadata::COLUMN_ENCODING.into(),
                        metadata::encodings::PLAIN.into(),
                    )]),
                ),
                Field::new(consts::ID, DataType::UInt32, true).with_metadata(HashMap::from_iter(
                    vec![(
                        metadata::COLUMN_ENCODING.into(),
                        metadata::encodings::PLAIN.into(),
                    )],
                )),
            ])),
            vec![
                Arc::new(UInt16Array::from_iter_values(vec![0, 0, 0])),
                Arc::new(UInt32Array::from_iter_values(vec![1, 2, 3])),
            ],
        )
        .unwrap();

        let span_event_attrs = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(consts::PARENT_ID, DataType::UInt32, true).with_metadata(
                    HashMap::from_iter(vec![(
                        metadata::COLUMN_ENCODING.into(),
                        metadata::encodings::PLAIN.into(),
                    )]),
                ),
            ])),
            vec![Arc::new(UInt32Array::from_iter_values(vec![1, 2, 3]))],
        )
        .unwrap();

        // Use RawTracesStore directly since these batches have non-spec types
        let mut raw_store = RawTracesStore::new();
        raw_store.set(ArrowPayloadType::SpanEvents, span_events);
        raw_store.set(ArrowPayloadType::SpanEventAttrs, span_event_attrs);
        let mut traces_batch = OtapParquetRecords::Traces(raw_store);

        let mut id_generator = PartitionSequenceIdGenerator::new();
        id_generator.curr_max = 2;
        id_generator.generate_unique_ids(&mut traces_batch).unwrap();

        let spans_events_result = traces_batch.get(ArrowPayloadType::SpanEvents).unwrap();
        let id_column = spans_events_result
            .column_by_name(consts::ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let expected_ids = UInt32Array::from_iter_values(vec![3, 4, 5]);
        assert_eq!(id_column, &expected_ids);

        let span_events_attrs_result = traces_batch.get(ArrowPayloadType::SpanEventAttrs).unwrap();
        let parent_id_column = span_events_attrs_result
            .column_by_name(consts::PARENT_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let expected_parent_ids = UInt32Array::from_iter_values(vec![3, 4, 5]);
        assert_eq!(parent_id_column, &expected_parent_ids);
    }

    #[test]
    fn test_partition_sequence_id_generator_update_root_payload_metadata() {
        // ensure that for other root payload types beyond logs, it also updates the metadata
        let root_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                consts::ID,
                DataType::UInt16,
                true,
            )])),
            vec![Arc::new(UInt16Array::from(vec![0u16]))],
        )
        .unwrap();

        let mut traces_batch: OtapParquetRecords = {
            let mut t = OtapArrowRecords::Traces(Traces::default());
            let spans_batch = complete_batch(ArrowPayloadType::Spans, root_batch.clone());
            t.set(ArrowPayloadType::Spans, spans_batch).unwrap();
            t.into()
        };
        let mut id_generator = PartitionSequenceIdGenerator::new();
        id_generator.generate_unique_ids(&mut traces_batch).unwrap();
        let spans_batch = traces_batch.get(ArrowPayloadType::Spans).unwrap();
        let partition_id =
            get_schema_metadata(spans_batch.schema_ref(), PARTITION_METADATA_KEY).unwrap();
        assert_eq!(partition_id, &format!("{}", id_generator.part_id));

        let mut metrics_batch: OtapParquetRecords = {
            let mut m = OtapArrowRecords::Metrics(Metrics::default());
            let metrics_root =
                complete_batch(ArrowPayloadType::UnivariateMetrics, root_batch.clone());
            m.set(ArrowPayloadType::UnivariateMetrics, metrics_root)
                .unwrap();
            m.into()
        };
        id_generator
            .generate_unique_ids(&mut metrics_batch)
            .unwrap();
        let spans_batch = metrics_batch
            .get(ArrowPayloadType::UnivariateMetrics)
            .unwrap();
        let partition_id =
            get_schema_metadata(spans_batch.schema_ref(), PARTITION_METADATA_KEY).unwrap();
        assert_eq!(partition_id, &format!("{}", id_generator.part_id));
    }
}
