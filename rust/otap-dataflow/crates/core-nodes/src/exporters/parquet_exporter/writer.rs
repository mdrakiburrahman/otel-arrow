// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use futures::future::LocalBoxFuture;
use futures::stream::FuturesUnordered;
use futures::{FutureExt, StreamExt, TryStreamExt};
use object_store::ObjectStore;
use otel_arrow_dfe_pdata::otap::child_payload_types;
use otel_arrow_dfe_pdata::proto::opentelemetry::arrow::v1::ArrowPayloadType;
use parquet::arrow::AsyncArrowWriter;
use parquet::arrow::async_writer::ParquetObjectWriter;
use parquet::errors::ParquetError;
use parquet::file::properties::WriterProperties;
use thiserror::Error;

use super::config::WriterOptions;
use super::partition::PartitionAttribute;
use super::records::OtapParquetRecords;

/// A file backend whose completion makes its rows visible to readers.
#[async_trait::async_trait(?Send)]
pub trait FileSink {
    /// Append prepared Arrow rows.
    async fn write(&mut self, batch: &RecordBatch) -> Result<(), ParquetError>;
    /// Close and publish the file, including any backend transaction.
    async fn close(self: Box<Self>) -> Result<(), ParquetError>;
    /// Finalize private data without publishing it. Ordinary Parquet defers close.
    async fn stage(&mut self) -> Result<(), ParquetError> {
        Ok(())
    }
    /// Publish during a forced drain, bypassing an external batching delay.
    #[must_use]
    async fn close_forced(self: Box<Self>) -> Result<(), ParquetError> {
        self.close().await
    }
}

/// Supplies file backends while retaining OTAP batching and visibility ordering.
pub trait FileSinkFactory {
    /// Create a backend for a payload's file path.
    fn create(&self, path: &str, schema: SchemaRef) -> Result<Box<dyn FileSink>, ParquetError>;
    /// Optional final preparation after the exporter's standard schema normalization.
    fn prepare(&self, _records: &mut OtapParquetRecords) -> Result<(), ParquetError> {
        Ok(())
    }
    /// Opt into local pipelined completion, bounded per payload type.
    fn max_pending_files(&self) -> Option<usize> {
        None
    }
    /// Force already submitted and subsequent files to complete during shutdown.
    fn start_draining(&self) -> Result<(), ParquetError> {
        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl FileSink for AsyncArrowWriter<ParquetObjectWriter> {
    async fn write(&mut self, batch: &RecordBatch) -> Result<(), ParquetError> {
        AsyncArrowWriter::write(self, batch).await
    }

    async fn close(self: Box<Self>) -> Result<(), ParquetError> {
        (*self).close().await.map(|_| ())
    }
}

/// Aggregated stats returned by a write/flush cycle. Useful for internal telemetry and tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct WriteStats {
    /// Number of new file writers created during this write call (one per newly encountered
    /// payload type/partition prefix). A proxy for file creations.
    pub files_created: u64,
    /// Number of Parquet writers closed during this cycle (files flushed and made visible).
    pub files_closed: u64,
    /// Total number of rows appended to file writers in this write call (not necessarily flushed yet).
    pub rows_written: u64,
    /// Number of writers scheduled for flush because they reached the target rows per file threshold.
    pub flush_scheduled_max_rows: u64,
    /// Number of writers scheduled for flush because they exceeded the age threshold.
    pub flush_scheduled_max_age: u64,
    /// Number of file close/flush attempts initiated.
    pub flush_attempts: u64,
    /// Number of file close/flush attempts that succeeded.
    pub flush_successes: u64,
    /// Number of file close/flush attempts that failed.
    pub flush_failures: u64,
}

#[derive(Debug, Default, Clone, Copy)]
struct FlushScheduleStats {
    scheduled_max_rows: u64,
    scheduled_max_age: u64,
}

#[derive(Debug, Default, Clone, Copy)]
struct FlushAttemptStats {
    files_closed: u64,
    flush_attempts: u64,
    flush_successes: u64,
    flush_failures: u64,
}

enum FlushEvent {
    Staged(FileWriter, Result<(), ParquetError>),
    Closed {
        batch_ids: BTreeSet<i64>,
        payload_type: ArrowPayloadType,
        result: Result<(), ParquetError>,
    },
}

/// Error returned by the writer with any flush stats collected before failure.
#[derive(Debug, Error)]
#[error("{error}")]
pub struct WriteError {
    /// Stats collected before returning the first underlying error.
    pub stats: WriteStats,
    /// The first underlying Parquet error.
    #[source]
    pub error: ParquetError,
}

impl WriteError {
    fn from_flush_stats(stats: FlushAttemptStats, error: ParquetError) -> Self {
        Self {
            stats: WriteStats {
                files_closed: stats.files_closed,
                flush_attempts: stats.flush_attempts,
                flush_successes: stats.flush_successes,
                flush_failures: stats.flush_failures,
                ..Default::default()
            },
            error,
        }
    }
}

pub struct WriteBatch<'a> {
    pub batch_id: i64,
    pub otap_batch: &'a OtapParquetRecords,
    pub partition_attributes: Option<&'a [PartitionAttribute]>,
}

impl<'a> WriteBatch<'a> {
    #[must_use]
    pub const fn new(
        batch_id: i64,
        otap_batch: &'a OtapParquetRecords,
        partition_attributes: Option<&'a [PartitionAttribute]>,
    ) -> Self {
        Self {
            batch_id,
            otap_batch,
            partition_attributes,
        }
    }
}

/// `WriterManager` controls the parquet object writers for each table associated with some
/// payload type of the OTAP data. It is capable of writing partitioned files, and
/// controlling the approximate number of rows in each file.
pub struct WriterManager {
    object_store: Arc<dyn ObjectStore>,
    options: WriterOptions,
    backend: Option<Arc<dyn FileSinkFactory>>,
    close_failed: bool,

    // the current filename used for each path prefix. Prefixes in this case are based
    // on the payload type, and the partition attributes. There can be multiple files
    // written within the same prefix as files are flushed.
    curr_writer_for_prefix: HashMap<String, FileWriter>,

    // state for unflushed batches. used to track which OTAP batches have unflushed writes
    // for given types of payloads. This is used to determine whether a file should be flushed
    // based on whether some child payload type has unflushed writes.
    unflushed_batches_state: UnflushedBatchState,

    // These are files that have had all their batches written, but the parquet writer
    // has not yet been closed. We hold these files until all the child rows for some
    // batch have been flushed. This is to ensure the parent record does not become
    // visible until all its children are flushed.
    pending_file_flushes: Vec<FileWriter>,
    in_flight: FuturesUnordered<LocalBoxFuture<'static, FlushEvent>>,
    files_by_type: HashMap<ArrowPayloadType, usize>,
}

impl WriterManager {
    pub fn new(object_store: Arc<dyn ObjectStore>, options: WriterOptions) -> Self {
        Self {
            object_store,
            options,
            backend: None,
            close_failed: false,
            curr_writer_for_prefix: HashMap::new(),
            unflushed_batches_state: UnflushedBatchState::new(),
            pending_file_flushes: Vec::new(),
            in_flight: FuturesUnordered::new(),
            files_by_type: HashMap::new(),
        }
    }

    /// Replace only the file backend, retaining scheduling and dependency rules.
    pub fn with_backend(mut self, backend: Arc<dyn FileSinkFactory>) -> Self {
        self.backend = Some(backend);
        self
    }

    pub async fn write(&mut self, writes: &[WriteBatch<'_>]) -> Result<WriteStats, WriteError> {
        if self.close_failed {
            return Err(WriteError::from_flush_stats(
                FlushAttemptStats::default(),
                ParquetError::General(
                    "previous file close failed; writer must be abandoned".into(),
                ),
            ));
        }
        // The inbox only bypasses admission to drain its bounded shutdown backlog.
        // Make room for that one batch without growing the external file budget.
        let mut stats = if self.is_backpressured() {
            self.flush_all().await?
        } else {
            WriteStats::default()
        };
        for write in writes {
            // schedule the writes for each payload type for this signal
            for payload_type in write.otap_batch.allowed_payload_types() {
                if let Some(record_batch) = write.otap_batch.get(*payload_type) {
                    let created = self
                        .schedule_write_batch(
                            write.batch_id,
                            *payload_type,
                            record_batch,
                            write.partition_attributes,
                        )
                        .map_err(|error| WriteError { stats, error })?;
                    if created {
                        stats.files_created += 1;
                    }
                }
            }
        }

        // write the scheduled batches to the files
        let rows_written_total = match self.write_scheduled().await {
            Ok(rows_written) => rows_written,
            Err(error) => return Err(WriteError { stats, error }),
        };
        stats.rows_written += rows_written_total as u64;

        // if we can determine after the write process that any files should be flushed
        // (e.g. if they've exceeded max size), we'll try to flush them immediately
        //
        // Note: the files might not actually get flushed if they have child rows that
        // aren't scheduled to be flushed. In this case, we won't continue appending to
        // these files, but we won't flush them until after the children are flushed.
        let schedule_stats = self.schedule_flushes();
        stats.flush_scheduled_max_rows += schedule_stats.scheduled_max_rows;
        stats.flush_scheduled_max_age += schedule_stats.scheduled_max_age;
        if self.is_backpressured() {
            self.schedule_all();
        }

        let attempt_stats = match self.attempt_flush_scheduled().await {
            Ok(stats) => stats,
            Err(mut error) => {
                error.stats.files_created += stats.files_created;
                error.stats.rows_written += stats.rows_written;
                error.stats.flush_scheduled_max_rows += stats.flush_scheduled_max_rows;
                error.stats.flush_scheduled_max_age += stats.flush_scheduled_max_age;
                return Err(error);
            }
        };
        stats.files_closed += attempt_stats.files_closed;
        stats.flush_attempts += attempt_stats.flush_attempts;
        stats.flush_successes += attempt_stats.flush_successes;
        stats.flush_failures += attempt_stats.flush_failures;

        Ok(stats)
    }

    /// Write all the scheduled writes to the files concurrently, returning the total rows written
    async fn write_scheduled(&mut self) -> Result<usize, ParquetError> {
        let total_rows_written = self
            .curr_writer_for_prefix
            .values_mut()
            .map(|fw| fw.write_scheduled())
            .collect::<FuturesUnordered<_>>()
            .try_fold(0usize, |acc, n| async move { Ok(acc + n) })
            .await?;

        Ok(total_rows_written)
    }

    /// Schedules the record batch for a given signal to be written. If there's not already a file writer
    /// for the given partition/payload type, a new one will be created.
    ///
    /// Returns boolean true/false indicating whether a new file writer was created.
    fn schedule_write_batch(
        &mut self,
        batch_id: i64,
        payload_type: ArrowPayloadType,
        record_batch: &RecordBatch,
        partition_attributes: Option<&[PartitionAttribute]>,
    ) -> Result<bool, ParquetError> {
        if record_batch.num_rows() == 0 {
            return Ok(false);
        }
        let path_prefix = format!(
            "{}/{}",
            payload_type.as_str_name().to_lowercase(),
            compute_partition_path_prefix(partition_attributes),
        );

        // get the current writer for the path, or create a new one
        let mut created = false;
        let file_writer = match self.curr_writer_for_prefix.entry(path_prefix) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let full_path = format!("{}/{}", e.key(), generate_filename());
                created = true;
                e.insert(FileWriter::new(
                    payload_type,
                    match &self.backend {
                        Some(backend) => backend.create(&full_path, record_batch.schema())?,
                        None => Box::new(new_parquet_arrow_writer(
                            self.object_store.clone(),
                            record_batch.schema(),
                            full_path,
                        )),
                    },
                ))
            }
        };

        file_writer.schedule_write(record_batch);

        // track that we have written some unflushed data for this batch_id and payload_type
        if file_writer.batch_ids.insert(batch_id) {
            self.unflushed_batches_state
                .incr_unflushed_write(batch_id, payload_type);
        }
        if created && self.is_pipelined() {
            *self.files_by_type.entry(payload_type).or_default() += 1;
        }

        Ok(created)
    }

    fn schedule_flushes(&mut self) -> FlushScheduleStats {
        let mut scheduled_max_rows = 0u64;
        let mut scheduled_max_age = 0u64;

        // collect the keys of writers that should flush
        let keys_to_flush: Vec<(String, bool)> = self
            .curr_writer_for_prefix
            .iter()
            .filter_map(|(key, fw)| {
                if self.should_flush(fw) {
                    // Determine reason: rows vs age
                    let reason_rows = if let Some(target) = self.options.target_rows_per_file {
                        fw.rows_written >= target
                    } else {
                        false
                    };
                    Some((key.clone(), reason_rows))
                } else {
                    None
                }
            })
            .collect();

        // remove them from the map and move into scheduled_flushes
        for (key, reason_rows) in keys_to_flush {
            if let Some(writer) = self.curr_writer_for_prefix.remove(&key) {
                if reason_rows {
                    scheduled_max_rows += 1;
                } else {
                    scheduled_max_age += 1;
                }
                self.pending_file_flushes.push(writer);
            }
        }

        FlushScheduleStats {
            scheduled_max_rows,
            scheduled_max_age,
        }
    }

    fn should_flush(&self, file_writer: &FileWriter) -> bool {
        self.options
            .target_rows_per_file
            .is_some_and(|target| file_writer.rows_written >= target)
            || self
                .options
                .flush_when_older_than
                .is_some_and(|age| file_writer.created_at.elapsed() > age)
    }

    /// This method attempts to flush all the scheduled writers. It tries to flush child
    /// payload types before parent records. The motivation is to ensure is to ensure that
    /// parent records do not become visible before their child payloads. E.g. we don't want
    /// Logs to become visible before LogAttrs.
    ///
    /// This isn't guaranteed to flush everything that is scheduled, because there may be
    /// cases where a parent record has children files manager by writers that are not scheduled
    /// to be flushed.
    ///
    /// To force everything to flush, you can call `flush_all` method, which will schedule
    /// all the current writers for flushing and then flush them all.
    async fn attempt_flush_scheduled(&mut self) -> Result<FlushAttemptStats, WriteError> {
        if self.close_failed {
            return Err(WriteError::from_flush_stats(
                FlushAttemptStats::default(),
                ParquetError::General(
                    "previous file close failed; writer must be abandoned".into(),
                ),
            ));
        }
        if self.is_pipelined() {
            let attempts = self.start_staging();
            self.start_publications(false);
            return Ok(FlushAttemptStats {
                flush_attempts: attempts,
                ..Default::default()
            });
        }
        let mut flushable = Vec::new();
        let mut requeue = Vec::new();
        let mut stats = FlushAttemptStats::default();

        loop {
            for file_writer in self.pending_file_flushes.drain(..) {
                if self.unflushed_batches_state.has_unflushed_child(
                    file_writer.batch_ids.iter().copied(),
                    file_writer.payload_type,
                ) {
                    requeue.push(file_writer);
                } else {
                    flushable.push(file_writer);
                }
            }

            self.pending_file_flushes.append(&mut requeue);
            if flushable.is_empty() {
                break;
            }

            stats.flush_attempts += flushable.len() as u64;
            // Drain all in-flight closes so per-file success/failure metrics are complete.
            let mut closes = flushable
                .drain(..)
                .map(|fw| async move {
                    let result = fw.writer.close().await;
                    (result, fw.batch_ids, fw.payload_type)
                })
                .collect::<FuturesUnordered<_>>();
            let mut first_error = None;

            while let Some((result, batch_ids, payload_type)) = closes.next().await {
                match result {
                    Ok(_) => {
                        self.unflushed_batches_state
                            .decr_unflushed_write(batch_ids.into_iter(), payload_type);
                        stats.files_closed += 1;
                        stats.flush_successes += 1;
                    }
                    Err(error) => {
                        stats.flush_failures += 1;
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
            }

            if let Some(error) = first_error {
                self.close_failed = true;
                return Err(WriteError::from_flush_stats(stats, error));
            }
        }

        Ok(stats)
    }

    /// This method flushes all the current writers, ensuring that all files are closed and
    /// all data is written to the object store
    pub async fn flush_all(&mut self) -> Result<WriteStats, WriteError> {
        if self.close_failed {
            return Err(WriteError::from_flush_stats(
                FlushAttemptStats::default(),
                ParquetError::General(
                    "previous file completion failed; writer must be abandoned".into(),
                ),
            ));
        }
        self.schedule_all();
        if self.is_pipelined() {
            if let Some(backend) = &self.backend {
                backend.start_draining().map_err(|error| WriteError {
                    stats: WriteStats::default(),
                    error,
                })?;
            }
            let mut stats = WriteStats {
                flush_attempts: self.start_staging(),
                ..Default::default()
            };
            loop {
                self.start_publications(true);
                if self.in_flight.is_empty() {
                    if !self.pending_file_flushes.is_empty() {
                        return Err(WriteError {
                            stats,
                            error: ParquetError::General(
                                "file dependency drain made no progress".into(),
                            ),
                        });
                    }
                    return Ok(stats);
                }
                match self.progress().await {
                    Ok(completed) => {
                        stats.files_closed += completed.files_closed;
                        stats.flush_successes += completed.flush_successes;
                    }
                    Err(mut error) => {
                        error.stats.files_closed += stats.files_closed;
                        error.stats.flush_successes += stats.flush_successes;
                        error.stats.flush_attempts += stats.flush_attempts;
                        return Err(error);
                    }
                }
            }
        }
        let attempt_stats = self.attempt_flush_scheduled().await?;
        Ok(WriteStats {
            files_closed: attempt_stats.files_closed,
            flush_attempts: attempt_stats.flush_attempts,
            flush_successes: attempt_stats.flush_successes,
            flush_failures: attempt_stats.flush_failures,
            ..Default::default()
        })
    }

    /// If this [`WriterManager`] was configured with [`WriterOptions::flush_when_older_than`],
    /// then this method will flush any current writers with rows older than this threshold.
    pub async fn flush_aged_beyond_threshold(&mut self) -> Result<WriteStats, WriteError> {
        // schedule flushes -- this will put every writer whose age is older than the threshold
        // into the pending queue
        let schedule_stats = self.schedule_flushes();
        let attempt_stats = match self.attempt_flush_scheduled().await {
            Ok(stats) => stats,
            Err(mut error) => {
                error.stats.flush_scheduled_max_rows += schedule_stats.scheduled_max_rows;
                error.stats.flush_scheduled_max_age += schedule_stats.scheduled_max_age;
                return Err(error);
            }
        };
        Ok(WriteStats {
            files_created: 0,
            rows_written: 0,
            flush_scheduled_max_rows: schedule_stats.scheduled_max_rows,
            flush_scheduled_max_age: schedule_stats.scheduled_max_age,
            files_closed: attempt_stats.files_closed,
            flush_attempts: attempt_stats.flush_attempts,
            flush_successes: attempt_stats.flush_successes,
            flush_failures: attempt_stats.flush_failures,
        })
    }

    fn is_pipelined(&self) -> bool {
        self.backend
            .as_ref()
            .is_some_and(|backend| backend.max_pending_files().is_some())
    }

    /// Gate only pdata admission; the exporter must keep polling completion and control.
    pub fn is_backpressured(&self) -> bool {
        self.backend
            .as_ref()
            .and_then(|backend| backend.max_pending_files())
            .is_some_and(|limit| self.files_by_type.values().any(|count| *count >= limit))
    }

    fn schedule_all(&mut self) {
        self.pending_file_flushes.extend(
            self.curr_writer_for_prefix
                .drain()
                .map(|(_, writer)| writer),
        );
    }

    fn start_staging(&mut self) -> u64 {
        let mut staged = Vec::new();
        let mut attempts = 0;
        for mut file in self.pending_file_flushes.drain(..) {
            if file.staged {
                staged.push(file);
            } else {
                attempts += 1;
                self.in_flight.push(
                    async move {
                        let result = file.writer.stage().await;
                        file.staged = true;
                        FlushEvent::Staged(file, result)
                    }
                    .boxed_local(),
                );
            }
        }
        self.pending_file_flushes = staged;
        attempts
    }

    fn start_publications(&mut self, force: bool) {
        let mut blocked = Vec::new();
        for file in self.pending_file_flushes.drain(..) {
            if !file.staged
                || self
                    .unflushed_batches_state
                    .has_unflushed_child(file.batch_ids.iter().copied(), file.payload_type)
            {
                blocked.push(file);
            } else {
                self.in_flight.push(
                    async move {
                        let result = if force {
                            file.writer.close_forced().await
                        } else {
                            file.writer.close().await
                        };
                        FlushEvent::Closed {
                            batch_ids: file.batch_ids,
                            payload_type: file.payload_type,
                            result,
                        }
                    }
                    .boxed_local(),
                );
            }
        }
        self.pending_file_flushes = blocked;
    }

    /// Poll a local finalization/publication future. Remains pending when there is no work.
    pub async fn progress(&mut self) -> Result<WriteStats, WriteError> {
        if self.close_failed {
            return Err(WriteError::from_flush_stats(
                FlushAttemptStats::default(),
                ParquetError::General(
                    "previous file completion failed; writer must be abandoned".into(),
                ),
            ));
        }
        if self.in_flight.is_empty() {
            return std::future::pending().await;
        }
        let event = self.in_flight.next().await;
        let mut stats = WriteStats::default();
        let result = match event {
            Some(FlushEvent::Staged(file, result)) => {
                if result.is_ok() {
                    self.pending_file_flushes.push(file);
                }
                result
            }
            Some(FlushEvent::Closed {
                batch_ids,
                payload_type,
                result,
            }) => {
                if result.is_ok() {
                    self.unflushed_batches_state
                        .decr_unflushed_write(batch_ids.into_iter(), payload_type);
                    if let Some(count) = self.files_by_type.get_mut(&payload_type) {
                        *count -= 1;
                    }
                    stats.files_closed = 1;
                    stats.flush_successes = 1;
                }
                result
            }
            None => Ok(()),
        };
        if let Err(error) = result {
            self.close_failed = true;
            stats.flush_failures = 1;
            return Err(WriteError { stats, error });
        }
        self.start_publications(false);
        Ok(stats)
    }
}

/// generates a filename of the form: `part-<millis since epoch>-<random-uuid>.parquet`
fn generate_filename() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("expect now to be after unix epoch")
        .as_millis();
    format!("part-{}-{}.parquet", timestamp, uuid::Uuid::new_v4())
}

/// Computes the partition path prefix based on the provided partition attributes.
/// This does hive-style partitioning, where the path segments are formatted as `key=value`.
fn compute_partition_path_prefix(partition_attributes: Option<&[PartitionAttribute]>) -> String {
    match partition_attributes {
        Some(attributes) => {
            let mut path_segments = Vec::with_capacity(attributes.len());
            for attr in attributes {
                path_segments.push(format!("{}={}", attr.key, attr.value));
            }
            path_segments.join("/")
        }
        None => "".to_string(),
    }
}

/// Creates a new Parquet Arrow writer for the given record batch's schema and full path.
fn new_parquet_arrow_writer(
    object_store: Arc<dyn ObjectStore>,
    schema: SchemaRef,
    full_path: String,
) -> AsyncArrowWriter<ParquetObjectWriter> {
    let object_writer = ParquetObjectWriter::new(object_store, full_path.into());
    AsyncArrowWriter::try_new(object_writer, schema, Some(WriterProperties::default()))
        .expect("Failed to create AsyncArrowWriter")
}

struct FileWriter {
    created_at: Instant,
    batch_ids: BTreeSet<i64>,
    payload_type: ArrowPayloadType,
    writer: Box<dyn FileSink>,
    rows_written: usize,
    scheduled_batches: Vec<RecordBatch>,
    staged: bool,
}

impl FileWriter {
    fn new(payload_type: ArrowPayloadType, writer: Box<dyn FileSink>) -> Self {
        Self {
            created_at: Instant::now(),
            batch_ids: BTreeSet::new(),
            payload_type,
            writer,
            rows_written: 0,
            scheduled_batches: Vec::new(),
            staged: false,
        }
    }

    fn schedule_write(&mut self, record_batch: &RecordBatch) {
        self.scheduled_batches.push(record_batch.clone());
    }

    async fn write_scheduled(&mut self) -> Result<usize, ParquetError> {
        let drained_batches = std::mem::take(&mut self.scheduled_batches);
        let mut rows = 0usize;
        for batch in drained_batches {
            rows += batch.num_rows();
            self.write(batch).await?;
        }

        Ok(rows)
    }

    async fn write(&mut self, record_batch: RecordBatch) -> Result<(), ParquetError> {
        self.writer.write(&record_batch).await?;
        self.rows_written += record_batch.num_rows();

        Ok(())
    }
}

struct UnflushedBatchState {
    // map keyed by (batch_id, payload_type) to values which are the number of files
    // currently being written that have not been flushed
    unflushed_writes: HashMap<(i64, ArrowPayloadType), usize>,
}

impl UnflushedBatchState {
    fn new() -> Self {
        Self {
            unflushed_writes: HashMap::new(),
        }
    }

    fn incr_unflushed_write(&mut self, batch_id: i64, payload_type: ArrowPayloadType) {
        let key = (batch_id, payload_type);
        let count = self.unflushed_writes.entry(key).or_insert(0);
        *count += 1;
    }

    fn decr_unflushed_write<T>(&mut self, batch_ids: T, payload_type: ArrowPayloadType)
    where
        T: Iterator<Item = i64>,
    {
        for batch_id in batch_ids {
            // Remove the unflushed write for the given batch_id and payload_type
            let key = (batch_id, payload_type);
            if let Some(count) = self.unflushed_writes.get_mut(&key) {
                if *count > 0 {
                    *count -= 1;
                }
                if *count == 0 {
                    _ = self.unflushed_writes.remove(&key);
                }
            }
        }
    }

    fn has_unflushed_child<T>(&self, batch_ids: T, payload_type: ArrowPayloadType) -> bool
    where
        T: Iterator<Item = i64>,
    {
        for batch_id in batch_ids {
            let has_unflushed = child_payload_types(payload_type)
                .iter()
                .any(|child_payload_type| self.has_unflushed_write(batch_id, *child_payload_type));
            if has_unflushed {
                return true;
            }
        }

        false
    }

    fn has_unflushed_write(&self, batch_id: i64, payload_type: ArrowPayloadType) -> bool {
        self.unflushed_writes
            .get(&(batch_id, payload_type))
            .is_some_and(|&count| count > 0)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use arrow::compute::concat_batches;
    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::StreamExt;
    use object_store::local::LocalFileSystem;
    use object_store::path::Path;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
        PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result,
    };
    use otel_arrow_dfe_pdata::otap::{OtapArrowRecords, from_record_messages};
    use otel_arrow_dfe_pdata::{Consumer, proto::opentelemetry::arrow::v1::BatchArrowRecords};
    use parquet::arrow::ParquetRecordBatchStreamBuilder;
    use std::fmt;
    use std::ops::Range;
    use tokio::fs::File;

    use crate::exporters::parquet_exporter::fixtures::{
        SimpleDataGenOptions, create_simple_logs_arrow_record_batches,
    };
    use crate::exporters::parquet_exporter::partition::PartitionAttributeValue;
    use crate::exporters::parquet_exporter::records::OtapParquetRecords;

    struct RecordingFactory {
        closed: Arc<std::sync::Mutex<Vec<String>>>,
        fail_children: bool,
    }

    struct RecordingSink {
        name: String,
        closed: Arc<std::sync::Mutex<Vec<String>>>,
        fail: bool,
    }

    #[async_trait(?Send)]
    impl FileSink for RecordingSink {
        async fn write(&mut self, _batch: &RecordBatch) -> Result<(), ParquetError> {
            Ok(())
        }

        async fn close(self: Box<Self>) -> Result<(), ParquetError> {
            if self.fail {
                return Err(ParquetError::General(
                    "injected child commit failure".into(),
                ));
            }
            self.closed
                .lock()
                .map_err(|_| ParquetError::General("test lock poisoned".into()))?
                .push(self.name.clone());
            Ok(())
        }
    }

    impl FileSinkFactory for RecordingFactory {
        fn create(
            &self,
            path: &str,
            _schema: SchemaRef,
        ) -> Result<Box<dyn FileSink>, ParquetError> {
            Ok(Box::new(RecordingSink {
                name: path.to_string(),
                closed: Arc::clone(&self.closed),
                fail: self.fail_children && path.starts_with("log_attrs/"),
            }))
        }
    }

    fn backend_records() -> OtapParquetRecords {
        use arrow::array::UInt32Array;
        use arrow::datatypes::{DataType, Field, Schema};
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::UInt32, false)])),
            vec![Arc::new(UInt32Array::from(vec![1]))],
        )
        .unwrap();
        let mut store = otel_arrow_dfe_pdata::otap::raw_batch_store::RawLogsStore::new();
        store.set(ArrowPayloadType::Logs, batch.clone());
        store.set(ArrowPayloadType::LogAttrs, batch);
        OtapParquetRecords::Logs(store)
    }

    struct PipelineFactory {
        staged: Arc<std::sync::Mutex<Vec<String>>>,
        closed: Arc<std::sync::Mutex<Vec<String>>>,
        gates: Arc<tokio::sync::Semaphore>,
        draining: Arc<std::sync::atomic::AtomicBool>,
        fail_children: bool,
    }

    struct PipelineSink {
        name: String,
        staged: Arc<std::sync::Mutex<Vec<String>>>,
        closed: Arc<std::sync::Mutex<Vec<String>>>,
        gates: Arc<tokio::sync::Semaphore>,
        draining: Arc<std::sync::atomic::AtomicBool>,
        fail: bool,
    }

    #[async_trait(?Send)]
    impl FileSink for PipelineSink {
        async fn write(&mut self, _batch: &RecordBatch) -> Result<(), ParquetError> {
            Ok(())
        }

        async fn stage(&mut self) -> Result<(), ParquetError> {
            self.staged.lock().unwrap().push(self.name.clone());
            Ok(())
        }

        async fn close(self: Box<Self>) -> Result<(), ParquetError> {
            if self.name.starts_with("log_attrs/") {
                if !self.draining.load(std::sync::atomic::Ordering::Acquire) {
                    self.gates.acquire().await.unwrap().forget();
                }
                if self.fail {
                    return Err(ParquetError::General("injected publication failure".into()));
                }
            }
            self.closed.lock().unwrap().push(self.name.clone());
            Ok(())
        }
    }

    impl FileSinkFactory for PipelineFactory {
        fn create(
            &self,
            path: &str,
            _schema: SchemaRef,
        ) -> Result<Box<dyn FileSink>, ParquetError> {
            Ok(Box::new(PipelineSink {
                name: path.to_string(),
                staged: Arc::clone(&self.staged),
                closed: Arc::clone(&self.closed),
                gates: Arc::clone(&self.gates),
                draining: Arc::clone(&self.draining),
                fail: self.fail_children,
            }))
        }

        fn max_pending_files(&self) -> Option<usize> {
            Some(2)
        }

        fn start_draining(&self) -> Result<(), ParquetError> {
            self.draining
                .store(true, std::sync::atomic::Ordering::Release);
            self.gates.add_permits(2);
            Ok(())
        }
    }

    fn pipeline_factory(fail_children: bool) -> Arc<PipelineFactory> {
        Arc::new(PipelineFactory {
            staged: Arc::new(std::sync::Mutex::new(Vec::new())),
            closed: Arc::new(std::sync::Mutex::new(Vec::new())),
            gates: Arc::new(tokio::sync::Semaphore::new(0)),
            draining: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            fail_children,
        })
    }

    /// Scenario: child publications stall while another batch arrives on the same core.
    /// Guarantees: private parent/child finalization is concurrent, ingestion continues up to its bound, and parents wait for publication.
    #[tokio::test]
    async fn pipeline_writes_while_commits_wait_and_bounds_pending_files() {
        let factory = pipeline_factory(false);
        let mut manager = WriterManager::new(
            Arc::new(object_store::memory::InMemory::new()),
            WriterOptions {
                target_rows_per_file: Some(1),
                flush_when_older_than: None,
            },
        )
        .with_backend(factory.clone());
        let records = backend_records();
        let _ = manager
            .write(&[WriteBatch::new(0, &records, None)])
            .await
            .unwrap();
        let stats = manager
            .write(&[WriteBatch::new(1, &records, None)])
            .await
            .unwrap();
        assert_eq!(stats.rows_written, 2);
        assert!(manager.is_backpressured());
        for _ in 0..4 {
            let _ = manager.progress().await.unwrap();
        }
        assert_eq!(factory.staged.lock().unwrap().len(), 4);
        assert!(factory.closed.lock().unwrap().is_empty());
        factory.gates.add_permits(2);
        for _ in 0..4 {
            let _ = manager.progress().await.unwrap();
        }
        assert!(!manager.is_backpressured());
        let closed = factory.closed.lock().unwrap();
        assert_eq!(closed.len(), 4);
        assert!(closed[0].starts_with("log_attrs/"));
    }

    /// Scenario: shutdown begins while all staged files still await a long collection window.
    /// Guarantees: every dependency wave is forced and completion includes parent publication.
    #[tokio::test]
    async fn pipeline_shutdown_forces_all_dependency_waves() {
        let factory = pipeline_factory(false);
        let mut manager = WriterManager::new(
            Arc::new(object_store::memory::InMemory::new()),
            WriterOptions {
                target_rows_per_file: Some(1),
                flush_when_older_than: None,
            },
        )
        .with_backend(factory.clone());
        let _ = manager
            .write(&[WriteBatch::new(0, &backend_records(), None)])
            .await
            .unwrap();
        let stats = tokio::time::timeout(std::time::Duration::from_secs(1), manager.flush_all())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stats.files_closed, 2);
        let closed = factory.closed.lock().unwrap();
        assert!(closed[0].starts_with("log_attrs/"));
        assert!(closed[1].starts_with("logs/"));
    }

    /// Scenario: an asynchronous child publication fails after private parent finalization.
    /// Guarantees: failure reaches the exporter and no subsequent parent or successful drain is reported.
    #[tokio::test]
    async fn pipeline_child_failure_poisoning_blocks_parents() {
        let factory = pipeline_factory(true);
        let mut manager = WriterManager::new(
            Arc::new(object_store::memory::InMemory::new()),
            WriterOptions {
                target_rows_per_file: Some(1),
                flush_when_older_than: None,
            },
        )
        .with_backend(factory.clone());
        let _ = manager
            .write(&[WriteBatch::new(0, &backend_records(), None)])
            .await
            .unwrap();
        assert!(manager.flush_all().await.is_err());
        assert!(manager.flush_all().await.is_err());
        assert!(factory.closed.lock().unwrap().is_empty());
    }

    /// Scenario: a transactional backend publishes child and parent payload files.
    /// Guarantees: parent completion occurs only after child completion succeeds.
    #[tokio::test]
    async fn injected_backend_preserves_visibility_order() {
        let closed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let backend = Arc::new(RecordingFactory {
            closed: Arc::clone(&closed),
            fail_children: false,
        });
        let mut manager = WriterManager::new(
            Arc::new(object_store::memory::InMemory::new()),
            WriterOptions {
                target_rows_per_file: Some(1),
                flush_when_older_than: None,
            },
        )
        .with_backend(backend);
        let records = backend_records();
        let _ = manager
            .write(&[WriteBatch::new(0, &records, None)])
            .await
            .unwrap();
        let closed = closed.lock().unwrap();
        assert_eq!(closed.len(), 2);
        assert!(closed[0].starts_with("log_attrs/"));
        assert!(closed[1].starts_with("logs/"));
    }

    /// Scenario: a child's transactional file completion fails.
    /// Guarantees: the exporter returns an error and does not publish the parent.
    #[tokio::test]
    async fn injected_child_failure_blocks_parent() {
        let closed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let backend = Arc::new(RecordingFactory {
            closed: Arc::clone(&closed),
            fail_children: true,
        });
        let mut manager = WriterManager::new(
            Arc::new(object_store::memory::InMemory::new()),
            WriterOptions {
                target_rows_per_file: Some(1),
                flush_when_older_than: None,
            },
        )
        .with_backend(backend);
        let records = backend_records();
        assert!(
            manager
                .write(&[WriteBatch::new(0, &records, None)])
                .await
                .is_err()
        );
        assert!(manager.flush_all().await.is_err());
        assert!(closed.lock().unwrap().is_empty());
    }

    /// Scenario: row and age thresholds are configured together for a small file.
    /// Guarantees: the age threshold still flushes files below the row target.
    #[test]
    fn age_threshold_is_independent_of_row_threshold() {
        let backend = RecordingSink {
            name: String::new(),
            closed: Arc::new(std::sync::Mutex::new(Vec::new())),
            fail: false,
        };
        let mut file = FileWriter::new(ArrowPayloadType::Logs, Box::new(backend));
        file.created_at = Instant::now() - std::time::Duration::from_secs(10);
        file.rows_written = 1;
        let manager = WriterManager::new(
            Arc::new(object_store::memory::InMemory::new()),
            WriterOptions {
                target_rows_per_file: Some(1_000),
                flush_when_older_than: Some(std::time::Duration::from_secs(2)),
            },
        );
        assert!(manager.should_flush(&file));
    }

    fn to_logs_record_batch(mut bar: BatchArrowRecords) -> OtapParquetRecords {
        let mut consumer = Consumer::default();
        let record_messages = consumer.consume_bar(&mut bar).unwrap();
        let otap: OtapArrowRecords =
            OtapArrowRecords::Logs(from_record_messages(record_messages).unwrap());
        otap.into()
    }

    fn sum_rows(batch: &OtapParquetRecords) -> u64 {
        batch
            .allowed_payload_types()
            .iter()
            .filter_map(|pt| batch.get(*pt).map(|rb| rb.num_rows() as u64))
            .sum()
    }

    #[derive(Debug)]
    struct FailPutForPrefixStore {
        inner: LocalFileSystem,
        prefixes: &'static [&'static str],
    }

    impl fmt::Display for FailPutForPrefixStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "FailPutForPrefixStore")
        }
    }

    #[async_trait]
    impl ObjectStore for FailPutForPrefixStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> Result<PutResult> {
            if self
                .prefixes
                .iter()
                .any(|prefix| location.as_ref().starts_with(prefix))
            {
                return Err(object_store::Error::Generic {
                    store: "fail_put_for_prefix",
                    source: Box::new(std::io::Error::other("injected put failure")),
                });
            }
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> Result<Box<dyn MultipartUpload>> {
            if self
                .prefixes
                .iter()
                .any(|prefix| location.as_ref().starts_with(prefix))
            {
                return Err(object_store::Error::Generic {
                    store: "fail_put_for_prefix",
                    source: Box::new(std::io::Error::other("injected put failure")),
                });
            }
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
            self.inner.get_opts(location, options).await
        }

        async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
            self.inner.get_ranges(location, ranges).await
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<'static, Result<Path>>,
        ) -> futures::stream::BoxStream<'static, Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> futures::stream::BoxStream<'static, Result<ObjectMeta>> {
            self.inner.list(prefix)
        }

        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> futures::stream::BoxStream<'static, Result<ObjectMeta>> {
            self.inner.list_with_offset(prefix, offset)
        }

        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
            self.inner.copy_opts(from, to, options).await
        }

        async fn rename_opts(&self, from: &Path, to: &Path, options: RenameOptions) -> Result<()> {
            self.inner.rename_opts(from, to, options).await
        }
    }

    #[tokio::test]
    async fn test_simple_single_batch_write_all_logs() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path();
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(path).unwrap());
        let mut writer = WriterManager::new(object_store, WriterOptions::default());

        // write some batch:
        let otap_batch = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions::default(),
        ));
        let stats = writer
            .write(&[WriteBatch::new(0, &otap_batch, None)])
            .await
            .unwrap();
        let expected_rows = sum_rows(&otap_batch);
        assert_eq!(stats.rows_written, expected_rows);

        // check that the files aren't flushed
        let mut files = Vec::new();
        let mut read_dir_stream = tokio::fs::read_dir(path).await.unwrap();
        while let Some(entry) = read_dir_stream.next_entry().await.unwrap() {
            files.push(entry)
        }

        assert!(files.is_empty());

        // flush the files
        let _ = writer.flush_all().await.unwrap();

        // check that we've written the file for each batch and it contains the correct content
        for payload_type in [
            ArrowPayloadType::Logs,
            ArrowPayloadType::LogAttrs,
            ArrowPayloadType::ResourceAttrs,
            ArrowPayloadType::ScopeAttrs,
        ] {
            let table_name = payload_type.as_str_name().to_lowercase();
            let mut files = Vec::new();
            let mut read_dir_stream =
                tokio::fs::read_dir(format!("{}/{}", path.to_string_lossy(), table_name))
                    .await
                    .unwrap();
            while let Some(entry) = read_dir_stream.next_entry().await.unwrap() {
                files.push(entry.path().to_string_lossy().to_string())
            }

            // we should have written one file
            assert_eq!(files.len(), 1);

            // read the file and ensure it's the equivalent data from the original batch
            let original_record_batch = otap_batch.get(payload_type).unwrap();
            let file = File::open(files[0].clone()).await.unwrap();
            let reader_builder = ParquetRecordBatchStreamBuilder::new(file).await.unwrap();
            let mut reader = reader_builder.build().unwrap();
            let read_batch = reader.next().await.unwrap().unwrap();
            assert_eq!(&read_batch, original_record_batch);

            // assert there's no extra data there
            assert!(reader.next().await.is_none())
        }
    }

    #[tokio::test]
    async fn test_flush_stats_count_attempts_successes_and_failures() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path();
        let object_store = Arc::new(FailPutForPrefixStore {
            inner: LocalFileSystem::new_with_prefix(path).unwrap(),
            prefixes: &["logs/"],
        });
        let mut writer = WriterManager::new(object_store, WriterOptions::default());

        let otap_batch = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions::default(),
        ));
        let _ = writer
            .write(&[WriteBatch::new(0, &otap_batch, None)])
            .await
            .unwrap();

        let err = writer.flush_all().await.unwrap_err();
        assert_eq!(err.stats.flush_attempts, 4);
        assert_eq!(err.stats.flush_successes, 3);
        assert_eq!(err.stats.flush_failures, 1);
        assert_eq!(err.stats.files_closed, 3);
        assert!(err.to_string().contains("injected put failure"));
    }

    #[tokio::test]
    async fn test_flush_stats_count_multiple_concurrent_failures() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path();
        let object_store = Arc::new(FailPutForPrefixStore {
            inner: LocalFileSystem::new_with_prefix(path).unwrap(),
            prefixes: &["log_attrs/", "resource_attrs/"],
        });
        let mut writer = WriterManager::new(object_store, WriterOptions::default());

        let otap_batch = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions::default(),
        ));
        let _ = writer
            .write(&[WriteBatch::new(0, &otap_batch, None)])
            .await
            .unwrap();

        let err = writer.flush_all().await.unwrap_err();
        assert_eq!(err.stats.flush_attempts, 3);
        assert_eq!(err.stats.flush_successes, 1);
        assert_eq!(err.stats.flush_failures, 2);
        assert_eq!(err.stats.files_closed, 1);
        assert!(err.to_string().contains("injected put failure"));
    }

    #[tokio::test]
    async fn test_simple_multi_batch_write_all_logs() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path();
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(path).unwrap());
        let mut writer = WriterManager::new(object_store, WriterOptions::default());

        let batch1 = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions {
                id_offset: 0,
                ..Default::default()
            },
        ));

        let batch2 = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions {
                id_offset: 1,
                ..Default::default()
            },
        ));

        let stats = writer
            .write(&[
                WriteBatch::new(0, &batch1, None),
                WriteBatch::new(1, &batch2, None),
            ])
            .await
            .unwrap();
        let expected_rows = sum_rows(&batch1) + sum_rows(&batch2);
        assert_eq!(stats.rows_written, expected_rows);

        let _ = writer.flush_all().await.unwrap();

        for payload_type in [
            ArrowPayloadType::Logs,
            ArrowPayloadType::LogAttrs,
            ArrowPayloadType::ResourceAttrs,
            ArrowPayloadType::ScopeAttrs,
        ] {
            let table_name = payload_type.as_str_name().to_lowercase();
            let mut files = Vec::new();
            let mut read_dir_stream =
                tokio::fs::read_dir(format!("{}/{}", path.to_string_lossy(), table_name))
                    .await
                    .unwrap();
            while let Some(entry) = read_dir_stream.next_entry().await.unwrap() {
                files.push(entry.path().to_string_lossy().to_string())
            }

            // we should have written one file
            assert_eq!(files.len(), 1);

            // read the file and ensure it's the equivalent data from the original batch
            let original_batch1 = batch1.get(payload_type).unwrap();
            let original_batch2 = batch2.get(payload_type).unwrap();
            let expected_batch = concat_batches(
                original_batch1.schema_ref(),
                vec![original_batch1, original_batch2],
            )
            .unwrap();
            let file = File::open(files[0].clone()).await.unwrap();
            let builder = ParquetRecordBatchStreamBuilder::new(file).await.unwrap();
            let mut reader = builder.build().unwrap();
            let read_batch = reader.next().await.unwrap().unwrap();
            assert_eq!(&read_batch, &expected_batch);

            // assert there's no extra data there
            assert!(reader.next().await.is_none())
        }
    }

    #[tokio::test]
    async fn test_partition_write_all_logs() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path();
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(path).unwrap());
        let mut writer = WriterManager::new(object_store, WriterOptions::default());

        let partition1_batch = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions {
                id_offset: 0,
                ..Default::default()
            },
        ));

        let partition1_attrs = vec![
            PartitionAttribute {
                key: "key1".to_string(),
                value: PartitionAttributeValue::String("valA".to_string()),
            },
            PartitionAttribute {
                key: "key2".to_string(),
                value: PartitionAttributeValue::String("1".to_string()),
            },
        ];

        let partition2_batch = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions {
                id_offset: 1,
                ..Default::default()
            },
        ));

        let partition2_attrs = vec![
            PartitionAttribute {
                key: "key1".to_string(),
                value: PartitionAttributeValue::String("valB".to_string()),
            },
            PartitionAttribute {
                key: "key2".to_string(),
                value: PartitionAttributeValue::String("1".to_string()),
            },
        ];

        let stats = writer
            .write(&[
                WriteBatch::new(0, &partition1_batch, Some(partition1_attrs.as_slice())),
                WriteBatch::new(1, &partition2_batch, Some(partition2_attrs.as_slice())),
            ])
            .await
            .unwrap();
        let expected_rows = sum_rows(&partition1_batch) + sum_rows(&partition2_batch);
        assert_eq!(stats.rows_written, expected_rows);

        // write all the files
        let _ = writer.flush_all().await.unwrap();

        let test_cases = vec![
            // original_batch, expected_partition_prefix
            (partition1_batch, "key1=valA/key2=1"),
            (partition2_batch, "key1=valB/key2=1"),
        ];

        for test_case in test_cases {
            let otap_batch = test_case.0;
            let partition_prefix = test_case.1;

            // check that we've written the file for each batch and it contains the correct content
            for payload_type in [
                ArrowPayloadType::Logs,
                ArrowPayloadType::LogAttrs,
                ArrowPayloadType::ResourceAttrs,
                ArrowPayloadType::ScopeAttrs,
            ] {
                let table_name = payload_type.as_str_name().to_lowercase();
                let dir = format!(
                    "{}/{}/{}",
                    path.to_string_lossy(),
                    table_name,
                    partition_prefix,
                );
                let mut files = Vec::new();
                let mut read_dir_stream = tokio::fs::read_dir(dir).await.unwrap();
                while let Some(entry) = read_dir_stream.next_entry().await.unwrap() {
                    files.push(entry.path().to_string_lossy().to_string());
                }

                // we should have written one file
                assert_eq!(files.len(), 1);

                // read the file and ensure it's the equivalent data from the original batch
                let original_record_batch = otap_batch.get(payload_type).unwrap();
                let file = File::open(files[0].clone()).await.unwrap();
                let builder = ParquetRecordBatchStreamBuilder::new(file).await.unwrap();
                let mut reader = builder.build().unwrap();
                let read_batch = reader.next().await.unwrap().unwrap();
                assert_eq!(&read_batch, original_record_batch);

                // assert there's no extra data there
                assert!(reader.next().await.is_none())
            }
        }
    }

    #[tokio::test]
    async fn test_auto_flushes_when_max_rows_exceeded() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path();
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(path).unwrap());
        let mut writer = WriterManager::new(
            object_store,
            WriterOptions {
                target_rows_per_file: Some(2),
                ..Default::default()
            },
        );

        let batch1 = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions {
                id_offset: 0,
                ..Default::default()
            },
        ));

        let batch2 = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions {
                id_offset: 1,
                ..Default::default()
            },
        ));

        let stats = writer
            .write(&[
                WriteBatch::new(0, &batch1, None),
                WriteBatch::new(1, &batch2, None),
            ])
            .await
            .unwrap();
        let expected_rows = sum_rows(&batch1) + sum_rows(&batch2);
        assert_eq!(stats.rows_written, expected_rows);
        assert!(stats.flush_scheduled_max_rows > 0);

        for payload_type in [
            ArrowPayloadType::Logs,
            ArrowPayloadType::LogAttrs,
            ArrowPayloadType::ResourceAttrs,
            ArrowPayloadType::ScopeAttrs,
        ] {
            let table_name = payload_type.as_str_name().to_lowercase();
            let mut files = Vec::new();
            let mut read_dir_stream =
                tokio::fs::read_dir(format!("{}/{}", path.to_string_lossy(), table_name))
                    .await
                    .unwrap();
            while let Some(entry) = read_dir_stream.next_entry().await.unwrap() {
                files.push(entry.path().to_string_lossy().to_string())
            }

            // we should have written one file
            assert_eq!(files.len(), 1);

            // read the file and ensure it's the equivalent data from the original batch
            let original_batch1 = batch1.get(payload_type).unwrap();
            let original_batch2 = batch2.get(payload_type).unwrap();
            let expected_batch = concat_batches(
                original_batch1.schema_ref(),
                vec![original_batch1, original_batch2],
            )
            .unwrap();
            let file = File::open(files[0].clone()).await.unwrap();
            let builder = ParquetRecordBatchStreamBuilder::new(file).await.unwrap();
            let mut reader = builder.build().unwrap();
            let read_batch = reader.next().await.unwrap().unwrap();
            assert_eq!(&read_batch, &expected_batch);

            // assert there's no extra data there
            assert!(reader.next().await.is_none())
        }
    }

    #[tokio::test]
    async fn test_doesnt_autoflush_parent_batch_if_children_not_flushed() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path();
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(path).unwrap());
        let mut writer = WriterManager::new(
            object_store,
            WriterOptions {
                target_rows_per_file: Some(2),
                ..Default::default()
            },
        );

        let batch1 = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions {
                id_offset: 0,
                ..Default::default()
            },
        ));

        let batch2 = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions {
                id_offset: 1,
                with_main_record_attrs: false,
                ..Default::default()
            },
        ));
        let stats = writer
            .write(&[
                WriteBatch::new(0, &batch1, None),
                WriteBatch::new(1, &batch2, None),
            ])
            .await
            .unwrap();
        let expected_rows = sum_rows(&batch1) + sum_rows(&batch2);
        assert_eq!(stats.rows_written, expected_rows);

        // at this point resource & scope attributes should have flushed, but the
        // log attributes won't have (because there's only one buffered log record).
        // Also the logs won't have flushed, because there's one batch whose child
        // (log_attrs) hasn't flushed

        for payload_type in [ArrowPayloadType::Logs, ArrowPayloadType::LogAttrs] {
            let table_name = payload_type.as_str_name().to_lowercase();
            let table_exists =
                tokio::fs::try_exists(format!("{}/{}", path.to_string_lossy(), table_name))
                    .await
                    .unwrap();
            assert!(!table_exists);
        }

        for payload_type in [
            ArrowPayloadType::ResourceAttrs,
            ArrowPayloadType::ScopeAttrs,
        ] {
            let table_name = payload_type.as_str_name().to_lowercase();
            let mut files = Vec::new();
            let mut read_dir_stream =
                tokio::fs::read_dir(format!("{}/{}", path.to_string_lossy(), table_name))
                    .await
                    .unwrap();
            while let Some(entry) = read_dir_stream.next_entry().await.unwrap() {
                files.push(entry.path().to_string_lossy().to_string())
            }

            // we should have written one file
            assert_eq!(files.len(), 1);

            // read the file and ensure it's the equivalent data from the original batch
            let original_batch1 = batch1.get(payload_type).unwrap();
            let original_batch2 = batch2.get(payload_type).unwrap();
            let expected_batch = concat_batches(
                original_batch1.schema_ref(),
                vec![original_batch1, original_batch2],
            )
            .unwrap();
            let file = File::open(files[0].clone()).await.unwrap();
            let builder = ParquetRecordBatchStreamBuilder::new(file).await.unwrap();
            let mut reader = builder.build().unwrap();
            let read_batch = reader.next().await.unwrap().unwrap();
            assert_eq!(&read_batch, &expected_batch);

            // assert there's no extra data there
            assert!(reader.next().await.is_none())
        }
    }

    #[tokio::test]
    async fn test_doesnt_autoflush_parent_batch_if_children_not_flushed_written_reverse_write_order()
     {
        // This is similar to the test above, but the child records are written after
        // the writer for the parent is already created.

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path();
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(path).unwrap());
        let mut writer = WriterManager::new(
            object_store,
            WriterOptions {
                target_rows_per_file: Some(2),
                ..Default::default()
            },
        );

        let batch0 = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions {
                id_offset: 0,
                num_rows: 1,
                with_resource_attrs: false,
                with_scope_attrs: false,
                with_main_record_attrs: false,
                ..Default::default()
            },
        ));

        let batch1 = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions {
                id_offset: 1,
                with_main_record_attrs: true,
                with_scope_attrs: false,
                with_resource_attrs: false,
                ..Default::default()
            },
        ));
        let stats0 = writer
            .write(&[WriteBatch::new(0, &batch0, None)])
            .await
            .unwrap();
        assert_eq!(stats0.rows_written, sum_rows(&batch0));
        let stats1 = writer
            .write(&[WriteBatch::new(1, &batch1, None)])
            .await
            .unwrap();
        assert_eq!(stats1.rows_written, sum_rows(&batch1));

        // At this point, the log writer has enough rows to flush but there are rows in this file
        // that are associated with log attributes, whose writer doesn't have enough rows to flush

        for payload_type in [ArrowPayloadType::Logs, ArrowPayloadType::LogAttrs] {
            let table_name = payload_type.as_str_name().to_lowercase();
            let table_exists =
                tokio::fs::try_exists(format!("{}/{}", path.to_string_lossy(), table_name))
                    .await
                    .unwrap();
            assert!(!table_exists);
        }

        let batch2 = to_logs_record_batch(create_simple_logs_arrow_record_batches(
            SimpleDataGenOptions {
                id_offset: 1,
                with_main_record_attrs: true,
                with_scope_attrs: false,
                with_resource_attrs: false,
                ..Default::default()
            },
        ));
        let stats2 = writer
            .write(&[WriteBatch::new(2, &batch2, None)])
            .await
            .unwrap();
        assert_eq!(stats2.rows_written, sum_rows(&batch2));

        // now enough rows are in the attrs record batch that it can be written
        for payload_type in [ArrowPayloadType::Logs, ArrowPayloadType::LogAttrs] {
            let table_name = payload_type.as_str_name().to_lowercase();
            let mut files = Vec::new();
            let mut read_dir_stream =
                tokio::fs::read_dir(format!("{}/{}", path.to_string_lossy(), table_name))
                    .await
                    .unwrap();
            while let Some(entry) = read_dir_stream.next_entry().await.unwrap() {
                files.push(entry.path().to_string_lossy().to_string())
            }

            // read the file and ensure it's the equivalent data from the original batch
            let mut batches = vec![];
            if let Some(original) = batch0.get(payload_type) {
                batches.push(original);
            }
            if let Some(original) = batch1.get(payload_type) {
                batches.push(original);
            }

            // the attributes from the last batch will be written to the first file
            if payload_type == ArrowPayloadType::LogAttrs {
                batches.push(batch2.get(payload_type).unwrap());
            }

            // let original_batch2 = batch1.get(payload_type).unwrap();
            let expected_batch = concat_batches(batches[0].schema_ref(), batches).unwrap();
            let file = File::open(files[0].clone()).await.unwrap();
            let builder = ParquetRecordBatchStreamBuilder::new(file).await.unwrap();
            let mut reader = builder.build().unwrap();
            let read_batch = reader.next().await.unwrap().unwrap();
            assert_eq!(&read_batch, &expected_batch);

            // assert there's no extra data there
            assert!(reader.next().await.is_none())
        }
    }
}
