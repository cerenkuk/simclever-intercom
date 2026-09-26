"""Transkripsiyon çalışması ve zaman damgalı segmentler.

Revision ID: 0003_transcription_segments
Revises: 0002_track_media_metadata
"""
from alembic import op
import sqlalchemy as sa


revision = "0003_transcription_segments"
down_revision = "0002_track_media_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "transcription_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("source_track_id", sa.String(36), nullable=False),
        sa.Column("output_track_id", sa.String(36), nullable=False),
        sa.Column("engine", sa.String(80), nullable=False),
        sa.Column("model", sa.String(160), nullable=False),
        sa.Column("engine_version", sa.String(80), nullable=False),
        sa.Column("config_version", sa.String(80), nullable=False),
        sa.Column("language", sa.String(16), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(9), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_track_id"], ["tracks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["output_track_id"], ["tracks.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_transcription_runs_source_track_id", "transcription_runs", ["source_track_id"])
    op.create_index("ix_transcription_runs_output_track_id", "transcription_runs", ["output_track_id"])
    op.create_table(
        "transcript_segments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("track_id", sa.String(36), nullable=False),
        sa.Column("run_id", sa.String(36), nullable=False),
        sa.Column("segment_ref", sa.String(16), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float()),
        sa.Column("asr_engine", sa.String(80), nullable=False),
        sa.Column("asr_model", sa.String(160), nullable=False),
        sa.Column("asr_version", sa.String(80), nullable=False),
        sa.Column("language", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["track_id"], ["tracks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["transcription_runs.id"], ondelete="CASCADE"),
        sa.CheckConstraint("sequence_no >= 1", name="ck_segment_sequence_positive"),
        sa.CheckConstraint("start_ms >= 0", name="ck_segment_start_nonnegative"),
        sa.CheckConstraint("end_ms >= start_ms", name="ck_segment_end_after_start"),
        sa.CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_segment_confidence"),
    )
    op.create_index("ix_transcript_segments_track_id", "transcript_segments", ["track_id"])
    op.create_index("ix_transcript_segments_run_id", "transcript_segments", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_transcript_segments_run_id", table_name="transcript_segments")
    op.drop_index("ix_transcript_segments_track_id", table_name="transcript_segments")
    op.drop_table("transcript_segments")
    op.drop_index("ix_transcription_runs_output_track_id", table_name="transcription_runs")
    op.drop_index("ix_transcription_runs_source_track_id", table_name="transcription_runs")
    op.drop_table("transcription_runs")
