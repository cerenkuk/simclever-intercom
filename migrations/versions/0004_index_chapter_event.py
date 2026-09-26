"""LLM index run, chapter ve event tablolari."""
from alembic import op
import sqlalchemy as sa

revision = "0004_index_chapter_event"
down_revision = "0003_transcription_segments"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table("index_runs", sa.Column("id",sa.String(36),primary_key=True), sa.Column("source_track_id",sa.String(36),sa.ForeignKey("tracks.id",ondelete="CASCADE"),nullable=False), sa.Column("output_track_id",sa.String(36),sa.ForeignKey("tracks.id",ondelete="CASCADE"),nullable=False), sa.Column("llm_provider",sa.String(80),nullable=False), sa.Column("llm_model",sa.String(160),nullable=False), sa.Column("llm_version",sa.String(80),nullable=False), sa.Column("prompt_version",sa.String(40),nullable=False), sa.Column("category_version",sa.String(40),nullable=False), sa.Column("schema_version",sa.String(40),nullable=False), sa.Column("lead_in_ms",sa.Integer(),nullable=False), sa.Column("status",sa.String(9),nullable=False), sa.Column("created_at",sa.DateTime(timezone=True),nullable=False))
    op.create_index("ix_index_runs_source_track_id","index_runs",["source_track_id"]); op.create_index("ix_index_runs_output_track_id","index_runs",["output_track_id"])
    op.create_table("chapters", sa.Column("id",sa.String(36),primary_key=True), sa.Column("index_run_id",sa.String(36),sa.ForeignKey("index_runs.id",ondelete="CASCADE")), sa.Column("track_id",sa.String(36),sa.ForeignKey("tracks.id",ondelete="CASCADE")), sa.Column("start_segment_id",sa.String(36),sa.ForeignKey("transcript_segments.id"),nullable=False), sa.Column("end_segment_id",sa.String(36),sa.ForeignKey("transcript_segments.id"),nullable=False), sa.Column("start_ms",sa.Integer(),nullable=False), sa.Column("end_ms",sa.Integer(),nullable=False), sa.Column("title",sa.String(200),nullable=False), sa.Column("summary",sa.Text(),nullable=False), sa.Column("confidence",sa.Float(),nullable=False), sa.Column("sequence_no",sa.Integer(),nullable=False))
    op.create_index("ix_chapters_index_run_id","chapters",["index_run_id"]); op.create_index("ix_chapters_track_id","chapters",["track_id"])
    op.create_table("events", sa.Column("id",sa.String(36),primary_key=True), sa.Column("index_run_id",sa.String(36),sa.ForeignKey("index_runs.id",ondelete="CASCADE")), sa.Column("track_id",sa.String(36),sa.ForeignKey("tracks.id",ondelete="CASCADE")), sa.Column("source_segment_id",sa.String(36),sa.ForeignKey("transcript_segments.id"),nullable=False), sa.Column("event_time_ms",sa.Integer(),nullable=False), sa.Column("playback_time_ms",sa.Integer(),nullable=False), sa.Column("title",sa.String(200),nullable=False), sa.Column("description",sa.Text(),nullable=False), sa.Column("category",sa.String(80),nullable=False), sa.Column("source",sa.String(40),nullable=False), sa.Column("confidence",sa.Float(),nullable=False), sa.Column("importance",sa.Float(),nullable=False), sa.Column("status",sa.String(10),nullable=False))
    op.create_index("ix_events_index_run_id","events",["index_run_id"]); op.create_index("ix_events_track_id","events",["track_id"])

def downgrade() -> None:
    op.drop_table("events"); op.drop_table("chapters"); op.drop_table("index_runs")
