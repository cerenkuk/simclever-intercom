"""Add durable processing job records.

Revision ID: 0011_processing_jobs
Revises: 0010_frozen_event_evidence
"""

from alembic import op
import sqlalchemy as sa


revision = "0011_processing_jobs"
down_revision = "0010_frozen_event_evidence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("session_id", sa.String(length=36), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("track_id", sa.String(length=36), sa.ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=11), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_processing_jobs_session_id", "processing_jobs", ["session_id"])
    op.create_index("ix_processing_jobs_track_id", "processing_jobs", ["track_id"])


def downgrade() -> None:
    op.drop_index("ix_processing_jobs_track_id", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_session_id", table_name="processing_jobs")
    op.drop_table("processing_jobs")
