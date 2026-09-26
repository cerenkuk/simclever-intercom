"""add processing metrics and deterministic index fingerprint

Revision ID: 0008_processing_metrics
Revises: 0007_event_source_constraint
"""
import sqlalchemy as sa
from alembic import op

revision = "0008_processing_metrics"
down_revision = "0007_event_source_constraint"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("transcription_runs") as batch:
        batch.add_column(sa.Column("processing_ms", sa.Integer(), nullable=True))
    with op.batch_alter_table("index_runs") as batch:
        batch.add_column(sa.Column("processing_ms", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("model_load_ms", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("chunk_count", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("failed_chunk_count", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("peak_ram_mb", sa.Float(), nullable=True))
        batch.add_column(sa.Column("input_fingerprint", sa.String(length=64), nullable=True))
        batch.create_index("ix_index_runs_input_fingerprint", ["input_fingerprint"], unique=False)


def downgrade():
    with op.batch_alter_table("index_runs") as batch:
        batch.drop_index("ix_index_runs_input_fingerprint")
        for name in ("input_fingerprint", "peak_ram_mb", "failed_chunk_count", "chunk_count", "model_load_ms", "processing_ms"):
            batch.drop_column(name)
    with op.batch_alter_table("transcription_runs") as batch:
        batch.drop_column("processing_ms")
