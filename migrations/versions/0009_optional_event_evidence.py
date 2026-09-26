"""Allow bookmarks without ASR evidence.

Revision ID: 0009_optional_event_evidence
Revises: 0008_processing_metrics
"""

from alembic import op
import sqlalchemy as sa


revision = "0009_optional_event_evidence"
down_revision = "0008_processing_metrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("events") as batch_op:
        batch_op.alter_column(
            "source_segment_id",
            existing_type=sa.String(length=36),
            nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("events") as batch_op:
        batch_op.alter_column(
            "source_segment_id",
            existing_type=sa.String(length=36),
            nullable=False,
        )
