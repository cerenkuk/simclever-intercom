"""Freeze ASR evidence on bookmarks.

Revision ID: 0010_frozen_event_evidence
Revises: 0009_optional_event_evidence
"""

from alembic import op
import sqlalchemy as sa


revision = "0010_frozen_event_evidence"
down_revision = "0009_optional_event_evidence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("events") as batch:
        batch.add_column(sa.Column("evidence_segment_ref", sa.String(length=16), nullable=True))
        batch.add_column(sa.Column("evidence_text", sa.Text(), nullable=True))
        batch.add_column(sa.Column("evidence_start_ms", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("evidence_end_ms", sa.Integer(), nullable=True))
    op.execute(sa.text("""
        UPDATE events
        SET evidence_segment_ref = (SELECT segment_ref FROM transcript_segments WHERE id = events.source_segment_id),
            evidence_text = (SELECT text FROM transcript_segments WHERE id = events.source_segment_id),
            evidence_start_ms = (SELECT start_ms FROM transcript_segments WHERE id = events.source_segment_id),
            evidence_end_ms = (SELECT end_ms FROM transcript_segments WHERE id = events.source_segment_id)
        WHERE source_segment_id IS NOT NULL
    """))


def downgrade() -> None:
    with op.batch_alter_table("events") as batch:
        batch.drop_column("evidence_end_ms")
        batch.drop_column("evidence_start_ms")
        batch.drop_column("evidence_text")
        batch.drop_column("evidence_segment_ref")
