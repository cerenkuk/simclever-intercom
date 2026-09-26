"""Add index provenance and normalized external event support.

Revision ID: 0013_index_provenance
Revises: 0012_resumable_jobs
"""

import sqlalchemy as sa
from alembic import op


revision = "0013_index_provenance"
down_revision = "0012_resumable_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("index_runs") as batch:
        batch.add_column(sa.Column("source_fingerprint", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("config_fingerprint", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("model_fingerprint", sa.String(length=64), nullable=True))
    with op.batch_alter_table("events") as batch:
        batch.drop_constraint("ck_events_source_allowed", type_="check")
        batch.alter_column(
            "evidence_segment_ref",
            existing_type=sa.String(length=16),
            type_=sa.String(length=255),
            nullable=True,
        )
        batch.create_check_constraint(
            "ck_events_source_allowed",
            "source IN ('llm', 'eğitmen', 'eğitmen_notu', 'cihaz', 'sistem')",
        )


def downgrade() -> None:
    with op.batch_alter_table("events") as batch:
        batch.drop_constraint("ck_events_source_allowed", type_="check")
        batch.alter_column(
            "evidence_segment_ref",
            existing_type=sa.String(length=255),
            type_=sa.String(length=16),
            nullable=True,
        )
        batch.create_check_constraint(
            "ck_events_source_allowed",
            "source IN ('llm', 'eğitmen', 'eğitmen_notu', 'cihaz')",
        )
    with op.batch_alter_table("index_runs") as batch:
        batch.drop_column("model_fingerprint")
        batch.drop_column("config_fingerprint")
        batch.drop_column("source_fingerprint")
