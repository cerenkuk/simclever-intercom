"""Add resumable job state and cancellation.

Revision ID: 0012_resumable_jobs
Revises: 0011_processing_jobs
"""

from alembic import op
import sqlalchemy as sa


revision = "0012_resumable_jobs"
down_revision = "0011_processing_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column("status", existing_type=sa.String(length=11), type_=sa.String(length=16), nullable=False)
        batch.add_column(sa.Column("stage", sa.String(length=120), nullable=True))
        batch.add_column(sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="2"))
        batch.add_column(sa.Column("checkpoint_json", sa.Text(), nullable=True))
        batch.add_column(sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_column("heartbeat_at")
        batch.drop_column("max_attempts")
        batch.drop_column("checkpoint_json")
        batch.drop_column("attempt_count")
        batch.drop_column("stage")
        batch.alter_column("status", existing_type=sa.String(length=16), type_=sa.String(length=11), nullable=False)
