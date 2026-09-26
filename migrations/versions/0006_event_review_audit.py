"""event review audit

Revision ID: 0006_event_review_audit
Revises: 0005_index_foreign_keys_not_null
"""
from alembic import op
import sqlalchemy as sa


revision = "0006_event_review_audit"
down_revision = "0005_index_foreign_keys_not_null"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("events") as batch:
        batch.add_column(sa.Column("reviewed_by", sa.String(120), nullable=True))
        batch.add_column(sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True))


def downgrade():
    with op.batch_alter_table("events") as batch:
        batch.drop_column("reviewed_at")
        batch.drop_column("reviewed_by")
