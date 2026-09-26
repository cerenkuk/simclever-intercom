"""constrain bookmark source values

Revision ID: 0007_event_source_constraint
Revises: 0006_event_review_audit
"""
from alembic import op


revision = "0007_event_source_constraint"
down_revision = "0006_event_review_audit"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("events") as batch:
        batch.create_check_constraint(
            "ck_events_source_allowed",
            "source IN ('llm', 'eğitmen', 'eğitmen_notu', 'cihaz')",
        )


def downgrade():
    with op.batch_alter_table("events") as batch:
        batch.drop_constraint("ck_events_source_allowed", type_="check")
