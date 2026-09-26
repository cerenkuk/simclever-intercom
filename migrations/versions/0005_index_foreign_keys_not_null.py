"""Chapter ve event sahiplik alanlarini zorunlu yap."""
from alembic import op
import sqlalchemy as sa

revision = "0005_index_foreign_keys_not_null"
down_revision = "0004_index_chapter_event"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("chapters") as batch:
        batch.alter_column("index_run_id", existing_type=sa.String(36), nullable=False)
        batch.alter_column("track_id", existing_type=sa.String(36), nullable=False)
    with op.batch_alter_table("events") as batch:
        batch.alter_column("index_run_id", existing_type=sa.String(36), nullable=False)
        batch.alter_column("track_id", existing_type=sa.String(36), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("events") as batch:
        batch.alter_column("track_id", existing_type=sa.String(36), nullable=True)
        batch.alter_column("index_run_id", existing_type=sa.String(36), nullable=True)
    with op.batch_alter_table("chapters") as batch:
        batch.alter_column("track_id", existing_type=sa.String(36), nullable=True)
        batch.alter_column("index_run_id", existing_type=sa.String(36), nullable=True)
