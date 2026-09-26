"""Session ve Track çekirdeği.

Revision ID: 0001_session_track
Revises:
"""
from alembic import op
import sqlalchemy as sa


revision = "0001_session_track"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("scenario", sa.Text()),
        sa.Column("started_at_utc", sa.DateTime(timezone=True)),
        sa.Column("timezone_name", sa.String(64), nullable=False),
        sa.Column("duration_ms", sa.Integer()),
        sa.Column("privacy_mode", sa.String(13), nullable=False),
        sa.Column("external_context_id", sa.String(255)),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("duration_ms IS NULL OR duration_ms >= 0", name="ck_session_duration_nonnegative"),
    )
    op.create_table(
        "tracks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("parent_track_id", sa.String(36)),
        sa.Column("type", sa.String(16), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("room", sa.String(120)),
        sa.Column("source_label", sa.String(120)),
        sa.Column("media_path", sa.Text()),
        sa.Column("offset_ms", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer()),
        sa.Column("checksum_sha256", sa.String(64)),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["parent_track_id"], ["tracks.id"], ondelete="SET NULL"),
        sa.CheckConstraint("duration_ms IS NULL OR duration_ms >= 0", name="ck_track_duration_nonnegative"),
    )
    op.create_index("ix_tracks_session_id", "tracks", ["session_id"])
    op.create_index("ix_tracks_parent_track_id", "tracks", ["parent_track_id"])


def downgrade() -> None:
    op.drop_index("ix_tracks_parent_track_id", table_name="tracks")
    op.drop_index("ix_tracks_session_id", table_name="tracks")
    op.drop_table("tracks")
    op.drop_table("sessions")


