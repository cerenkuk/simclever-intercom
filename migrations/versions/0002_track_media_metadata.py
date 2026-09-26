"""Track medya metadata alanları.

Revision ID: 0002_track_media_metadata
Revises: 0001_session_track
"""
from alembic import op
import sqlalchemy as sa


revision = "0002_track_media_metadata"
down_revision = "0001_session_track"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tracks", sa.Column("original_filename", sa.String(255)))
    op.add_column("tracks", sa.Column("mime_type", sa.String(120)))
    op.add_column("tracks", sa.Column("container_format", sa.String(120)))
    op.add_column("tracks", sa.Column("codec_name", sa.String(120)))
    op.add_column("tracks", sa.Column("sample_rate", sa.Integer()))
    op.add_column("tracks", sa.Column("channel_count", sa.Integer()))


def downgrade() -> None:
    op.drop_column("tracks", "channel_count")
    op.drop_column("tracks", "sample_rate")
    op.drop_column("tracks", "codec_name")
    op.drop_column("tracks", "container_format")
    op.drop_column("tracks", "mime_type")
    op.drop_column("tracks", "original_filename")


