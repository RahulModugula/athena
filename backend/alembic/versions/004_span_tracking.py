"""Add span-level tracking for citations: offsets and raw document text

Revision ID: 004
Revises: 003
Create Date: 2026-04-10 14:30:00.000000
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add raw_text to documents table to store immutable source for span recovery
    op.add_column(
        "documents",
        sa.Column("raw_text", sa.Text(), nullable=True),
    )

    # Add span tracking columns to chunks table
    op.add_column(
        "chunks",
        sa.Column("start_offset", sa.Integer(), nullable=True),
    )
    op.add_column(
        "chunks",
        sa.Column("end_offset", sa.Integer(), nullable=True),
    )
    op.add_column(
        "chunks",
        sa.Column("source_text_hash", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chunks", "source_text_hash")
    op.drop_column("chunks", "end_offset")
    op.drop_column("chunks", "start_offset")
    op.drop_column("documents", "raw_text")
