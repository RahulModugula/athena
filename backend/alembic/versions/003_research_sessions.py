"""add research_sessions table and graph_extracted column on chunks

Revision ID: 003
Revises: 002
Create Date: 2026-03-24 19:00:00.000000
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_sessions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("plan", sa.Text(), nullable=True),
        sa.Column("final_answer", sa.Text(), nullable=True),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column("graph_context", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.add_column(
        "chunks",
        sa.Column("graph_extracted", sa.Boolean(), server_default="false", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("chunks", "graph_extracted")
    op.drop_table("research_sessions")
