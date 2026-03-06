"""add status column to eval_runs

Revision ID: 002
Revises: 001
Create Date: 2026-03-23 10:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "eval_runs",
        sa.Column("status", sa.String(), nullable=False, server_default="completed"),
    )


def downgrade() -> None:
    op.drop_column("eval_runs", "status")
