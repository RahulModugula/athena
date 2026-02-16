"""initial schema with pgvector

Revision ID: 001
Revises:
Create Date: 2026-03-18 19:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False, unique=True),
        sa.Column("mime_type", sa.String(), nullable=False),
        sa.Column("metadata", sa.JSON(), server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "document_id", sa.UUID(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("chunking_strategy", sa.String(), nullable=False),
        sa.Column("metadata", sa.JSON(), server_default=sa.text("'{}'")),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.execute("""
        CREATE INDEX idx_chunks_embedding ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    op.execute("CREATE INDEX idx_chunks_tsv ON chunks USING GIN (to_tsvector('english', content))")
    op.create_index("idx_chunks_document_id", "chunks", ["document_id"])

    op.create_table(
        "queries",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("context_chunks", sa.ARRAY(sa.UUID()), nullable=False),
        sa.Column("retrieval_strategy", sa.String(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "eval_runs",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("dataset_name", sa.String(), nullable=False),
        sa.Column("chunking_strategy", sa.String(), nullable=False),
        sa.Column("retrieval_strategy", sa.String(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("eval_runs")
    op.drop_table("queries")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
