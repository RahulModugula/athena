"""Tenant management and API key provisioning routes."""

import hashlib
import secrets
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.orm import APIKey, Document, Query, Tenant

logger = structlog.get_logger()
router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TenantCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9\-]*$")
    plan: str = Field(default="free")


class TenantResponse(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    plan: str
    settings: dict[str, object]
    created_at: str


class APIKeyCreate(BaseModel):
    name: str = Field(default="default", max_length=100)
    key_type: str = Field(default="secret", pattern=r"^(secret|publishable)$")


class APIKeyResponse(BaseModel):
    id: uuid.UUID
    prefix: str
    name: str
    key_type: str
    created_at: str


class APIKeyCreated(APIKeyResponse):
    """Returned only on creation — the raw key is shown once."""
    raw_key: str


class UsageResponse(BaseModel):
    queries_this_month: int
    documents_ingested: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_api_key(key_type: str) -> tuple[str, str, str]:
    """Generate raw key, hash, and prefix."""
    prefix = "sk_" if key_type == "secret" else "pk_"
    random_part = secrets.token_urlsafe(32)
    raw_key = f"{prefix}{random_part}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    return raw_key, key_hash, raw_key[:12]


def _get_tenant_id(request: Request) -> uuid.UUID:
    tid = getattr(request.state, "tenant_id", None)
    if tid is None:
        raise HTTPException(401, "authentication required")
    return uuid.UUID(str(tid))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/tenants", response_model=APIKeyCreated, status_code=201)
async def create_tenant(
    body: TenantCreate,
    db: AsyncSession = Depends(get_db),
) -> APIKeyCreated:
    """Create a new tenant and return its first secret API key."""
    existing = await db.execute(select(Tenant).where(Tenant.slug == body.slug))
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"slug already taken: {body.slug}")

    tenant = Tenant(name=body.name, slug=body.slug, plan=body.plan)
    db.add(tenant)
    await db.flush()

    raw_key, key_hash, prefix = _generate_api_key("secret")
    api_key = APIKey(
        tenant_id=tenant.id,
        key_hash=key_hash,
        prefix=prefix,
        name="default",
        key_type="secret",
    )
    db.add(api_key)
    await db.flush()

    return APIKeyCreated(
        id=api_key.id,
        prefix=prefix,
        name=api_key.name,
        key_type=api_key.key_type,
        created_at=str(api_key.created_at),
        raw_key=raw_key,
    )


@router.get("/tenants/me", response_model=TenantResponse)
async def get_current_tenant(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TenantResponse:
    tid = _get_tenant_id(request)
    result = await db.execute(select(Tenant).where(Tenant.id == tid))
    tenant = result.scalar_one_or_none()
    if tenant is None:
        raise HTTPException(404, "tenant not found")
    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        plan=tenant.plan,
        settings=tenant.settings,
        created_at=str(tenant.created_at),
    )


@router.post("/tenants/me/api-keys", response_model=APIKeyCreated, status_code=201)
async def create_api_key(
    body: APIKeyCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> APIKeyCreated:
    tid = _get_tenant_id(request)
    raw_key, key_hash, prefix = _generate_api_key(body.key_type)
    api_key = APIKey(
        tenant_id=tid,
        key_hash=key_hash,
        prefix=prefix,
        name=body.name,
        key_type=body.key_type,
    )
    db.add(api_key)
    await db.flush()

    return APIKeyCreated(
        id=api_key.id,
        prefix=prefix,
        name=api_key.name,
        key_type=api_key.key_type,
        created_at=str(api_key.created_at),
        raw_key=raw_key,
    )


@router.get("/tenants/me/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> list[APIKeyResponse]:
    tid = _get_tenant_id(request)
    result = await db.execute(
        select(APIKey)
        .where(APIKey.tenant_id == tid, APIKey.is_active.is_(True))
        .order_by(APIKey.created_at.desc())
    )
    keys = result.scalars().all()
    return [
        APIKeyResponse(
            id=k.id, prefix=k.prefix, name=k.name,
            key_type=k.key_type, created_at=str(k.created_at),
        )
        for k in keys
    ]


@router.delete("/tenants/me/api-keys/{key_id}")
async def revoke_api_key(
    key_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict[str, bool]:
    tid = _get_tenant_id(request)
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id, APIKey.tenant_id == tid)
    )
    key = result.scalar_one_or_none()
    if key is None:
        raise HTTPException(404, "api key not found")
    key.is_active = False
    return {"revoked": True}


@router.get("/tenants/me/usage", response_model=UsageResponse)
async def tenant_usage(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> UsageResponse:
    tid = _get_tenant_id(request)

    doc_count = (
        await db.execute(
            select(func.count()).select_from(Document).where(Document.tenant_id == tid)
        )
    ).scalar_one()

    query_count = (
        await db.execute(
            select(func.count()).select_from(Query).where(Query.tenant_id == tid)
        )
    ).scalar_one()

    return UsageResponse(queries_this_month=query_count, documents_ingested=doc_count)
