"""
nephilim.api.server
---------------------
FastAPI application that mounts the Strawberry GraphQL endpoint and
exposes REST health/metrics routes.

Routes
------
GET  /health          — liveness probe
GET  /metrics         — Prometheus-style plaintext metrics
POST /graphql         — GraphQL endpoint
GET  /graphql         — GraphiQL interactive IDE (development)
"""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from strawberry.fastapi import GraphQLRouter

from nephilim.api.graphql_schema import schema
from nephilim.config import get_settings

# Module-level counters for the /metrics endpoint
_request_count = 0
_start_time = time.time()


def create_app(
    neo4j_client: Any = None,
    timescale_client: Any = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    neo4j_client, timescale_client:
        Injected storage clients. When provided they are attached to the
        GraphQL context so resolvers can call them. In testing, mock objects
        can be passed here.
    """
    settings = get_settings()

    app = FastAPI(
        title="NEPHILIM API",
        description=(
            "Real-time on-chain entity intelligence and MEV attribution engine."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow all origins for an open portfolio project
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging middleware ────────────────────────────────────────────

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Response:
        global _request_count
        _request_count += 1
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(
            "{} {} {} {:.1f}ms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    # ── REST endpoints ────────────────────────────────────────────────────────

    @app.get("/health", tags=["Infrastructure"])
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.1.0",
            "uptime_seconds": round(time.time() - _start_time, 1),
        }

    @app.get("/metrics", tags=["Infrastructure"])
    async def metrics() -> Response:
        body = (
            f"# HELP nephilim_requests_total Total HTTP requests\n"
            f"# TYPE nephilim_requests_total counter\n"
            f"nephilim_requests_total {_request_count}\n"
            f"# HELP nephilim_uptime_seconds Engine uptime in seconds\n"
            f"# TYPE nephilim_uptime_seconds gauge\n"
            f"nephilim_uptime_seconds {round(time.time() - _start_time, 1)}\n"
        )
        return Response(content=body, media_type="text/plain")

    # ── GraphQL ───────────────────────────────────────────────────────────────

    async def get_context() -> Dict[str, Any]:
        return {
            "neo4j_client": neo4j_client,
            "timescale_client": timescale_client,
        }

    graphql_router = GraphQLRouter(
        schema,
        context_getter=get_context,
        graphiql=True,  # Enable interactive IDE
    )
    app.include_router(graphql_router, prefix="/graphql")

    logger.info(
        "NEPHILIM API ready — GraphQL at /graphql, GraphiQL at /graphql"
    )
    return app


# Standalone runner (used by docker-compose)
if __name__ == "__main__":
    import uvicorn
    from nephilim.storage.neo4j_client import Neo4jClient
    from nephilim.storage.timescale_client import TimescaleClient

    settings = get_settings()
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    timescale = TimescaleClient(dsn=settings.timescale_dsn)

    app = create_app(neo4j_client=neo4j, timescale_client=timescale)

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )
