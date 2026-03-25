"""Standalone MCP server entry point for stdio transport."""

from app.mcp.server import mcp


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
