"""Web UI for visualizing and exploring procedural memory.

Provides a FastAPI-based interface for:
- Timeline visualization of experiences
- Search and filtering
- Concept tag cloud
- Experience type distribution
- Success/failure analysis

Usage:
    python -m rec_praxis_rlm.web_viewer --memory-path ./memory.jsonl --port 8080
"""

import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from typing import Optional

try:
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from rec_praxis_rlm.memory import ProceduralMemory
from rec_praxis_rlm.config import MemoryConfig

logger = logging.getLogger(__name__)


def create_app(memory_path: str) -> "FastAPI":
    """Create FastAPI app for memory viewer.

    Args:
        memory_path: Path to memory JSONL file

    Returns:
        FastAPI application instance
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for web viewer. Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(title="REC Praxis RLM Memory Viewer", version="0.9.1")

    # Load memory
    config = MemoryConfig(storage_path=memory_path, embedding_model="")
    memory = ProceduralMemory(config=config, use_faiss=False)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Render main dashboard."""
        total = memory.size()

        if total == 0:
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>REC Praxis RLM Memory Viewer</title>
                <style>
                    body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }
                    h1 { color: #333; }
                </style>
            </head>
            <body>
                <h1>REC Praxis RLM Memory Viewer</h1>
                <p>No experiences found in memory. Start using rec-praxis-rlm to capture experiences!</p>
            </body>
            </html>
            """

        # Calculate statistics
        success_count = sum(1 for exp in memory.experiences if exp.success)
        success_rate = (success_count / total * 100) if total > 0 else 0

        # Experience type distribution
        type_counts = Counter(
            exp.experience_type if hasattr(exp, 'experience_type') else 'unknown'
            for exp in memory.experiences
        )

        # Tag cloud data
        all_tags = []
        for exp in memory.experiences:
            if hasattr(exp, 'tags') and exp.tags:
                all_tags.extend(exp.tags)
        tag_counts = Counter(all_tags).most_common(20)

        # Recent experiences (last 10)
        recent = sorted(memory.experiences, key=lambda x: x.timestamp, reverse=True)[:10]

        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>REC Praxis RLM Memory Viewer</title>
            <meta charset="utf-8">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                }}
                h1 {{
                    font-size: 2em;
                    margin-bottom: 10px;
                }}
                .subtitle {{
                    opacity: 0.9;
                    font-size: 1.1em;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    padding: 30px;
                    background: #f8f9fa;
                }}
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                }}
                .stat-value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    color: #667eea;
                    margin-bottom: 5px;
                }}
                .stat-label {{
                    color: #666;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                .section {{
                    padding: 30px;
                    border-bottom: 1px solid #eee;
                }}
                .section:last-child {{ border-bottom: none; }}
                h2 {{
                    margin-bottom: 20px;
                    color: #333;
                    font-size: 1.5em;
                }}
                .type-chart {{
                    display: flex;
                    gap: 15px;
                    flex-wrap: wrap;
                }}
                .type-badge {{
                    display: inline-flex;
                    align-items: center;
                    padding: 10px 15px;
                    border-radius: 6px;
                    font-weight: 500;
                    gap: 8px;
                }}
                .type-badge.learn {{ background: #e3f2fd; color: #1976d2; }}
                .type-badge.recover {{ background: #fff3e0; color: #f57c00; }}
                .type-badge.optimize {{ background: #f3e5f5; color: #7b1fa2; }}
                .type-badge.explore {{ background: #e8f5e9; color: #388e3c; }}
                .type-badge .count {{
                    background: rgba(0,0,0,0.1);
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-size: 0.9em;
                }}
                .tag-cloud {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }}
                .tag {{
                    display: inline-block;
                    padding: 6px 12px;
                    background: #f5f5f5;
                    border-radius: 4px;
                    font-size: 0.9em;
                    color: #555;
                    transition: all 0.2s;
                }}
                .tag:hover {{
                    background: #667eea;
                    color: white;
                    transform: translateY(-2px);
                    cursor: pointer;
                }}
                .experience-list {{
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                }}
                .experience {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }}
                .experience.success {{ border-left-color: #4caf50; }}
                .experience.failure {{ border-left-color: #f44336; }}
                .experience-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }}
                .experience-goal {{
                    font-weight: 600;
                    color: #333;
                    font-size: 1.05em;
                }}
                .experience-meta {{
                    display: flex;
                    gap: 10px;
                    font-size: 0.85em;
                    color: #666;
                }}
                .experience-type {{
                    padding: 3px 8px;
                    border-radius: 4px;
                    font-weight: 500;
                    font-size: 0.8em;
                }}
                .experience-type.learn {{ background: #e3f2fd; color: #1976d2; }}
                .experience-type.recover {{ background: #fff3e0; color: #f57c00; }}
                .experience-type.optimize {{ background: #f3e5f5; color: #7b1fa2; }}
                .experience-type.explore {{ background: #e8f5e9; color: #388e3c; }}
                .experience-result {{
                    margin-top: 10px;
                    color: #555;
                    line-height: 1.5;
                }}
                .timestamp {{
                    color: #999;
                    font-size: 0.85em;
                }}
                .success-indicator {{
                    display: inline-block;
                    padding: 4px 10px;
                    border-radius: 4px;
                    font-size: 0.85em;
                    font-weight: 500;
                }}
                .success-indicator.success {{
                    background: #e8f5e9;
                    color: #2e7d32;
                }}
                .success-indicator.failure {{
                    background: #ffebee;
                    color: #c62828;
                }}
                .api-links {{
                    background: #f5f5f5;
                    padding: 15px;
                    border-radius: 6px;
                    margin-top: 20px;
                }}
                .api-links h3 {{
                    font-size: 1em;
                    margin-bottom: 10px;
                    color: #555;
                }}
                .api-links a {{
                    color: #667eea;
                    text-decoration: none;
                    margin-right: 15px;
                }}
                .api-links a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧠 REC Praxis RLM Memory Viewer</h1>
                    <div class="subtitle">Procedural memory visualization and exploration</div>
                </div>

                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value">{total}</div>
                        <div class="stat-label">Total Experiences</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{success_rate:.1f}%</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(tag_counts)}</div>
                        <div class="stat-label">Unique Tags</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(type_counts)}</div>
                        <div class="stat-label">Experience Types</div>
                    </div>
                </div>

                <div class="section">
                    <h2>📊 Experience Type Distribution</h2>
                    <div class="type-chart">
        """

        for exp_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            html += f'''
                        <div class="type-badge {exp_type}">
                            <span>{exp_type}</span>
                            <span class="count">{count}</span>
                        </div>
            '''

        html += """
                    </div>
                </div>

                <div class="section">
                    <h2>🏷️ Tag Cloud (Top 20)</h2>
                    <div class="tag-cloud">
        """

        for tag, count in tag_counts:
            size = min(1.5, 0.9 + (count / max(c for _, c in tag_counts)) * 0.6)
            html += f'<span class="tag" style="font-size: {size}em">{tag} ({count})</span>'

        html += """
                    </div>
                </div>

                <div class="section">
                    <h2>📜 Recent Experiences</h2>
                    <div class="experience-list">
        """

        for exp in recent:
            exp_type = exp.experience_type if hasattr(exp, 'experience_type') else 'unknown'
            status_class = 'success' if exp.success else 'failure'
            status_text = '✓ Success' if exp.success else '✗ Failure'
            timestamp = datetime.fromtimestamp(exp.timestamp).strftime('%Y-%m-%d %H:%M:%S')

            html += f'''
                        <div class="experience {status_class}">
                            <div class="experience-header">
                                <div class="experience-goal">{exp.goal[:100]}</div>
                                <div class="experience-meta">
                                    <span class="experience-type {exp_type}">{exp_type}</span>
                                    <span class="success-indicator {status_class}">{status_text}</span>
                                </div>
                            </div>
                            <div class="experience-result">{exp.result[:200]}</div>
                            <div class="timestamp">🕐 {timestamp}</div>
                        </div>
            '''

        html += f"""
                    </div>
                </div>

                <div class="section">
                    <div class="api-links">
                        <h3>📡 API Endpoints:</h3>
                        <a href="/api/experiences">All Experiences (JSON)</a>
                        <a href="/api/stats">Statistics (JSON)</a>
                        <a href="/api/tags">Tags (JSON)</a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    @app.get("/api/experiences")
    async def get_experiences(
        type: Optional[str] = Query(None, description="Filter by experience type"),
        success: Optional[bool] = Query(None, description="Filter by success status"),
        limit: int = Query(100, description="Maximum number of results"),
    ):
        """Get experiences as JSON with optional filters."""
        experiences = memory.experiences

        # Apply filters
        if type:
            experiences = [e for e in experiences if hasattr(e, 'experience_type') and e.experience_type == type]
        if success is not None:
            experiences = [e for e in experiences if e.success == success]

        # Sort by timestamp descending and limit
        experiences = sorted(experiences, key=lambda x: x.timestamp, reverse=True)[:limit]

        # Convert to dict
        return {
            "count": len(experiences),
            "experiences": [exp.model_dump() for exp in experiences]
        }

    @app.get("/api/stats")
    async def get_stats():
        """Get memory statistics as JSON."""
        total = memory.size()
        success_count = sum(1 for exp in memory.experiences if exp.success)

        type_counts = Counter(
            exp.experience_type if hasattr(exp, 'experience_type') else 'unknown'
            for exp in memory.experiences
        )

        return {
            "total_experiences": total,
            "success_count": success_count,
            "failure_count": total - success_count,
            "success_rate": (success_count / total * 100) if total > 0 else 0,
            "type_distribution": dict(type_counts),
        }

    @app.get("/api/tags")
    async def get_tags(limit: int = Query(50, description="Maximum number of tags")):
        """Get tag frequency as JSON."""
        all_tags = []
        for exp in memory.experiences:
            if hasattr(exp, 'tags') and exp.tags:
                all_tags.extend(exp.tags)

        tag_counts = Counter(all_tags).most_common(limit)

        return {
            "count": len(tag_counts),
            "tags": [{"tag": tag, "count": count} for tag, count in tag_counts]
        }

    return app


def main():
    """Run web viewer CLI."""
    parser = argparse.ArgumentParser(
        description="REC Praxis RLM Memory Viewer - Web UI for exploring experiences"
    )
    parser.add_argument(
        "--memory-path",
        default="./memory.jsonl",
        help="Path to memory JSONL file (default: ./memory.jsonl)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run server on (default: 8080)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )

    args = parser.parse_args()

    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI is required for web viewer.")
        print("Install with: pip install fastapi uvicorn")
        return 1

    print(f"🚀 Starting REC Praxis RLM Memory Viewer...")
    print(f"📂 Memory file: {args.memory_path}")
    print(f"🌐 URL: http://{args.host}:{args.port}")
    print(f"📡 API docs: http://{args.host}:{args.port}/docs")
    print()

    app = create_app(args.memory_path)
    uvicorn.run(app, host=args.host, port=args.port)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
