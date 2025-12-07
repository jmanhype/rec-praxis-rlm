# REC Praxis RLM Web Viewer

Beautiful web interface for visualizing and exploring procedural memory.

![Web Viewer Dashboard](https://img.shields.io/badge/UI-FastAPI-009688?style=for-the-badge&logo=fastapi)

## Features

### 📊 Dashboard
- **Statistics Overview**: Total experiences, success rate, unique tags, experience types
- **Experience Type Distribution**: Visual breakdown of learn/recover/optimize/explore patterns
- **Tag Cloud**: Top 20 most frequent semantic tags with visual emphasis
- **Recent Experiences Timeline**: Last 10 experiences with full details

### 📡 REST API
- **GET /api/experiences**: Query experiences with filters (type, success, limit)
- **GET /api/stats**: Memory statistics (totals, success rates, type distribution)
- **GET /api/tags**: Tag frequency analysis

### 🎨 Beautiful UI
- Gradient header with modern design
- Responsive grid layout
- Color-coded experience types
- Success/failure indicators
- Smooth transitions and hover effects

## Installation

Install with web dependencies:

```bash
# Install rec-praxis-rlm with web viewer support
pip install -e ".[web]"

# Or install dependencies separately
pip install fastapi uvicorn
```

## Quick Start

### Launch Web Viewer

```bash
# View memory from default location
rec-praxis-web --memory-path ./memory.jsonl

# Custom port and host
rec-praxis-web --memory-path .claude/memory.jsonl --port 3000 --host 0.0.0.0

# Use Python module directly
python -m rec_praxis_rlm.web_viewer --memory-path ./memory.jsonl
```

Then open your browser to:
- **Dashboard**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs (interactive Swagger UI)

### Programmatic Usage

```python
from rec_praxis_rlm.web_viewer import create_app
import uvicorn

# Create app
app = create_app(memory_path="./memory.jsonl")

# Run server
uvicorn.run(app, host="127.0.0.1", port=8080)
```

## API Examples

### Get All Experiences

```bash
curl http://localhost:8080/api/experiences
```

**Response:**
```json
{
  "count": 127,
  "experiences": [
    {
      "env_features": ["python", "testing"],
      "goal": "Fix failing pytest tests",
      "action": "Updated test assertions",
      "result": "All tests passing",
      "success": true,
      "timestamp": 1765150829.1,
      "experience_type": "recover",
      "tags": ["test", "python", "pytest"]
    }
  ]
}
```

### Filter by Experience Type

```bash
curl "http://localhost:8080/api/experiences?type=optimize&limit=10"
```

### Get Statistics

```bash
curl http://localhost:8080/api/stats
```

**Response:**
```json
{
  "total_experiences": 127,
  "success_count": 98,
  "failure_count": 29,
  "success_rate": 77.17,
  "type_distribution": {
    "learn": 45,
    "recover": 38,
    "optimize": 32,
    "explore": 12
  }
}
```

### Get Tag Cloud Data

```bash
curl "http://localhost:8080/api/tags?limit=20"
```

**Response:**
```json
{
  "count": 20,
  "tags": [
    {"tag": "python", "count": 45},
    {"tag": "test", "count": 38},
    {"tag": "database", "count": 32}
  ]
}
```

## Dashboard Sections

### Statistics Cards

Four key metrics at a glance:
- **Total Experiences**: Complete count of captured experiences
- **Success Rate**: Percentage of successful experiences
- **Unique Tags**: Number of distinct semantic tags
- **Experience Types**: Number of experience type categories

### Experience Type Distribution

Visual badges showing the breakdown of:
- 🔵 **learn**: New pattern/knowledge acquisition
- 🟠 **recover**: Error recovery/debugging
- 🟣 **optimize**: Performance/efficiency improvements
- 🟢 **explore**: Experimentation/discovery

### Tag Cloud

Top 20 most frequent tags with:
- Font size indicating frequency
- Hover effects
- Click-to-filter (future feature)

### Recent Experiences

Timeline of last 10 experiences showing:
- Experience type badge
- Success/failure indicator
- Goal description
- Result preview (200 chars)
- Timestamp

## Use Cases

### 1. Debugging Past Failures

View recent failures to understand what didn't work:

```bash
curl "http://localhost:8080/api/experiences?success=false&limit=5"
```

### 2. Learning from Successes

Find successful optimization patterns:

```bash
curl "http://localhost:8080/api/experiences?type=optimize&success=true"
```

### 3. Exploring Tag Patterns

Analyze which tags appear most frequently:

```bash
curl http://localhost:8080/api/tags
```

### 4. Team Knowledge Sharing

Share the dashboard URL with your team to showcase:
- What's been learned
- Common failure patterns
- Optimization strategies
- Exploration efforts

### 5. Progress Tracking

Monitor success rate over time to measure:
- Agent improvement
- Task difficulty trends
- Learning effectiveness

## CLI Options

```bash
rec-praxis-web --help
```

**Options:**
- `--memory-path PATH`: Path to memory JSONL file (default: ./memory.jsonl)
- `--port PORT`: Port to run server on (default: 8080)
- `--host HOST`: Host to bind to (default: 127.0.0.1)

## Architecture

### Tech Stack

- **FastAPI**: Modern, fast web framework
- **Uvicorn**: ASGI server
- **HTML/CSS**: No JavaScript dependencies for simplicity
- **ProceduralMemory**: Core memory backend

### Request Flow

```
Browser Request
    ↓
FastAPI Router
    ↓
ProceduralMemory.experiences
    ↓
Filter/Transform
    ↓
HTML Response / JSON API
```

### Data Flow

1. **Load Memory**: FastAPI app loads memory.jsonl on startup
2. **Query**: Routes query ProceduralMemory.experiences list
3. **Filter**: Apply type/success/limit filters
4. **Transform**: Convert Experience objects to dicts
5. **Render**: Return HTML dashboard or JSON API response

## Security Considerations

### Local-Only by Default

The web viewer binds to `127.0.0.1` by default, meaning:
- ✅ Only accessible from your machine
- ✅ Not exposed to network
- ✅ Safe for sensitive data

### Exposing to Network

To allow network access:

```bash
rec-praxis-web --host 0.0.0.0 --port 8080
```

⚠️ **Warning**: This exposes the viewer to your local network. Consider:
- Firewall rules
- VPN/SSH tunneling for remote access
- Authentication (not built-in, use reverse proxy)

### Privacy

The web viewer:
- **Does not** send data externally
- **Does not** modify memory.jsonl
- **Does** respect privacy redaction in experiences
- **Does** keep all data local

## Troubleshooting

### FastAPI Not Installed

```
Error: FastAPI is required for web viewer.
Install with: pip install fastapi uvicorn
```

**Solution:**
```bash
pip install -e ".[web]"
```

### Memory File Not Found

```
FileNotFoundError: ./memory.jsonl
```

**Solution:**
```bash
# Specify correct path
rec-praxis-web --memory-path /path/to/memory.jsonl

# Or check .claude/memory.jsonl if using hooks
rec-praxis-web --memory-path .claude/memory.jsonl
```

### Port Already in Use

```
ERROR:    [Errno 48] Address already in use
```

**Solution:**
```bash
# Use different port
rec-praxis-web --port 8081

# Or kill process on port 8080
lsof -ti:8080 | xargs kill
```

### Empty Dashboard

If you see "No experiences found":
1. Check memory file path is correct
2. Verify memory.jsonl has experiences (not just version marker)
3. Try capturing some experiences first

## Advanced Usage

### Custom FastAPI Middleware

```python
from rec_praxis_rlm.web_viewer import create_app
from fastapi.middleware.cors import CORSMiddleware

app = create_app("./memory.jsonl")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Embedding in Existing App

```python
from fastapi import FastAPI
from rec_praxis_rlm.web_viewer import create_app

main_app = FastAPI()

# Mount memory viewer under /memory
memory_viewer = create_app("./memory.jsonl")
main_app.mount("/memory", memory_viewer)
```

### Async API Client

```python
import httpx
import asyncio

async def get_stats():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8080/api/stats")
        return response.json()

stats = asyncio.run(get_stats())
print(stats)
```

## Future Enhancements

Potential features for future versions:

- [ ] Search bar with full-text search
- [ ] Date range filtering
- [ ] Export to CSV/JSON
- [ ] Charts and graphs (success rate over time)
- [ ] Real-time updates via WebSocket
- [ ] Tag filtering (click tag to filter)
- [ ] Experience detail modal
- [ ] Pagination for large datasets
- [ ] Authentication/authorization
- [ ] Dark mode toggle

## Contributing

To improve the web viewer:

1. Edit `rec_praxis_rlm/web_viewer.py`
2. Test changes:
   ```bash
   python -m rec_praxis_rlm.web_viewer --memory-path .claude/memory.jsonl
   ```
3. Submit PR with:
   - Description of feature
   - Screenshots (for UI changes)
   - API examples (for endpoint changes)

## License

MIT License - Same as rec-praxis-rlm
