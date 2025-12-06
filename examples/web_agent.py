"""Web scraping agent example using procedural memory.

Demonstrates how an agent can learn extraction strategies for different website structures.

Run: python examples/web_agent.py
"""
import time
from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig


def simulate_web_scraping():
    """Simulate a web agent learning extraction strategies."""
    print("=== Web Scraping Agent with Procedural Memory ===\n")

    # Initialize memory
    memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))

    # Simulate agent encountering different website structures
    print("Agent encounters various website structures...")

    # E-commerce site with sidebar
    memory.store(Experience(
        env_features=["ecommerce", "sidebar", "grid_layout", "pagination"],
        goal="extract product prices",
        action="Use CSS selector '.product .price' with pagination",
        result="Extracted 200 prices across 8 pages",
        success=True,
        timestamp=time.time(),
        cost=0.15,  # API cost
    ))

    # Blog site with article structure
    memory.store(Experience(
        env_features=["blog", "article_layout", "comments"],
        goal="extract blog posts",
        action="Use CSS selector 'article .content' and filter by date",
        result="Extracted 50 posts",
        success=True,
        timestamp=time.time(),
        cost=0.08,
    ))

    # News site with complex structure
    memory.store(Experience(
        env_features=["news", "infinite_scroll", "dynamic_loading"],
        goal="extract articles",
        action="Scroll to trigger lazy loading, wait for elements",
        result="Failed: timeout waiting for elements",
        success=False,
        timestamp=time.time(),
        cost=0.20,
    ))

    # Retry with adjusted strategy
    memory.store(Experience(
        env_features=["news", "infinite_scroll", "dynamic_loading"],
        goal="extract articles",
        action="Use API endpoint instead of scraping",
        result="Extracted 100 articles via JSON API",
        success=True,
        timestamp=time.time(),
        cost=0.05,
    ))

    print(f"Stored {memory.size()} experiences\n")

    # New site encountered - recall similar strategies
    print("Agent encounters new e-commerce site with sidebar...")
    similar_experiences = memory.recall(
        env_features=["ecommerce", "sidebar", "grid_layout"],
        goal="extract product data",
        top_k=3
    )

    print(f"Recalled {len(similar_experiences)} relevant strategies:")
    for exp in similar_experiences:
        status = "✓" if exp.success else "✗"
        print(f"  {status} {exp.goal}")
        print(f"    Strategy: {exp.action}")
        print(f"    Result: {exp.result}")
        if exp.cost:
            print(f"    Cost: ${exp.cost:.2f}")
        print()

    # Filter for only successful strategies
    print("Filtering for successful strategies only...")
    successful = memory.recall(
        env_features=["ecommerce", "sidebar"],
        goal="extract product",
        top_k=3,
        config=MemoryConfig(storage_path=":memory:", require_success=True)
    )

    print(f"Found {len(successful)} successful strategies")


if __name__ == "__main__":
    simulate_web_scraping()
    print("\n✅ Web agent example complete!")
    print("\nKey takeaway: Agent learns from past successes/failures")
    print("and retrieves relevant strategies for new websites.")
