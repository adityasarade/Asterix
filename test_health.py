import asyncio
from src.memgpt.utils.health import check_service_health

async def test_health():
    print("=== Service Health Test ===")
    results = await check_service_health()
    
    for service, result in results.items():
        status = "✅" if result.status == "healthy" else "❌"
        print(f"{status} {service}: {result.status}")
        if result.error:
            print(f"   Error: {result.error}")
        if result.response_time:
            print(f"   Response Time: {result.response_time:.3f}s")

if __name__ == "__main__":
    asyncio.run(test_health())