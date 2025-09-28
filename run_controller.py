#!/usr/bin/env python3
"""
MemGPT Controller Startup Script

Starts the FastAPI controller with proper configuration and logging.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn
from src.memgpt.utils.config import get_config
from src.memgpt.utils.health import check_service_health
from src.memgpt.controller.api import app


async def verify_services_before_start():
    """Verify all services are healthy before starting the controller"""
    print("🔍 Verifying service health before starting controller...")
    
    try:
        # Check service health
        health_results = await check_service_health()
        
        print("\nService Health Status:")
        all_healthy = True
        for service, result in health_results.items():
            status = "✅" if result.status == "healthy" else "❌"
            print(f"  {status} {service}: {result.status}")
            if result.error:
                print(f"     Error: {result.error}")
            if result.response_time:
                print(f"     Response Time: {result.response_time:.3f}s")
            
            if result.status != "healthy":
                all_healthy = False
        
        if not all_healthy:
            print("\n⚠️  Warning: Some services are unhealthy!")
            print("   The controller will still start, but some features may not work properly.")
            print("   Please check the service configurations and health status.")
        else:
            print("\n✅ All services are healthy!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Service health check failed: {e}")
        print("   The controller will still start, but please verify services manually.")
        return False


def setup_logging():
    """Configure logging for the controller"""
    config = get_config()
    logging_config = config.get_logging_config()
    
    # Configure logging
    log_level = getattr(logging, logging_config["level"].upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            # Add file handler if needed
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


def print_startup_info():
    """Print startup information"""
    config = get_config()
    controller_config = config.get_controller_config()
    
    print("🚀 MemGPT Controller Starting")
    print("=" * 40)
    print(f"Host: {controller_config.host}")
    print(f"Port: {controller_config.port}")
    print(f"Max Concurrent Requests: {controller_config.max_concurrent_requests}")
    print(f"Heartbeat Max Steps: {controller_config.heartbeat_max_steps}")
    print(f"Log Level: {config.get_logging_config()['level']}")
    print("=" * 40)
    
    print("\n📋 Available Endpoints:")
    print("  GET  /health              - Basic health check")
    print("  GET  /health/services     - Detailed service health")
    print("  POST /chat               - Main chat endpoint")
    print("  POST /agents             - Create new agent")
    print("  GET  /agents             - List all agents")
    print("  GET  /docs               - API documentation")
    
    print(f"\n🌐 Controller will be available at: http://{controller_config.host}:{controller_config.port}")
    print(f"📚 API docs will be available at: http://{controller_config.host}:{controller_config.port}/docs")


async def main():
    """Main startup function"""
    try:
        # Setup logging
        setup_logging()
        
        # Print startup info
        print_startup_info()
        
        # Verify services
        await verify_services_before_start()
        
        # Get configuration
        config = get_config()
        controller_config = config.get_controller_config()
        logging_config = config.get_logging_config()
        
        print(f"\n🚀 Starting controller on {controller_config.host}:{controller_config.port}...")
        print("   Press Ctrl+C to stop the server")
        print()
        
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app,
            host=controller_config.host,
            port=controller_config.port,
            log_level=logging_config["level"].lower(),
            access_log=True,
            reload=False,  # Set to True for development
            workers=1      # Single worker for development
        )
        
        # Start the server
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Controller stopped by user")
    except Exception as e:
        print(f"\n❌ Controller startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if we're in the right directory
    if not Path("src/memgpt").exists():
        print("❌ Error: Please run this script from the memgpt project root directory")
        print("   Current directory should contain: src/memgpt/")
        sys.exit(1)
    
    # Check for .env file
    if not Path(".env").exists():
        print("⚠️  Warning: .env file not found")
        print("   Make sure your environment variables are configured")
        print("   You can copy .env.example to .env and fill in your values")
        print()
    
    # Run the controller
    asyncio.run(main())