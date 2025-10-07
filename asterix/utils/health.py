"""
MemGPT Service Health Monitoring

Monitors the health and availability of all external services:
- Letta server
- Qdrant Cloud
- Embedding services (OpenAI/SentenceTransformers)

Provides immediate failure reporting with specific reasons.
"""

import asyncio
import time
import httpx
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

from letta_client import Letta
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
import openai
from sentence_transformers import SentenceTransformer

from .config import get_config, ServiceStatus

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Service type enumeration"""
    LETTA = "letta"
    QDRANT = "qdrant"
    OPENAI_EMBEDDINGS = "openai_embeddings"
    OPENAI_LLM = "openai_llm"
    GROQ = "groq"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


@dataclass
class HealthCheckResult:
    """Result of a health check operation"""
    service: str
    status: str  # "healthy", "unhealthy", "unknown"
    response_time: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class ServiceHealthMonitor:
    """
    Monitors health and availability of all external services.
    
    Provides both on-demand and periodic health checking with detailed
    error reporting for immediate troubleshooting.
    """
    
    def __init__(self):
        self.config = get_config()
        self._service_statuses: Dict[str, ServiceStatus] = {}
        self._last_check_times: Dict[str, float] = {}
        self._check_intervals = {
            ServiceType.LETTA.value: 30,
            ServiceType.QDRANT.value: 60,
            ServiceType.OPENAI_EMBEDDINGS.value: 45,
            ServiceType.OPENAI_LLM.value: 45,
            ServiceType.GROQ.value: 45,
            ServiceType.SENTENCE_TRANSFORMERS.value: 120
        }
    
    async def check_letta_health(self) -> HealthCheckResult:
        """Check Letta server health and availability."""
        start_time = time.time()
        
        try:
            letta_config = self.config.get_letta_config()
            
            # Test Letta client functionality directly (no health endpoint)
            try:
                client = Letta(base_url=letta_config['url'])
                
                # Test basic functionality
                models = client.models.list()
                
                response_time = time.time() - start_time
                
                return HealthCheckResult(
                    service=ServiceType.LETTA.value,
                    status="healthy",
                    response_time=response_time,
                    details={
                        "models_available": len(models),
                        "server_url": letta_config['url'],
                        "client_connected": True
                    }
                )
                
            except Exception as client_error:
                return HealthCheckResult(
                    service=ServiceType.LETTA.value,
                    status="unhealthy", 
                    response_time=time.time() - start_time,
                    error=f"Letta client error: {str(client_error)[:200]}"
                )
                
        except Exception as e:
            return HealthCheckResult(
                service=ServiceType.LETTA.value,
                status="unhealthy",
                response_time=time.time() - start_time,
                error=f"Connection failed: {str(e)[:200]}"
            )
    
    async def check_qdrant_health(self) -> HealthCheckResult:
        """Check Qdrant Cloud health and connectivity."""
        start_time = time.time()
        
        try:
            qdrant_config = self.config.get_qdrant_config()
            
            # Test Qdrant client connectivity
            client = QdrantClient(
                url=qdrant_config.url,
                api_key=qdrant_config.api_key,
                timeout=qdrant_config.timeout
            )
            
            # Test basic operations
            collections = client.get_collections()
            
            response_time = time.time() - start_time
            
            # Check if our collection exists
            collection_exists = any(
                c.name == qdrant_config.collection_name 
                for c in collections.collections
            )
            
            return HealthCheckResult(
                service=ServiceType.QDRANT.value,
                status="healthy",
                response_time=response_time,
                details={
                    "collections_count": len(collections.collections),
                    "target_collection_exists": collection_exists,
                    "target_collection_name": qdrant_config.collection_name,
                    "url": qdrant_config.url
                }
            )
            
        except ResponseHandlingException as e:
            return HealthCheckResult(
                service=ServiceType.QDRANT.value,
                status="unhealthy",
                response_time=time.time() - start_time,
                error=f"Qdrant API error: {str(e)[:200]}"
            )
        except Exception as e:
            return HealthCheckResult(
                service=ServiceType.QDRANT.value,
                status="unhealthy",
                response_time=time.time() - start_time,
                error=f"Qdrant connection failed: {str(e)[:200]}"
            )
    
    async def check_openai_health(self, service_type: str = "embeddings") -> HealthCheckResult:
        """
        Check OpenAI API health for embeddings or LLM.
        
        Args:
            service_type: "embeddings" or "llm"
            
        Returns:
            Health check result with detailed status
        """
        start_time = time.time()
        service_name = f"openai_{service_type}"
        
        try:
            # Get OpenAI API key
            api_key = self.config.get_env("OPENAI_API_KEY")
            if not api_key:
                return HealthCheckResult(
                    service=service_name,
                    status="unhealthy", 
                    response_time=time.time() - start_time,
                    error="OpenAI API key not configured"
                )
            
            client = openai.OpenAI(api_key=api_key)
            
            if service_type == "embeddings":
                # Test embeddings endpoint
                embed_config = self.config.get_embedding_config()
                if embed_config.provider == "openai":
                    response = client.embeddings.create(
                        model=embed_config.model,
                        input="health check test"
                    )
                    
                    return HealthCheckResult(
                        service=service_name,
                        status="healthy",
                        response_time=time.time() - start_time,
                        details={
                            "model": embed_config.model,
                            "dimensions": len(response.data[0].embedding),
                            "usage_tokens": response.usage.total_tokens
                        }
                    )
                else:
                    return HealthCheckResult(
                        service=service_name,
                        status="healthy",
                        response_time=time.time() - start_time,
                        details={"provider": "not_active", "active_provider": embed_config.provider}
                    )
            
            elif service_type == "llm":
                # Test chat completions endpoint
                llm_config = self.config.get_llm_config("openai")
                response = client.chat.completions.create(
                    model=llm_config.model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                
                return HealthCheckResult(
                    service=service_name,
                    status="healthy",
                    response_time=time.time() - start_time,
                    details={
                        "model": llm_config.model,
                        "usage_tokens": response.usage.total_tokens
                    }
                )
                
        except openai.AuthenticationError:
            return HealthCheckResult(
                service=service_name,
                status="unhealthy",
                response_time=time.time() - start_time,
                error="OpenAI authentication failed - check API key"
            )
        except openai.RateLimitError:
            return HealthCheckResult(
                service=service_name,
                status="unhealthy",
                response_time=time.time() - start_time,
                error="OpenAI rate limit exceeded"
            )
        except Exception as e:
            return HealthCheckResult(
                service=service_name,
                status="unhealthy",
                response_time=time.time() - start_time,
                error=f"OpenAI {service_type} error: {str(e)[:200]}"
            )
    
    async def check_groq_health(self) -> HealthCheckResult:
        """
        Check Groq API health and connectivity.
        
        Returns:
            Health check result with detailed status
        """
        start_time = time.time()
        
        try:
            # Get Groq API key
            api_key = self.config.get_env("GROQ_API_KEY")
            if not api_key:
                return HealthCheckResult(
                    service=ServiceType.GROQ.value,
                    status="unhealthy",
                    response_time=time.time() - start_time,
                    error="Groq API key not configured"
                )
            
            # Test Groq API connectivity
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # Test with a minimal request
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return HealthCheckResult(
                        service=ServiceType.GROQ.value,
                        status="healthy",
                        response_time=time.time() - start_time,
                        details={
                            "model": data.get("model", "unknown"),
                            "usage_tokens": data.get("usage", {}).get("total_tokens", 0)
                        }
                    )
                else:
                    return HealthCheckResult(
                        service=ServiceType.GROQ.value,
                        status="unhealthy",
                        response_time=time.time() - start_time,
                        error=f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                service=ServiceType.GROQ.value,
                status="unhealthy",
                response_time=time.time() - start_time,
                error=f"Groq API error: {str(e)[:200]}"
            )
    
    async def check_sentence_transformers_health(self) -> HealthCheckResult:
        """
        Check Sentence Transformers availability.
        
        Returns:
            Health check result with detailed status
        """
        start_time = time.time()
        
        try:
            embed_config = self.config.get_embedding_config()
            
            if embed_config.provider != "sentence_transformers":
                return HealthCheckResult(
                    service=ServiceType.SENTENCE_TRANSFORMERS.value,
                    status="healthy",
                    response_time=time.time() - start_time,
                    details={"provider": "not_active", "active_provider": embed_config.provider}
                )
            
            # Test loading the model
            model = SentenceTransformer(embed_config.model)
            
            # Test encoding
            embeddings = model.encode(["health check test"])
            
            return HealthCheckResult(
                service=ServiceType.SENTENCE_TRANSFORMERS.value,
                status="healthy",
                response_time=time.time() - start_time,
                details={
                    "model": embed_config.model,
                    "dimensions": len(embeddings[0]),
                    "device": str(model.device)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                service=ServiceType.SENTENCE_TRANSFORMERS.value,
                status="unhealthy",
                response_time=time.time() - start_time,
                error=f"SentenceTransformers error: {str(e)[:200]}"
            )
    
    async def check_all_services(self, services: Optional[List[str]] = None) -> Dict[str, HealthCheckResult]:
        """
        Check health of all or specified services.
        
        Args:
            services: List of service names to check, or None for all
            
        Returns:
            Dictionary mapping service names to health check results
        """
        if services is None:
            services = [
                ServiceType.LETTA.value,
                ServiceType.QDRANT.value,
                ServiceType.OPENAI_EMBEDDINGS.value,
                ServiceType.GROQ.value,
                ServiceType.SENTENCE_TRANSFORMERS.value
            ]
        
        results = {}
        
        # Run health checks concurrently
        tasks = []
        for service in services:
            if service == ServiceType.LETTA.value:
                tasks.append(self.check_letta_health())
            elif service == ServiceType.QDRANT.value:
                tasks.append(self.check_qdrant_health())
            elif service == ServiceType.OPENAI_EMBEDDINGS.value:
                tasks.append(self.check_openai_health("embeddings"))
            elif service == ServiceType.OPENAI_LLM.value:
                tasks.append(self.check_openai_health("llm"))
            elif service == ServiceType.GROQ.value:
                tasks.append(self.check_groq_health())
            elif service == ServiceType.SENTENCE_TRANSFORMERS.value:
                tasks.append(self.check_sentence_transformers_health())
        
        # Wait for all checks to complete
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(task_results):
            service = services[i]
            if isinstance(result, Exception):
                results[service] = HealthCheckResult(
                    service=service,
                    status="unhealthy",
                    error=f"Health check failed: {str(result)[:200]}"
                )
            else:
                results[service] = result
        
        # Update internal status tracking
        for service, result in results.items():
            self._service_statuses[service] = ServiceStatus(
                name=service,
                available=(result.status == "healthy"),
                error=result.error,
                last_check=result.timestamp,
                response_time=result.response_time
            )
            self._last_check_times[service] = time.time()
        
        return results
    
    def get_service_status(self, service: str) -> Optional[ServiceStatus]:
        """
        Get current status of a specific service.
        
        Args:
            service: Service name
            
        Returns:
            Service status or None if not found
        """
        return self._service_statuses.get(service)
    
    def get_all_service_statuses(self) -> Dict[str, ServiceStatus]:
        """
        Get current status of all tracked services.
        
        Returns:
            Dictionary mapping service names to their status
        """
        return self._service_statuses.copy()
    
    def is_service_healthy(self, service: str) -> bool:
        """
        Check if a service is currently healthy.
        
        Args:
            service: Service name
            
        Returns:
            True if service is healthy, False otherwise
        """
        status = self._service_statuses.get(service)
        return status is not None and status.available
    
    def get_unhealthy_services(self) -> List[Tuple[str, str]]:
        """
        Get list of unhealthy services with their error messages.
        
        Returns:
            List of tuples (service_name, error_message)
        """
        unhealthy = []
        for service, status in self._service_statuses.items():
            if not status.available:
                unhealthy.append((service, status.error or "Unknown error"))
        return unhealthy
    
    def needs_health_check(self, service: str) -> bool:
        """
        Check if a service needs a health check based on interval.
        
        Args:
            service: Service name
            
        Returns:
            True if health check is needed
        """
        last_check = self._last_check_times.get(service, 0)
        interval = self._check_intervals.get(service, 60)
        return time.time() - last_check > interval
    
    async def ensure_services_healthy(self, required_services: List[str]) -> Tuple[bool, List[str]]:
        """
        Ensure specified services are healthy, checking if necessary.
        
        Args:
            required_services: List of service names that must be healthy
            
        Returns:
            Tuple of (all_healthy: bool, error_messages: List[str])
        """
        # Check which services need health checks
        services_to_check = [
            service for service in required_services 
            if self.needs_health_check(service) or not self.is_service_healthy(service)
        ]
        
        # Perform health checks if needed
        if services_to_check:
            await self.check_all_services(services_to_check)
        
        # Verify all required services are healthy
        errors = []
        all_healthy = True
        
        for service in required_services:
            if not self.is_service_healthy(service):
                all_healthy = False
                status = self._service_statuses.get(service)
                error_msg = f"{service}: {status.error if status else 'Unknown status'}"
                errors.append(error_msg)
        
        return all_healthy, errors
    
    def get_service_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all service statuses.
        
        Returns:
            Summary dictionary with counts and details
        """
        total_services = len(self._service_statuses)
        healthy_services = sum(1 for status in self._service_statuses.values() if status.available)
        unhealthy_services = total_services - healthy_services
        
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": unhealthy_services,
            "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0,
            "services": {
                name: {
                    "status": "healthy" if status.available else "unhealthy",
                    "last_check": status.last_check,
                    "response_time": status.response_time,
                    "error": status.error
                }
                for name, status in self._service_statuses.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global health monitor instance
health_monitor = ServiceHealthMonitor()


async def check_service_health(services: Optional[List[str]] = None) -> Dict[str, HealthCheckResult]:
    """
    Convenience function to check service health.
    
    Args:
        services: List of service names to check, or None for all
        
    Returns:
        Dictionary mapping service names to health check results
    """
    return await health_monitor.check_all_services(services)


async def ensure_required_services(services: List[str]) -> Tuple[bool, List[str]]:
    """
    Convenience function to ensure required services are healthy.
    
    Args:
        services: List of required service names
        
    Returns:
        Tuple of (all_healthy: bool, error_messages: List[str])
    """
    return await health_monitor.ensure_services_healthy(services)


def get_service_status_summary() -> Dict[str, Any]:
    """
    Convenience function to get service status summary.
    
    Returns:
        Service status summary dictionary
    """
    return health_monitor.get_service_summary()


# Service health check tool for agents
class ServiceHealthTool:
    """Tool for agents to check service health"""
    
    @staticmethod
    async def service_health_check(services: List[str] = None) -> Dict[str, Any]:
        """
        Tool function for agents to check service health.
        
        Args:
            services: List of services to check (letta, qdrant, embeddings)
            
        Returns:
            Service health status with detailed information
        """
        if services is None:
            services = ["letta", "qdrant", "embeddings"]
        
        # Map generic service names to specific service types
        service_mapping = {
            "letta": ServiceType.LETTA.value,
            "qdrant": ServiceType.QDRANT.value,
            "embeddings": ServiceType.OPENAI_EMBEDDINGS.value,
            "groq": ServiceType.GROQ.value,
            "openai": ServiceType.OPENAI_LLM.value
        }
        
        # Convert to specific service names
        specific_services = []
        for service in services:
            if service in service_mapping:
                specific_services.append(service_mapping[service])
            else:
                specific_services.append(service)
        
        # Check service health
        results = await health_monitor.check_all_services(specific_services)
        
        # Format results for agent consumption
        formatted_results = {}
        for service, result in results.items():
            formatted_results[service] = {
                "status": result.status,
                "available": result.status == "healthy",
                "response_time_ms": round(result.response_time * 1000) if result.response_time else None,
                "error": result.error,
                "details": result.details,
                "last_check": result.timestamp
            }
        
        # Add summary
        healthy_count = sum(1 for r in results.values() if r.status == "healthy")
        total_count = len(results)
        
        return {
            "services": formatted_results,
            "summary": {
                "total_services": total_count,
                "healthy_services": healthy_count,
                "all_healthy": healthy_count == total_count,
                "health_percentage": round((healthy_count / total_count) * 100, 1) if total_count > 0 else 0
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }