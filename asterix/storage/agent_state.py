"""
Asterix Agent State Storage Backends

Provides storage backends for agent state persistence:
- JSON backend (default, simple)
- SQLite backend (better for many agents)
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ============================================================================
# SQLite Backend
# ============================================================================

class SQLiteStateBackend:
    """
    SQLite backend for agent state storage.
    
    Better than JSON for:
    - Managing many agents (hundreds+)
    - Querying agents by properties
    - Atomic updates
    - Built-in indexing
    
    Schema:
        agents table:
        - agent_id (TEXT PRIMARY KEY)
        - state_json (TEXT) - full state as JSON
        - created_at (TEXT)
        - last_updated (TEXT)
        - model (TEXT) - for querying
        - block_count (INTEGER) - for querying
        - message_count (INTEGER) - for querying
    """
    
    def __init__(self, db_path: str = "agents.db"):
        """
        Initialize SQLite backend.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._ensure_database()
    
    def _ensure_database(self):
        """Create database and tables if they don't exist."""
        # Create directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect and create table
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                model TEXT,
                block_count INTEGER,
                message_count INTEGER
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_updated 
            ON agents(last_updated)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model 
            ON agents(model)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"SQLite database ready at {self.db_path}")
    
    def save(self, agent_id: str, state_dict: Dict[str, Any]):
        """
        Save agent state to database.
        
        Args:
            agent_id: Agent identifier
            state_dict: State dictionary from _to_state_dict()
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Serialize state to JSON
            state_json = json.dumps(state_dict, ensure_ascii=False)
            
            # Extract metadata for indexing
            created_at = state_dict.get("created_at")
            last_updated = state_dict.get("last_updated", datetime.now(timezone.utc).isoformat())
            model = state_dict.get("config", {}).get("model")
            block_count = len(state_dict.get("blocks", {}))
            message_count = len(state_dict.get("conversation_history", []))
            
            # Insert or replace
            cursor.execute("""
                INSERT OR REPLACE INTO agents 
                (agent_id, state_json, created_at, last_updated, model, block_count, message_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (agent_id, state_json, created_at, last_updated, model, block_count, message_count))
            
            conn.commit()
            logger.info(f"Saved agent '{agent_id}' to SQLite database")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save agent to SQLite: {e}")
            raise
        finally:
            conn.close()
    
    def load(self, agent_id: str) -> Dict[str, Any]:
        """
        Load agent state from database.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            State dictionary for _from_state_dict()
            
        Raises:
            FileNotFoundError: If agent not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT state_json FROM agents WHERE agent_id = ?
            """, (agent_id,))
            
            row = cursor.fetchone()
            
            if not row:
                raise FileNotFoundError(
                    f"Agent '{agent_id}' not found in database.\n"
                    f"Available agents: {', '.join(self.list_agents())}"
                )
            
            state_json = row[0]
            state_dict = json.loads(state_json)
            
            logger.info(f"Loaded agent '{agent_id}' from SQLite database")
            
            return state_dict
            
        finally:
            conn.close()
    
    def exists(self, agent_id: str) -> bool:
        """
        Check if agent exists in database.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if agent exists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 1 FROM agents WHERE agent_id = ? LIMIT 1
            """, (agent_id,))
            
            return cursor.fetchone() is not None
            
        finally:
            conn.close()
    
    def delete(self, agent_id: str) -> bool:
        """
        Delete agent from database.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM agents WHERE agent_id = ?
            """, (agent_id,))
            
            deleted = cursor.rowcount > 0
            conn.commit()
            
            if deleted:
                logger.info(f"Deleted agent '{agent_id}' from database")
            
            return deleted
            
        finally:
            conn.close()
    
    def list_agents(self, limit: Optional[int] = None) -> List[str]:
        """
        List all agent IDs in database.
        
        Args:
            limit: Maximum number of agents to return
            
        Returns:
            List of agent IDs
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if limit:
                cursor.execute("""
                    SELECT agent_id FROM agents 
                    ORDER BY last_updated DESC 
                    LIMIT ?
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT agent_id FROM agents 
                    ORDER BY last_updated DESC
                """)
            
            return [row[0] for row in cursor.fetchall()]
            
        finally:
            conn.close()
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about an agent without loading full state.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent metadata or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT agent_id, created_at, last_updated, model, block_count, message_count
                FROM agents WHERE agent_id = ?
            """, (agent_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return {
                "agent_id": row[0],
                "created_at": row[1],
                "last_updated": row[2],
                "model": row[3],
                "block_count": row[4],
                "message_count": row[5]
            }
            
        finally:
            conn.close()
    
    def list_all_info(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List metadata for all agents.
        
        Args:
            limit: Maximum number of agents
            
        Returns:
            List of agent metadata dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if limit:
                cursor.execute("""
                    SELECT agent_id, created_at, last_updated, model, block_count, message_count
                    FROM agents 
                    ORDER BY last_updated DESC 
                    LIMIT ?
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT agent_id, created_at, last_updated, model, block_count, message_count
                    FROM agents 
                    ORDER BY last_updated DESC
                """)
            
            return [
                {
                    "agent_id": row[0],
                    "created_at": row[1],
                    "last_updated": row[2],
                    "model": row[3],
                    "block_count": row[4],
                    "message_count": row[5]
                }
                for row in cursor.fetchall()
            ]
            
        finally:
            conn.close()