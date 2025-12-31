"""
Supervision service for managing AI agent autonomy levels.

Provides access to supervision settings and pending action queue.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import PROJECT_ROOT


@dataclass
class SupervisionLevel:
    """Supervision level configuration."""
    level: int  # 0-4
    name: str
    description: str
    updated_at: Optional[str] = None
    updated_by: str = 'gui'


@dataclass
class PendingAction:
    """An action awaiting human approval."""
    id: str
    action_type: str  # 'stage_run', 'file_write', 'commit', etc.
    description: str
    details: dict
    created_at: str
    agent_id: Optional[str] = None


@dataclass
class AuditEntry:
    """An audit log entry for agent actions."""
    id: str
    timestamp: str
    action: str
    agent_id: Optional[str]
    result: str  # 'approved', 'rejected', 'auto_executed', 'failed'
    details: Optional[str] = None


# Level definitions
SUPERVISION_LEVELS = {
    0: SupervisionLevel(
        level=0,
        name="Human Only",
        description="All actions executed manually by human. AI provides no assistance."
    ),
    1: SupervisionLevel(
        level=1,
        name="AI Suggests",
        description="AI suggests actions, human approves everything before execution."
    ),
    2: SupervisionLevel(
        level=2,
        name="AI Proposes",
        description="AI proposes specific actions, human reviews before execution."
    ),
    3: SupervisionLevel(
        level=3,
        name="AI Executes",
        description="AI executes autonomously, human monitors results."
    ),
    4: SupervisionLevel(
        level=4,
        name="Fully Autonomous",
        description="AI operates independently, human handles exceptions only."
    ),
}


class SupervisionService:
    """Service for managing AI supervision settings."""

    def __init__(self):
        self.config_dir = PROJECT_ROOT / ".centaur"
        self.config_file = self.config_dir / "supervision.json"
        self.audit_file = self.config_dir / "audit_log.jsonl"
        self._current_level: int = 1  # Default to Level 1
        self._pending_actions: list[PendingAction] = []
        self._load_config()

    def _load_config(self) -> None:
        """Load supervision config from disk."""
        if self.config_file.exists():
            try:
                data = json.loads(self.config_file.read_text())
                self._current_level = data.get('level', 1)
            except (json.JSONDecodeError, KeyError):
                self._current_level = 1

    def _save_config(self) -> None:
        """Save supervision config to disk."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        data = {
            'level': self._current_level,
            'updated_at': datetime.now().isoformat(),
            'updated_by': 'gui',
        }
        self.config_file.write_text(json.dumps(data, indent=2))

    def get_level(self) -> SupervisionLevel:
        """Get current supervision level."""
        level_info = SUPERVISION_LEVELS.get(self._current_level, SUPERVISION_LEVELS[1])
        level_info.updated_at = None

        if self.config_file.exists():
            try:
                data = json.loads(self.config_file.read_text())
                level_info.updated_at = data.get('updated_at')
                level_info.updated_by = data.get('updated_by', 'gui')
            except (json.JSONDecodeError, KeyError):
                pass

        return level_info

    def set_level(self, level: int) -> SupervisionLevel:
        """
        Set supervision level.

        Parameters
        ----------
        level : int
            Supervision level (0-4)

        Returns
        -------
        SupervisionLevel
            Updated level info

        Raises
        ------
        ValueError
            If level is out of range
        """
        if level < 0 or level > 4:
            raise ValueError(f"Level must be 0-4, got {level}")

        self._current_level = level
        self._save_config()

        # Log the change
        self._log_audit(
            action=f"supervision_level_changed",
            result="approved",
            details=f"Level set to {level} ({SUPERVISION_LEVELS[level].name})"
        )

        return self.get_level()

    def get_pending_actions(self) -> list[PendingAction]:
        """Get list of actions awaiting approval."""
        # In a real implementation, this would read from a queue
        # For now, return the in-memory list
        return self._pending_actions

    def add_pending_action(self, action: PendingAction) -> None:
        """Add an action to the pending queue."""
        self._pending_actions.append(action)

    def approve_action(self, action_id: str) -> bool:
        """Approve a pending action."""
        for i, action in enumerate(self._pending_actions):
            if action.id == action_id:
                self._pending_actions.pop(i)
                self._log_audit(
                    action=f"action_approved:{action.action_type}",
                    result="approved",
                    details=action.description
                )
                return True
        return False

    def reject_action(self, action_id: str, reason: str = "") -> bool:
        """Reject a pending action."""
        for i, action in enumerate(self._pending_actions):
            if action.id == action_id:
                self._pending_actions.pop(i)
                self._log_audit(
                    action=f"action_rejected:{action.action_type}",
                    result="rejected",
                    details=f"{action.description} - Reason: {reason}"
                )
                return True
        return False

    def _log_audit(
        self,
        action: str,
        result: str,
        details: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> None:
        """Log an action to the audit file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        entry = AuditEntry(
            id=datetime.now().strftime("%Y%m%d%H%M%S%f"),
            timestamp=datetime.now().isoformat(),
            action=action,
            agent_id=agent_id,
            result=result,
            details=details,
        )

        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')

    def get_audit_log(self, limit: int = 50) -> list[AuditEntry]:
        """Get recent audit log entries."""
        if not self.audit_file.exists():
            return []

        entries = []
        try:
            with open(self.audit_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        entries.append(AuditEntry(**data))
        except (json.JSONDecodeError, KeyError):
            return []

        # Return most recent entries
        return list(reversed(entries[-limit:]))

    def clear_audit_log(self) -> int:
        """Clear the audit log."""
        if not self.audit_file.exists():
            return 0

        # Count entries before clearing
        count = 0
        with open(self.audit_file, 'r') as f:
            count = sum(1 for line in f if line.strip())

        self.audit_file.unlink()
        return count


# Singleton instance
_supervision_service: Optional[SupervisionService] = None


def get_supervision_service() -> SupervisionService:
    """Get the supervision service singleton."""
    global _supervision_service
    if _supervision_service is None:
        _supervision_service = SupervisionService()
    return _supervision_service
