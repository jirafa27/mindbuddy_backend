from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SyncCommandEntity:
    id: int
    user_id: int
    user_file_id: Optional[int]
    command_type: str
    payload_json: dict = field(default_factory=dict)
    status: str = "pending"
    created_at: Optional[datetime] = None
    acked_at: Optional[datetime] = None
