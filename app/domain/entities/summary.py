from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class SummaryEntity:
    id: int
    file_id: int
    lookup_key: str
    text: str
    used_prompt: Optional[str]
    model_name: Optional[str]
    created_at: datetime
    updated_at: datetime