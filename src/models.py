from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PatientRecord:
    patient_id: str
    name: str
    age: int | None = None
    gender: str | None = None
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    summary: str | None = None
    raw_text: str | None = None
    source: str = "records.xlsx"
    metadata: dict[str, Any] = field(default_factory=dict)

    def searchable_text(self) -> str:
        parts = [
            self.name,
            str(self.age or ""),
            self.gender or "",
            self.phone or "",
            self.address or "",
            self.summary or "",
            self.raw_text or "",
        ]
        return "\n".join(part for part in parts if part).strip()


@dataclass
class ToolResult:
    tool: str
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
