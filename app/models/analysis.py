from enum import Enum
from pydantic import BaseModel


class RiskLabel(str, Enum):
    ok = "OK"
    risk = "Риск"


class SourceRef(BaseModel):
    type: str          # "ПП ВС РФ", "КС РФ", "Доктрина"
    number: str        # № постановления / определения
    short_title: str   # краткое описание
    url: str | None = None


class FragmentAnalysis(BaseModel):
    fragment_text: str
    label: RiskLabel
    comment: str               # коротко: что не так / где риск
    correct_position: str      # краткая суть корректной позиции
    sources: list[SourceRef]   # хотя бы один источник


class DocumentAnalysis(BaseModel):
    topic: str                 # например, "госпошлина"
    fragments: list[FragmentAnalysis]
