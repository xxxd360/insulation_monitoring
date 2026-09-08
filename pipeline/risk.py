from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .config import RiskConfig
from .detection import Detection


@dataclass
class RiskAssessment:
    level: int
    level_name: str
    reason: str
    max_defect_area_ratio: float
    defect_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def assess_risk(defects: Iterable[Detection], crop_area: float, config: RiskConfig) -> RiskAssessment:
    defects = list(defects)
    max_ratio = max((d.area / crop_area for d in defects), default=0.0) if crop_area > 0 else 0.0
    names = {d.class_name.casefold() for d in defects}
    high = {n.casefold() for n in config.high_risk_classes}
    warning = {n.casefold() for n in config.warning_classes}

    if not defects:
        return RiskAssessment(1, "正常", "未检出缺陷", 0.0, 0)
    if names & high or max_ratio >= config.high_risk_area:
        return RiskAssessment(4, "高风险", "检出高风险缺陷类别或缺陷覆盖范围较大", max_ratio, len(defects))
    if max_ratio >= config.warning_area:
        return RiskAssessment(3, "预警", "缺陷覆盖范围达到预警阈值", max_ratio, len(defects))
    if names & warning or max_ratio >= config.attention_area:
        return RiskAssessment(2, "关注", "检出需复核的缺陷或轻度异常", max_ratio, len(defects))
    return RiskAssessment(1, "正常", "缺陷面积低于当前启发式阈值", max_ratio, len(defects))
