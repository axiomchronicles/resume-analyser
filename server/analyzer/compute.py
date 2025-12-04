from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any

from .helpers import (
    clean_text,
    extract_bullets,
    coverage_score,
    keyword_match_score,
    starts_with_action_verb,
    contains_metric,
    bullet_quality_stats,
    readability_scores,
    passive_voice_ratio,
    first_person_ratio,
    estimate_experience_years,
    skill_coverage_score,
)


@dataclass(frozen=True)
class ATSWeights:
    section: float = 0.20
    keyword: float = 0.30
    action: float = 0.20
    metric: float = 0.15
    length: float = 0.15

    def normalized(self) -> "ATSWeights":
        total = self.section + self.keyword + self.action + self.metric + self.length
        if total <= 0:
            equal = 1.0 / 5
            return ATSWeights(equal, equal, equal, equal, equal)
        return ATSWeights(
            section=self.section / total,
            keyword=self.keyword / total,
            action=self.action / total,
            metric=self.metric / total,
            length=self.length / total,
        )


@dataclass(frozen=True)
class LengthConfig:
    min_wc: int = 200
    optimal_min_wc: int = 200
    optimal_max_wc: int = 800
    acceptable_max_wc: int = 1200

    too_short_score: float = 0.30
    optimal_score: float = 1.00
    slightly_long_score: float = 0.70
    too_long_score: float = 0.40


@dataclass(frozen=True)
class BulletFallbackConfig:
    action_score_no_bullets: float = 0.30
    metric_score_no_bullets: float = 0.20


@dataclass(frozen=True)
class ATSConfig:
    weights: ATSWeights = ATSWeights()
    length: LengthConfig = LengthConfig()
    bullet_fallbacks: BulletFallbackConfig = BulletFallbackConfig()
    max_final_score: float = 100.0


@dataclass
class ATSScores:
    final_score: float
    section_score: float
    keyword_score: float
    action_score: float
    metric_score: float
    length_score: float
    word_count: int
    bullets_count: int
    section_found: Dict[str, bool]
    bullets: List[str]
    explanation: Optional[Dict[str, Any]] = None
    readability: Dict[str, float] = field(default_factory=dict)
    bullet_quality: Dict[str, float] = field(default_factory=dict)
    passive_voice_ratio: float = 0.0
    first_person_ratio: float = 0.0
    estimated_experience_years: float = 0.0
    skill_coverage: float = 0.0

    def to_dict(self, include_explanation: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if not include_explanation:
            data.pop("explanation", None)
        return data


def _compute_length_score(word_count: int, cfg: LengthConfig) -> float:
    if word_count < cfg.min_wc:
        return cfg.too_short_score
    if cfg.optimal_min_wc <= word_count <= cfg.optimal_max_wc:
        return cfg.optimal_score
    if cfg.optimal_max_wc < word_count <= cfg.acceptable_max_wc:
        return cfg.slightly_long_score
    return cfg.too_long_score


def _compute_bullet_based_scores(
    bullets: List[str],
    fallbacks: BulletFallbackConfig,
) -> Tuple[float, float, Dict[str, Any]]:
    if not bullets:
        explanation = {
            "reason": "no_bullets_found",
            "action_score_fallback": fallbacks.action_score_no_bullets,
            "metric_score_fallback": fallbacks.metric_score_no_bullets,
        }
        return fallbacks.action_score_no_bullets, fallbacks.metric_score_no_bullets, explanation

    total_bullets = len(bullets)

    action_flags = [bool(starts_with_action_verb(b)) for b in bullets]
    metric_flags = [bool(contains_metric(b)) for b in bullets]

    action_score = sum(action_flags) / total_bullets
    metric_score = sum(metric_flags) / total_bullets

    explanation = {
        "total_bullets": total_bullets,
        "action_bullets": sum(action_flags),
        "metric_bullets": sum(metric_flags),
        "action_flags": action_flags,
        "metric_flags": metric_flags,
    }

    return action_score, metric_score, explanation


def compute_ats_scores(
    resume_text: str,
    jd_text: str = "",
    config: Optional[ATSConfig] = None,
    include_explanation: bool = False,
    required_skills: Optional[List[str]] = None,
) -> Dict:
    cfg = config or ATSConfig()
    weights = cfg.weights.normalized()

    cleaned_resume = clean_text(resume_text or "")
    bullets = extract_bullets(cleaned_resume) or []

    word_count = len(cleaned_resume.split())

    section_score_raw, section_found = coverage_score(cleaned_resume)
    keyword_score_raw = keyword_match_score(cleaned_resume, jd_text or "")

    action_score_raw, metric_score_raw, bullets_explanation = _compute_bullet_based_scores(
        bullets,
        cfg.bullet_fallbacks,
    )
    length_score_raw = _compute_length_score(word_count, cfg.length)

    readability = readability_scores(cleaned_resume)
    bullet_quality = bullet_quality_stats(bullets)
    pv_ratio = passive_voice_ratio(cleaned_resume)
    fp_ratio = first_person_ratio(cleaned_resume)
    exp_years = estimate_experience_years(cleaned_resume)
    skill_cov = (
        skill_coverage_score(cleaned_resume, required_skills or [])
        if required_skills
        else 0.0
    )

    final_raw = (
        section_score_raw * weights.section +
        keyword_score_raw * weights.keyword +
        action_score_raw * weights.action +
        metric_score_raw * weights.metric +
        length_score_raw * weights.length
    ) * 100

    final_capped = min(final_raw, cfg.max_final_score)

    explanation: Optional[Dict[str, Any]] = None
    if include_explanation:
        explanation = {
            "weights_used": asdict(weights),
            "raw_scores": {
                "section_score_raw": section_score_raw,
                "keyword_score_raw": keyword_score_raw,
                "action_score_raw": action_score_raw,
                "metric_score_raw": metric_score_raw,
                "length_score_raw": length_score_raw,
            },
            "length_config": asdict(cfg.length),
            "bullet_analysis": bullets_explanation,
            "readability": readability,
            "bullet_quality": bullet_quality,
            "passive_voice_ratio": pv_ratio,
            "first_person_ratio": fp_ratio,
            "estimated_experience_years": exp_years,
            "skill_coverage": skill_cov,
        }

    scores = ATSScores(
        final_score=round(final_capped, 1),
        section_score=round(section_score_raw * 100, 1),
        keyword_score=round(keyword_score_raw * 100, 1),
        action_score=round(action_score_raw * 100, 1),
        metric_score=round(metric_score_raw * 100, 1),
        length_score=round(length_score_raw * 100, 1),
        word_count=word_count,
        bullets_count=len(bullets),
        section_found=section_found,
        bullets=bullets,
        explanation=explanation,
        readability=readability,
        bullet_quality=bullet_quality,
        passive_voice_ratio=pv_ratio,
        first_person_ratio=fp_ratio,
        estimated_experience_years=exp_years,
        skill_coverage=skill_cov,
    )

    return scores.to_dict(include_explanation=include_explanation)
