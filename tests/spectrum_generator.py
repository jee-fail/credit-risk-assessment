"""
Deterministic generation of credit-risk test applicants across the risk spectrum.

risk_level in [0, 1]:
  0.0 = designed "safest" profile
  1.0 = designed "riskiest" profile

Profiles interpolate continuous drivers (EXT_SOURCE, leverage, age, employment)
and pick categorical labels from ordered tiers. Used by the 1000-case suite.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _pick(tier: float, options: list[Any]) -> Any:
    """Map tier in [0,1] to a discrete option (0=safest, 1=riskiest)."""
    if not options:
        return None
    idx = int(np.clip(np.floor(tier * len(options)), 0, len(options) - 1))
    return options[idx]


# Ordered safest → riskiest
_EDUCATION = [
    "Higher education",
    "Academic degree",
    "Incomplete higher",
    "Secondary / secondary special",
    "Lower secondary",
]
_INCOME_TYPE = [
    "State servant",
    "Commercial associate",
    "Working",
    "Pensioner",
    "Unemployed",
]
_FAMILY = [
    "Married",
    "Civil marriage",
    "Separated",
    "Widow",
    "Single / not married",
]
_HOUSING = [
    "House / apartment",
    "Municipal apartment",
    "Rented apartment",
    "Office apartment",
    "With parents",
]
_OCCUPATION = [
    "Managers",
    "Accountants",
    "Core staff",
    "Sales staff",
    "Drivers",
    "Laborers",
    "Low-skill Laborers",
]
_ORG = [
    "School",
    "Government",
    "Medicine",
    "Business Entity Type 3",
    "Self-employed",
    "XNA",
]


def make_spectrum_case(
    case_id: int,
    risk_level: float,
    *,
    rng: np.random.RandomState | None = None,
    jitter: float = 0.04,
) -> dict[str, Any]:
    """
    Build one applicant dict.

    Continuous fields interpolate from safe→risky by risk_level.
    Small deterministic jitter (if rng given) spreads the cloud without
    destroying the spectrum ordering in aggregate.
    """
    t = float(np.clip(risk_level, 0.0, 1.0))
    if rng is not None and jitter > 0:
        t = float(np.clip(t + rng.uniform(-jitter, jitter), 0.0, 1.0))

    # External scores: high = safer in Home Credit
    ext = _lerp(0.88, 0.05, t)
    # Leverage: high credit / low income = riskier
    income = _lerp(320_000, 28_000, t)
    credit = _lerp(180_000, 1_100_000, t)
    goods = credit * _lerp(1.0, 0.95, t)
    annuity = credit * _lerp(0.06, 0.09, t)

    age_years = _lerp(52, 21, t)
    years_employed = _lerp(15.0, 0.15, t)
    children = int(round(_lerp(0, 4, t)))
    region_rating = int(round(_lerp(1, 3, t)))

    days_birth = -int(age_years * 365.25)
    if years_employed <= 0.05:
        days_employed = 365243  # Home Credit unemployed sentinel
    else:
        days_employed = -int(years_employed * 365.25)

    own_car = "Y" if t < 0.45 else "N"
    own_realty = "Y" if t < 0.50 else "N"
    gender = "F" if t < 0.55 else "M"

    case = {
        "case_id": int(case_id),
        "risk_level": float(risk_level),
        "risk_level_effective": t,
        "AMT_INCOME_TOTAL": float(income),
        "AMT_CREDIT": float(credit),
        "AMT_ANNUITY": float(annuity),
        "AMT_GOODS_PRICE": float(goods),
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "CNT_CHILDREN": children,
        "CNT_FAM_MEMBERS": children + (2 if t < 0.5 else 1),
        "EXT_SOURCE_1": float(np.clip(ext + (0.0 if rng is None else rng.normal(0, 0.02)), 0.01, 0.99)),
        "EXT_SOURCE_2": float(np.clip(ext + (0.0 if rng is None else rng.normal(0, 0.02)), 0.01, 0.99)),
        "EXT_SOURCE_3": float(np.clip(ext + (0.0 if rng is None else rng.normal(0, 0.02)), 0.01, 0.99)),
        "REGION_RATING_CLIENT": region_rating,
        "REGION_RATING_CLIENT_W_CITY": region_rating,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_EDUCATION_TYPE": _pick(t, _EDUCATION),
        "NAME_INCOME_TYPE": _pick(t, _INCOME_TYPE),
        "NAME_FAMILY_STATUS": _pick(t, _FAMILY),
        "NAME_HOUSING_TYPE": _pick(t, _HOUSING),
        "OCCUPATION_TYPE": _pick(t, _OCCUPATION),
        "ORGANIZATION_TYPE": _pick(t, _ORG),
        "WEEKDAY_APPR_PROCESS_START": "TUESDAY",
        "HOUR_APPR_PROCESS_START": int(_lerp(14, 3, t)),
        "FLAG_DOCUMENT_3": 1 if t < 0.6 else 0,
        "DAYS_ID_PUBLISH": -int(_lerp(4000, 200, t)),
        "DAYS_REGISTRATION": -int(_lerp(8000, 300, t)),
    }
    return case


def generate_spectrum_cases(
    n: int = 1000,
    *,
    seed: int = 42,
    include_anchors: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate ``n`` cases spanning risk_level in [0, 1].

    If include_anchors, force first/last few to pure safe/risky corners so
    spectrum checks always have extreme poles.
    """
    rng = np.random.RandomState(seed)
    cases: list[dict[str, Any]] = []

    # Evenly spaced primary grid
    levels = np.linspace(0.0, 1.0, n)
    for i, level in enumerate(levels):
        cases.append(make_spectrum_case(i, float(level), rng=rng, jitter=0.03))

    if include_anchors and n >= 10:
        # Overwrite poles with zero-jitter extremes for clean logic checks
        cases[0] = make_spectrum_case(0, 0.0, rng=None, jitter=0.0)
        cases[1] = make_spectrum_case(1, 0.05, rng=None, jitter=0.0)
        cases[-2] = make_spectrum_case(n - 2, 0.95, rng=None, jitter=0.0)
        cases[-1] = make_spectrum_case(n - 1, 1.0, rng=None, jitter=0.0)

    return cases


def features_only(case: dict[str, Any]) -> dict[str, Any]:
    """Drop meta keys for model input."""
    skip = {"case_id", "risk_level", "risk_level_effective"}
    return {k: v for k, v in case.items() if k not in skip}
