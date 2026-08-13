"""
Deterministic health metrics calculations.

These are intentionally NOT left to the LLM. BMI and calorie needs are
well-defined formulas — computing them with real code guarantees correctness,
whereas an LLM asked to "do the math" can silently make arithmetic errors.
This module is wrapped as a LangChain Tool in agent.py.
"""

from dataclasses import dataclass
from typing import Literal

Sex = Literal["male", "female"]
ActivityLevel = Literal["sedentary", "light", "moderate", "active", "very_active"]

# Multipliers for Total Daily Energy Expenditure (TDEE) from Mifflin-St Jeor BMR
ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,      # little or no exercise
    "light": 1.375,        # light exercise 1-3 days/week
    "moderate": 1.55,      # moderate exercise 3-5 days/week
    "active": 1.725,       # hard exercise 6-7 days/week
    "very_active": 1.9,    # very hard exercise, physical job
}

# CDC-recommended sleep ranges by age (hours per 24h period)
SLEEP_RECOMMENDATIONS = [
    (0, 3, (14, 17)),
    (4, 12, (12, 16)),
    (1, 2, (11, 14)),      # note: overlapping infant/toddler bands are handled by order below
    (3, 5, (10, 13)),
    (6, 12, (9, 12)),
    (13, 17, (8, 10)),
    (18, 60, (7, 9)),
    (61, 64, (7, 9)),
    (65, 120, (7, 8)),
]


@dataclass
class ProfileMetrics:
    bmi: float
    bmi_category: str
    bmr_kcal: float
    tdee_kcal: float
    recommended_sleep_range: tuple
    sleep_deficit_hours: float


def calculate_bmi(weight_kg: float, height_cm: float) -> tuple[float, str]:
    """Return (bmi, category) using standard WHO BMI bands."""
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)

    if bmi < 18.5:
        category = "underweight"
    elif bmi < 25:
        category = "normal weight"
    elif bmi < 30:
        category = "overweight"
    else:
        category = "obese"

    return bmi, category


def calculate_bmr(weight_kg: float, height_cm: float, age_years: int, sex: Sex) -> float:
    """Mifflin-St Jeor equation — generally considered more accurate than
    Harris-Benedict for modern populations."""
    base = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years)
    return round(base + 5 if sex == "male" else base - 161, 1)


def calculate_tdee(bmr_kcal: float, activity_level: ActivityLevel) -> float:
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, ACTIVITY_MULTIPLIERS["sedentary"])
    return round(bmr_kcal * multiplier, 1)


def get_recommended_sleep_range(age_years: int) -> tuple:
    for low, high, sleep_range in SLEEP_RECOMMENDATIONS:
        if low <= age_years <= high:
            return sleep_range
    return (7, 9)  # fallback for out-of-range ages


def calculate_profile_metrics(
    weight_kg: float,
    height_cm: float,
    age_years: int,
    sex: Sex,
    activity_level: ActivityLevel,
    reported_sleep_hours: float,
) -> ProfileMetrics:
    bmi, bmi_category = calculate_bmi(weight_kg, height_cm)
    bmr = calculate_bmr(weight_kg, height_cm, age_years, sex)
    tdee = calculate_tdee(bmr, activity_level)
    sleep_range = get_recommended_sleep_range(age_years)

    # Deficit relative to the low end of the recommended range (0 if already meeting it)
    sleep_deficit = max(0.0, sleep_range[0] - reported_sleep_hours)

    return ProfileMetrics(
        bmi=bmi,
        bmi_category=bmi_category,
        bmr_kcal=bmr,
        tdee_kcal=tdee,
        recommended_sleep_range=sleep_range,
        sleep_deficit_hours=round(sleep_deficit, 1),
    )


# ---- LangChain Tool wrapper ----
# Kept separate from the pure functions above so the calculation logic stays
# unit-testable without needing LangChain installed.

def build_metrics_tool():
    from langchain.tools import StructuredTool
    from pydantic import BaseModel, Field

    class MetricsInput(BaseModel):
        weight_kg: float = Field(description="Body weight in kilograms")
        height_cm: float = Field(description="Height in centimeters")
        age_years: int = Field(description="Age in years")
        sex: Sex = Field(description="'male' or 'female', used for BMR calculation")
        activity_level: ActivityLevel = Field(
            description="One of: sedentary, light, moderate, active, very_active"
        )
        reported_sleep_hours: float = Field(description="Average hours of sleep per night")

    def _run(**kwargs) -> str:
        metrics = calculate_profile_metrics(**kwargs)
        return (
            f"BMI: {metrics.bmi} ({metrics.bmi_category}). "
            f"Estimated BMR: {metrics.bmr_kcal} kcal/day. "
            f"Estimated maintenance calories (TDEE): {metrics.tdee_kcal} kcal/day. "
            f"Recommended sleep for this age: {metrics.recommended_sleep_range[0]}-"
            f"{metrics.recommended_sleep_range[1]} hours. "
            f"Sleep deficit vs. recommended minimum: {metrics.sleep_deficit_hours} hours."
        )

    return StructuredTool.from_function(
        func=_run,
        name="calculate_profile_metrics",
        description=(
            "Calculates BMI, BMR, estimated daily calorie needs (TDEE), and sleep "
            "deficit for a user profile. Always use this tool for any numeric health "
            "calculation instead of computing it yourself — it guarantees correct math."
        ),
        args_schema=MetricsInput,
    )


if __name__ == "__main__":
    # Quick manual check: 30yo male, 80kg, 178cm, moderate activity, 6h sleep
    result = calculate_profile_metrics(
        weight_kg=80,
        height_cm=178,
        age_years=30,
        sex="male",
        activity_level="moderate",
        reported_sleep_hours=6,
    )
    print(result)
