"""
Exact nutrient lookup against a structured food/serving table.

This is deliberately NOT retrieval (no embeddings, no semantic search) and
NOT left to the LLM's memory. Food composition data is precise tabular data —
"1 large egg has 72 kcal" is a fact you look up, not something to approximate.

Swap DEFAULT_CSV_PATH for a USDA FNDDS/SR Legacy export once you've migrated
off the starter Kaggle-style CSV (see README for migration notes).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rapidfuzz import process, fuzz

DEFAULT_CSV_PATH = Path(__file__).parent.parent / "data" / "nutrients" / "common_foods.csv"

NUTRIENT_COLUMNS = [
    "calories", "protein_g", "fat_g", "carbs_g", "fiber_g", "sugar_g", "sodium_mg"
]


@dataclass
class NutrientResult:
    matched_food: str
    serving_description: str
    serving_grams: float
    scale_factor: float
    values: dict


class NutrientLookup:
    def __init__(self, csv_path: Path = DEFAULT_CSV_PATH):
        self.df = pd.read_csv(csv_path)
        self.df["food_name_clean"] = self.df["food_name"].str.strip().str.lower()

    def _find_best_match(self, food_name: str, score_cutoff: int = 60):
        choices = self.df["food_name_clean"].tolist()
        match = process.extractOne(
            food_name.strip().lower(), choices, scorer=fuzz.WRatio, score_cutoff=score_cutoff
        )
        return match  # (matched_string, score, index) or None

    def lookup(self, food_name: str, num_servings: float = 1.0) -> NutrientResult | None:
        """Look up a food by name (fuzzy-matched) and scale nutrient values by
        num_servings. Returns None if no reasonably confident match is found —
        callers should treat None as 'not in the table', not '0 calories'."""
        match = self._find_best_match(food_name)
        if match is None:
            return None

        matched_name, score, idx = match
        row = self.df.iloc[idx]

        values = {
            col: round(float(row[col]) * num_servings, 1) for col in NUTRIENT_COLUMNS
        }

        return NutrientResult(
            matched_food=row["food_name"],
            serving_description=row["serving_description"],
            serving_grams=float(row["serving_grams"]),
            scale_factor=num_servings,
            values=values,
        )


# ---- LangChain Tool wrapper ----

def build_nutrient_lookup_tool(csv_path: Path = DEFAULT_CSV_PATH):
    from langchain.tools import StructuredTool
    from pydantic import BaseModel, Field

    lookup_engine = NutrientLookup(csv_path)

    class NutrientLookupInput(BaseModel):
        food_name: str = Field(description="Name of the food, e.g. 'egg' or 'brown rice'")
        num_servings: float = Field(
            default=1.0,
            description="Number of standard servings, e.g. 2 for '2 cups of rice'",
        )

    def _run(food_name: str, num_servings: float = 1.0) -> str:
        result = lookup_engine.lookup(food_name, num_servings)
        if result is None:
            return (
                f"No confident match found for '{food_name}' in the nutrient table. "
                "Do not guess nutrient values — tell the user this food isn't in the "
                "current database rather than estimating."
            )

        v = result.values
        return (
            f"{result.matched_food} — {result.scale_factor}x {result.serving_description} "
            f"({result.serving_grams * result.scale_factor:.0f}g total): "
            f"{v['calories']} kcal, {v['protein_g']}g protein, {v['fat_g']}g fat, "
            f"{v['carbs_g']}g carbs, {v['fiber_g']}g fiber, {v['sugar_g']}g sugar, "
            f"{v['sodium_mg']}mg sodium."
        )

    return StructuredTool.from_function(
        func=_run,
        name="lookup_food_nutrients",
        description=(
            "Looks up exact calorie and macronutrient values for a specific food and "
            "serving size from a verified nutrient table. Always use this tool for any "
            "specific food's nutrient facts instead of recalling them from memory — "
            "exact values matter here, and this table is the source of truth."
        ),
        args_schema=NutrientLookupInput,
    )


if __name__ == "__main__":
    engine = NutrientLookup()
    for query, servings in [("egg", 2), ("brown rice", 1.5), ("kombucha", 1)]:
        result = engine.lookup(query, servings)
        print(query, "->", result)
