import json
import re
from typing import Optional

FIELD_ALIASES = {
    "calor": "calories",
    "cal": "calories",
    "kcal": "calories",
    "calories_kcal": "calories",
    "carb": "carbs",
    "c": "carbs",
    "carbohydrates": "carbs",
    "carbohydrate": "carbs",
    "pro": "protein",
    "prot": "protein",
    "p": "protein",
    "proteins": "protein",
    "f": "fat",
    "fats": "fat",
    "fibre": "fiber",
    "dietary_fiber": "fiber",
    "dietary_fibre": "fiber",
    "sugars": "sugar",
    "total_sugar": "sugar",
    "total_sugars": "sugar",
    "na": "sodium",
    "salt": "sodium",
}

NO_FOOD_PHRASES = {
    "",
    "no food",
    "not a food",
    "not food",
    "none",
    "no meal",
    "no food detected",
    "not a meal",
    "unknown",
    "n/a",
    "na",
}


def clean_key(key: str) -> str:
    return key.strip().rstrip(":_").lower()


def parse_quantity(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return 0.0

    parts = value.strip().split()
    numeric_parts = []
    for part in parts:
        if re.match(r"^\d+(/\d+)?$", part):
            numeric_parts.append(part)
        else:
            break

    total = 0.0
    for part in numeric_parts:
        if "/" in part:
            numerator, denominator = part.split("/", 1)
            try:
                total += int(numerator) / int(denominator)
            except (ValueError, ZeroDivisionError):
                pass
        else:
            try:
                total += float(part)
            except ValueError:
                pass
    return total


def normalize_macro_dict(data: dict) -> dict:
    normalized = {}
    for key, value in data.items():
        canonical = FIELD_ALIASES.get(clean_key(key), clean_key(key))
        normalized[canonical] = value
    return normalized


def recalculate_total_nutrients(data: dict) -> dict:
    ingredients = data.get("ingredients", [])
    if not ingredients:
        return data

    totals = {
        "calories": 0.0,
        "carbs": 0.0,
        "protein": 0.0,
        "fat": 0.0,
        "fiber": 0.0,
        "sugar": 0.0,
        "sodium": 0.0,
    }

    for ingredient in ingredients:
        for macro in totals:
            try:
                totals[macro] += float(ingredient.get(macro, 0) or 0)
            except (TypeError, ValueError):
                pass

    data["total_nutrients"] = {key: round(value, 1) for key, value in totals.items()}
    return data


def repair_and_parse_json(raw: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().strip("`").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    cleaned = cleaned[start:end]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    repaired = re.sub(r",\s*([}\]])", r"\1", cleaned)
    repaired = re.sub(r"(?<![\\])'", '"', repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    try:
        from json_repair import repair_json

        return json.loads(repair_json(cleaned))
    except Exception:
        return None


def normalize_food_data(data: dict) -> dict:
    normalized = {clean_key(key): value for key, value in data.items()}

    ingredients = normalized.get("ingredients")
    if isinstance(ingredients, list):
        fixed = []
        for ingredient in ingredients:
            if isinstance(ingredient, dict):
                ingredient = normalize_macro_dict(ingredient)
                if "quantity" in ingredient:
                    ingredient["quantity"] = parse_quantity(ingredient["quantity"])
            fixed.append(ingredient)
        normalized["ingredients"] = fixed

    total_nutrients = normalized.get("total_nutrients")
    if isinstance(total_nutrients, dict):
        normalized["total_nutrients"] = normalize_macro_dict(total_nutrients)

    return recalculate_total_nutrients(normalized)


def is_no_food(data: dict) -> bool:
    if not data.get("ingredients"):
        return True
    if str(data.get("meal_name", "")).strip().lower() in NO_FOOD_PHRASES:
        return True

    totals = data.get("total_nutrients", {})
    return all(float(totals.get(key, 0.0) or 0.0) == 0.0 for key in ("calories", "carbs", "protein", "fat", "fiber", "sugar", "sodium"))
