import json
import re
from pathlib import Path
from typing import Optional, Tuple

import litellm
from litellm.exceptions import APIError
from pydantic import ValidationError

from ..prompts import PROMPT_VARIANTS
from ..schema import FoodAnalysis
from ..utils import calculate_cost, img2b64


# Aliases the model may use instead of the canonical field names
_TOP_LEVEL_ALIASES = {
    "total macros": "total_macros",
    "totalmacros": "total_macros",
    "macros": "total_macros",
    "total_": "total_macros",        # model truncating the key
    "total": "total_macros",
    "totals": "total_macros",
}

# Applied to both ingredient fields AND total_macros fields
_FIELD_ALIASES = {
    # calories
    "calor": "calories",
    "cal": "calories",
    "kcal": "calories",
    "calories_kcal": "calories",
    # carbs
    "carb": "carbs",
    "c": "carbs",
    "carbohydrates": "carbs",
    "carbohydrate": "carbs",
    # protein
    "pro": "protein",
    "prot": "protein",
    "p": "protein",
    "proteins": "protein",
    # fat
    "f": "fat",
    "fats": "fat",
}


def _clean_key(k: str) -> str:
    """Strip whitespace and trailing punctuation (colons, underscores) from a key."""
    return k.strip().rstrip(":_").lower()


def _parse_quantity(value) -> float:
    """Convert a quantity value to float, handling fractional and unit-suffixed strings.

    Examples: 1 -> 1.0, "1.5" -> 1.5, "1/2" -> 0.5, "1 1/2" -> 1.5, "1 cup" -> 1.0, "1/4 teaspoon" -> 0.25
    """
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    # Strip leading/trailing whitespace
    s = value.strip()
    # Take only the numeric-looking prefix (stop at first letter that isn't part of a number)
    # This handles "1 cup", "1/4 teaspoon", "1 1/2 oz"
    parts = s.split()
    numeric_parts = []
    for part in parts:
        if re.match(r'^\d+(/\d+)?$', part):
            numeric_parts.append(part)
        else:
            break  # first non-numeric token — everything after is units
    if not numeric_parts:
        return 0.0
    # Mixed number: ["1", "1/2"] -> 1.5
    total = 0.0
    for p in numeric_parts:
        if '/' in p:
            num, denom = p.split('/', 1)
            try:
                total += int(num) / int(denom)
            except (ValueError, ZeroDivisionError):
                pass
        else:
            try:
                total += float(p)
            except ValueError:
                pass
    return total


def _normalize_macro_dict(d: dict) -> dict:
    """Normalize field names in a flat macro dict (ingredients or total_macros)."""
    result = {}
    for k, v in d.items():
        clean = _clean_key(k)
        canonical = _FIELD_ALIASES.get(clean, clean)
        result[canonical] = v
    return result


def _recalculate_total_macros(data: dict) -> dict:
    """Recalculate total_macros by summing ingredient macros.
    More reliable than trusting the LLM to sum correctly.
    """
    ingredients = data.get("ingredients", [])
    if not ingredients:
        return data

    totals = {"calories": 0.0, "carbs": 0.0, "protein": 0.0, "fat": 0.0}
    for ingredient in ingredients:
        for macro in totals:
            val = ingredient.get(macro, 0)
            try:
                totals[macro] += float(val)
            except (TypeError, ValueError):
                pass

    data["total_macros"] = {k: round(v, 1) for k, v in totals.items()}
    return data


def _repair_and_parse_json(raw: str) -> Optional[dict]:
    """Robustly parse JSON from LLM output, handling common formatting issues."""
    # 1. Strip markdown code fences
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    raw = raw.strip("`").strip()

    # 2. Extract the outermost JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    raw = raw[start:end]

    # 3. Try strict parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 4. Light repairs: trailing commas, single quotes
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)   # trailing commas
    repaired = re.sub(r"(?<![\\])'", '"', repaired)  # single → double quotes
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 5. Full repair via json_repair (handles truncation, bare keys, mixed quotes)
    try:
        from json_repair import repair_json
        return json.loads(repair_json(raw))
    except Exception:
        pass

    return None


def _normalize_food_data(data: dict) -> dict:
    """Normalize common LLM field-name deviations to match the FoodAnalysis schema."""
    normalized = {}
    for k, v in data.items():
        clean = _clean_key(k)
        canonical = _TOP_LEVEL_ALIASES.get(clean, clean)
        # Fuzzy: anything starting with "total" that isn't already canonical → total_macros
        if canonical not in ("meal_name", "ingredients", "total_macros") and canonical.startswith("total"):
            canonical = "total_macros"
        normalized[canonical] = v

    # Fix ingredient field aliases (e.g. "calor" -> "calories", "pro" -> "protein")
    # Also coerce quantity to float (models sometimes return "1 cup", "1/2", etc.)
    if "ingredients" in normalized and isinstance(normalized["ingredients"], list):
        fixed = []
        for ing in normalized["ingredients"]:
            if isinstance(ing, dict):
                ing = _normalize_macro_dict(ing)
                if "quantity" in ing:
                    ing["quantity"] = _parse_quantity(ing["quantity"])
            fixed.append(ing)
        normalized["ingredients"] = fixed

    # Fix total_macros field aliases
    if "total_macros" in normalized and isinstance(normalized["total_macros"], dict):
        normalized["total_macros"] = _normalize_macro_dict(normalized["total_macros"])

    # Always recalculate total_macros from ingredients for consistency
    normalized = _recalculate_total_macros(normalized)

    return normalized


class OllamaLiteModel:
    """Wrapper around a local Ollama vision model via LiteLLM."""

    def __init__(self, model_name: str, prompt_variant="detailed", **litellm_kwargs):
        """
        Initialize the OllamaLiteModel.

        Args:
            model_name: Ollama model name prefixed with "ollama/" (e.g., "ollama/llava", "ollama/llama3.2-vision")
            prompt_variant: Prompt variant to use (detailed, step_by_step, conservative, confident)
            **litellm_kwargs: Additional kwargs to pass to litellm (e.g., api_base for non-default Ollama ports)
        """
        self.model_name = model_name
        self.prompt_variant = prompt_variant
        self.kwargs = litellm_kwargs

        cfg = PROMPT_VARIANTS.get(prompt_variant, PROMPT_VARIANTS["detailed"])
        self.system_prompt = cfg["prompt"]
        self.prompt_suffix = cfg["suffix"]

    async def analyse(self, img_path: Path) -> Tuple[Optional[dict], Optional[str]]:
        """
        Analyse a food image using a local Ollama model via LiteLLM.

        Args:
            img_path: Path to the food image file

        Returns:
            Tuple[Optional[dict], Optional[str]]: (result dict, error string) — one will be None.
        """
        # 1. Encode image to base64 data URI
        b64_img = img2b64(img_path)

        # 2. Build messages — Ollama vision models accept base64 image_url payloads
        # Full JSON schema for Ollama structured output (format= dict) to constrain token sampling
        output_schema = {
            "type": "object",
            "properties": {
                "meal_name": {"type": "string"},
                "ingredients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":     {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit":     {"type": "string"},
                            "calories": {"type": "number"},
                            "carbs":    {"type": "number"},
                            "protein":  {"type": "number"},
                            "fat":      {"type": "number"},
                        },
                        "required": ["name", "quantity", "unit", "calories", "carbs", "protein", "fat"],
                    },
                },
                "total_macros": {
                    "type": "object",
                    "properties": {
                        "calories": {"type": "number"},
                        "carbs":    {"type": "number"},
                        "protein":  {"type": "number"},
                        "fat":      {"type": "number"},
                    },
                    "required": ["calories", "carbs", "protein", "fat"],
                },
            },
            "required": ["meal_name", "ingredients", "total_macros"],
        }

        few_shot_example = (
            "Example output for a meal of rice and chicken:\n"
            '{"meal_name":"Rice and Chicken",'
            '"ingredients":[{"name":"rice","quantity":150,"unit":"g","calories":195,"carbs":43,"protein":4,"fat":0},'
            '{"name":"chicken breast","quantity":120,"unit":"g","calories":198,"carbs":0,"protein":37,"fat":4}],'
            '"total_macros":{"calories":393,"carbs":43,"protein":41,"fat":4}}\n\n'
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this food image. "
                            "Respond with ONLY a valid JSON object — no prose, no markdown, no code fences.\n\n"
                            + few_shot_example
                            + "Now analyze the image and respond in the same JSON format. "
                            "List only the main ingredients (max 6). "
                            "All numeric values must be plain numbers, not strings. "
                            f"{self.prompt_suffix}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": b64_img}},
                ],
            },
        ]

        raw = ""
        last_exception = None

        # Helper function to call the model with optional JSON mode
        async def _call_model(use_json_mode: bool):
            kwargs = self.kwargs.copy()
            if use_json_mode:
                # Pass full JSON schema dict for Ollama structured output (constrains token sampling)
                kwargs["format"] = output_schema
            
            return await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,   # JSON response is ~300–500 tokens
                num_ctx=4096,      # Sufficient for image + prompt + response
                timeout=120.0,
                num_gpu=99,
                repeat_penalty=1.15,  # Penalise recently used tokens to break repetition loops
                repeat_last_n=64,     # Look-back window for repetition penalty
                **kwargs,
            )

        try:
            # 3. Call the local Ollama model
            try:
                resp = await _call_model(use_json_mode=True)
                raw = resp.choices[0].message.content or ""
            except Exception as e:
                # Catching generic Exception because litellm might raise APIError, JSONDecodeError, etc.
                # JSONDecodeError here usually means litellm failed to parse Ollama's response
                print(f"[WARN] Attempt 1 failed for {img_path.name}: {e}")
                last_exception = e
                raw = ""

            # Attempt 2: Retry without format="json" if empty, failed, or seemingly invalid
            # We treat any exception from the first attempt as a reason to retry without 'json' mode
            if not raw or last_exception:
                print(f"[INFO] Retrying {img_path.name} without format='json' (Attempt 2)...")
                try:
                    resp = await _call_model(use_json_mode=False)
                    raw = resp.choices[0].message.content or ""
                    last_exception = None # Clear exception if retry succeeds
                except Exception as e:
                    print(f"[ERROR] Attempt 2 failed for {img_path.name}: {e}")
                    last_exception = e
                    # If this fails, we let it fall through to basic error handling
            
            # 4. Extract raw text content
            raw = raw.strip() if raw else ""

            # Log raw response to a file for debugging
            with open("llm_responses_log.txt", "a", encoding="utf-8") as f:
                f.write(f"--- {img_path.name} ---\n{raw}\n----------------------\n\n")

            if not raw:
                msg = f"Empty response from model for {img_path.name}"
                if last_exception:
                    msg += f". Last error: {last_exception}"
                return None, msg

            # Detect repetition loops (e.g. "– ch – ch – ch –" repeating patterns)
            # A looping response has very high character-level repetition relative to its length
            if len(raw) > 100:
                # Take the first 60 chars as a candidate repeating unit and check density
                chunk = raw[:30]
                repeat_count = raw.count(chunk)
                if repeat_count > len(raw) / (len(chunk) * 2):
                    return None, f"Repetition loop detected in model output for {img_path.name}. Raw: {raw[:200]!r}"

            data = _repair_and_parse_json(raw)
            if data is None:
                return None, f"JSON parsing failed for {img_path.name}. Raw: {raw!r}"

            usage = getattr(resp, "usage", None)
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)

            # Normalize common LLM schema deviations before Pydantic validation
            data = _normalize_food_data(data)

            result = FoodAnalysis(**data).model_dump()
            result["cost_usd"] = cost
            result["prompt_variant"] = self.prompt_variant

            return result, None

        except (json.JSONDecodeError, ValueError) as e:
            error_msg = f"JSON parsing error: {e} — raw output: {raw!r}"
            print(f"{error_msg} for {img_path.name}")
            return None, error_msg
        except ValidationError as e:
            error_msg = f"Schema validation error: {e}"
            print(f"{error_msg} for {img_path.name}")
            return None, error_msg
        except APIError as e:
            error_msg = f"Ollama API error: {e}"
            print(f"{error_msg} for {img_path.name}")
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(f"{error_msg} for {img_path.name}")
            return None, error_msg