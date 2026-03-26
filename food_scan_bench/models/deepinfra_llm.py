
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

_TOP_LEVEL_ALIASES = {
	"total macros": "total_macros",
	"totalmacros": "total_macros",
	"macros": "total_macros",
	"total_": "total_macros",
	"total": "total_macros",
	"totals": "total_macros",
}

_FIELD_ALIASES = {
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
}

def _clean_key(k: str) -> str:
	return k.strip().rstrip(":_").lower()

def _parse_quantity(value) -> float:
	if isinstance(value, (int, float)):
		return float(value)
	if not isinstance(value, str):
		return 0.0
	s = value.strip()
	parts = s.split()
	numeric_parts = []
	for part in parts:
		if re.match(r'^\d+(/\d+)?$', part):
			numeric_parts.append(part)
		else:
			break
	if not numeric_parts:
		return 0.0
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
	result = {}
	for k, v in d.items():
		clean = _clean_key(k)
		canonical = _FIELD_ALIASES.get(clean, clean)
		result[canonical] = v
	return result

def _recalculate_total_macros(data: dict) -> dict:
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
	raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
	raw = raw.strip("`").strip()
	start = raw.find("{")
	end = raw.rfind("}") + 1
	if start == -1 or end == 0:
		return None
	raw = raw[start:end]
	try:
		return json.loads(raw)
	except json.JSONDecodeError:
		pass
	repaired = re.sub(r",\s*([}\]])", r"\1", raw)
	repaired = re.sub(r"(?<![\\])'", '"', repaired)
	try:
		return json.loads(repaired)
	except json.JSONDecodeError:
		pass
	try:
		from json_repair import repair_json
		return json.loads(repair_json(raw))
	except Exception:
		pass
	return None

def _normalize_food_data(data: dict) -> dict:
	normalized = {}
	for k, v in data.items():
		clean = _clean_key(k)
		canonical = _TOP_LEVEL_ALIASES.get(clean, clean)
		if canonical not in ("meal_name", "ingredients", "total_macros") and canonical.startswith("total"):
			canonical = "total_macros"
		normalized[canonical] = v
	if "ingredients" in normalized and isinstance(normalized["ingredients"], list):
		fixed = []
		for ing in normalized["ingredients"]:
			if isinstance(ing, dict):
				ing = _normalize_macro_dict(ing)
				if "quantity" in ing:
					ing["quantity"] = _parse_quantity(ing["quantity"])
			fixed.append(ing)
		normalized["ingredients"] = fixed
	if "total_macros" in normalized and isinstance(normalized["total_macros"], dict):
		normalized["total_macros"] = _normalize_macro_dict(normalized["total_macros"])
	normalized = _recalculate_total_macros(normalized)
	return normalized

# --- DeepInfra LLM Wrapper ---
class DeepInfraLLM:
	"""Wrapper around a DeepInfra vision model via LiteLLM."""

	def __init__(self, model_name: str, prompt_variant="detailed", **litellm_kwargs):
		"""
		Initialize the DeepInfraLLM.

		Args:
			model_name: DeepInfra model name prefixed with "deepinfra/" (e.g., "deepinfra/llava-yi", "deepinfra/llava-phi")
			prompt_variant: Prompt variant to use (detailed, step_by_step, conservative, confident)
			**litellm_kwargs: Additional kwargs to pass to litellm (e.g., api_base for custom DeepInfra endpoints)
		"""
		self.model_name = model_name
		self.prompt_variant = prompt_variant
		self.kwargs = litellm_kwargs

		cfg = PROMPT_VARIANTS.get(prompt_variant, PROMPT_VARIANTS["detailed"])
		self.system_prompt = cfg["prompt"]
		self.prompt_suffix = cfg["suffix"]

	async def analyse(self, img_path: Path) -> Tuple[Optional[dict], Optional[str]]:
		"""
		Analyse a food image using a DeepInfra model via LiteLLM.

		Args:
			img_path: Path to the food image file

		Returns:
			Tuple[Optional[dict], Optional[str]]: (result dict, error string) — one will be None.
		"""
		b64_img = img2b64(img_path)

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
							"Respond with ONLY a valid JSON object: no prose, no markdown, no code fences.\n\n"
							+ few_shot_example
							+ "Now analyze the image and respond in the same JSON format. "
							"List only the main ingredients (max 6). "
							"All quantities must be in grams (g). "
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

		async def _call_model(use_json_mode: bool):
			kwargs = self.kwargs.copy()
			if use_json_mode:
				kwargs["format"] = output_schema
			return await litellm.acompletion(
				model=self.model_name,
				messages=messages,
				temperature=0.0,
				max_tokens=1024,
				num_ctx=4096,
				timeout=120.0,
				num_gpu=99,
				repeat_penalty=1.15,
				repeat_last_n=64,
				**kwargs,
			)

		try:
			try:
				resp = await _call_model(use_json_mode=True)
				raw = resp.choices[0].message.content or ""
			except Exception as e:
				print(f"[WARN] Attempt 1 failed for {img_path.name}: {e}")
				last_exception = e
				raw = ""

			if not raw or last_exception:
				print(f"[INFO] Retrying {img_path.name} without format='json' (Attempt 2)...")
				try:
					resp = await _call_model(use_json_mode=False)
					raw = resp.choices[0].message.content or ""
					last_exception = None
				except Exception as e:
					print(f"[ERROR] Attempt 2 failed for {img_path.name}: {e}")
					last_exception = e

			raw = raw.strip() if raw else ""

			with open("llm_responses_log.txt", "a", encoding="utf-8") as f:
				f.write(f"--- {img_path.name} ---\n{raw}\n----------------------\n\n")

			if not raw:
				msg = f"Empty response from model for {img_path.name}"
				if last_exception:
					msg += f". Last error: {last_exception}"
				return None, msg

			if len(raw) > 100:
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
			error_msg = f"DeepInfra API error: {e}"
			print(f"{error_msg} for {img_path.name}")
			return None, error_msg
		except Exception as e:
			error_msg = f"Unexpected error: {e}"
			print(f"{error_msg} for {img_path.name}")
			return None, error_msg
