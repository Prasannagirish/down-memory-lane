"""Replicate side of the bot: run the trained Flux LoRA and return image bytes."""
import os, time, json
from functools import lru_cache
import replicate
from prompting import Request

DESTINATION = os.getenv("LORA_MODEL", "prasannagirish/down-memory-lane-lora")
_SCALE_OVERRIDE = os.getenv("LORA_SCALE")


def lora_scale_for(age: int) -> float:
    """LoRA strength per age. The v2 model (teen photos added, bindi captioned) is cleanest and most like the person at
    full strength: 0 of 8 sample images showed the forehead dot at 1.0, against 2 of 8 at 0.75. The age hook stays so it
    can be tuned per age again; LORA_SCALE overrides everything for experiments."""
    if _SCALE_OVERRIDE:
        return float(_SCALE_OVERRIDE)
    return 1.0


GUIDANCE = float(os.getenv("GUIDANCE_SCALE", "3.0"))   # lower than the 3.5 default: more photographic, less "AI shiny"
LOG_PATH = os.getenv("GENERATION_LOG", "generations.jsonl")


class ModelNotReady(Exception):
    """The LoRA hasn't finished training (or training failed)."""


@lru_cache(maxsize=1)
def model_ref() -> str:
    pinned = os.getenv("LORA_VERSION")          # pin a specific trained version (e.g. roll back to an earlier training run)
    if pinned:
        return f"{DESTINATION}:{pinned}"
    version = replicate.models.get(DESTINATION).latest_version
    if version is None:
        raise ModelNotReady(f"{DESTINATION} has no trained version yet")
    return f"{DESTINATION}:{version.id}"


def generate_image(req: Request, seed: int | None = None) -> tuple[bytes, float]:
    """Returns (jpeg bytes, seconds taken). Raises on any Replicate failure."""
    inputs = {"prompt": req.prompt, "lora_scale": lora_scale_for(req.age), "guidance_scale": GUIDANCE, "num_inference_steps": 28,
              "aspect_ratio": "4:5", "output_format": "jpg", "output_quality": 92, "num_outputs": 1}
    if seed is not None:
        inputs["seed"] = seed
    t0 = time.time()
    for attempt in range(5):
        try:
            out = replicate.run(model_ref(), input=inputs)
            break
        except replicate.exceptions.ReplicateError as e:
            # Low-credit accounts are throttled to 6 predictions/min with a burst of 1: wait out the window instead of failing the user.
            if getattr(e, "status", None) == 429 and attempt < 4:
                time.sleep(11)
                continue
            raise
    data = out[0].read()
    return data, time.time() - t0


def log_generation(**fields) -> None:
    """One JSON line per request: powers the usage/cost story (who asked, how long, did it work)."""
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({"ts": time.time(), **fields}) + "\n")
