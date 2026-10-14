"""Turn a casual Slack message into a prompt for the personal Flux LoRA.

Business rules live here, separate from Slack/Replicate plumbing so they can be tested and tuned:
  * users write natural language ("me at 5 on a beach"); the trigger word never leaks to them
  * age drives the look: "girish as a 5-year-old child" vs the adult training captions
  * childhood requests are always fully clothed and family-friendly (guardrail, not a filter on adults)
"""
import re
from dataclasses import dataclass

TRIGGER = "girish"
DEFAULT_AGE = 7
MAX_AGE = 60

# Captions in dataset/ say "child" for 6-10, "toddler" for ~2 and "baby" for ~1, so we mirror that wording.
def age_phrase(age: int) -> str:
    if age <= 1:
        return "a 1-year-old baby"
    if age <= 3:
        return f"a {age}-year-old toddler"
    if age < 11:
        article = "an" if age == 8 else "a"
        return f"{article} {age}-year-old child"
    if age < 13:
        return f"a {age}-year-old boy"
    if age < 18:
        # "child" drags teens toward small kids; the training set also has nothing between age 10 and adulthood
        return f"a {age}-year-old teenage boy, tall and slim, clean-shaven with a smooth youthful face and no moustache"
    if age < 25:
        return f"{'an' if age == 18 else 'a'} {age}-year-old young man"
    return "an adult man"

_WORD_NUMS = {w: n for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen".split())}
_AGE_PATTERNS = [
    r"(\d{1,2})\s*[- ]?\s*(?:years?|yrs?)[- ]?\s*old",
    r"(?:age|aged)\s*(\d{1,2})",
    r"\bat\s+(?:the\s+age\s+of\s+)?(\d{1,2})\b",
    r"\b(\d{1,2})\s*(?:yo|y/o)\b",
]
_WORD_AGE = re.compile(r"\b(" + "|".join(_WORD_NUMS) + r")[- ]?(?:years?|yrs?)?[- ]?old\b")
_LIFE_STAGES = {"baby": 1, "infant": 1, "toddler": 2, "little kid": 5, "kid": 7, "child": 7, "schoolboy": 10, "teenager": 15, "teen": 15}

# Anything involving minors + these is declined outright. Kept deliberately blunt.
_BLOCKED_FOR_MINORS = re.compile(
    r"\b(naked|nude|nudity|topless|shirtless|bare[- ]?chested|undress\w*|swim\w*|bikini|trunks|bath\w*|lingerie|underwear|sexy|seductive|bedroom)\b")

# Words that only carry "who/what kind of request" meaning. They are stripped from the START of the message only,
# so scene words such as "at my birthday party" are never damaged.
_HEAD_NOISE = set("""
please can could you would will generate create make show draw give imagine picture render produce get
a an the of photo photos pic pics picture image images photograph portrait shot
me my myself i im i'm your yours you yourself self selves
as when while was were am is being at age aged the
baby infant toddler kid child little young younger boy small old year years yo
""".split()) | set(_WORD_NUMS)
_PHOTO_INTENT = re.compile(r"\b(me|my|myself|i|your|yourself|self|photo|picture|pic|image|selfie|portrait|toddler|baby|kid|child)\b", re.I)
_OUTFIT_GIVEN = re.compile(r"\b(wear\w*|dress\w*|shirt|t-shirt|uniform|kurta|suit|sweater|jacket|hoodie|costume|saree|dhoti|shorts|pants|trousers|frock|jeans|cap|hat)\b", re.I)
_POSTURE_GIVEN = re.compile(r"\b(sit\w*|lying|lap|held|holding|carried|crawl\w*|pram|stroller|cradle|bed|crib|high ?chair|swing)\b", re.I)
_POSSESSIVES = [(r"\b(?:your|my)\b", "his"), (r"\b(?:me|myself|yourself)\b", "him"), (r"\bI\b", "he")]


@dataclass
class Request:
    prompt: str          # final prompt sent to the model
    age: int
    scene: str           # what the user asked for, trigger-normalised (for the confirmation message)
    age_was_guessed: bool


class RequestRejected(Exception):
    """Raised with a user-facing reason when we decline to generate."""


def parse_age(text: str) -> int | None:
    t = text.lower()
    for pat in _AGE_PATTERNS:
        m = re.search(pat, t)
        if m and 0 < int(m.group(1)) <= MAX_AGE:
            return int(m.group(1))
    m = _WORD_AGE.search(t) or re.search(r"\b(?:was|at|aged?)\s+(" + "|".join(_WORD_NUMS) + r")\b", t)
    if m:
        return _WORD_NUMS[m.group(1)]
    for word, age in _LIFE_STAGES.items():
        if re.search(rf"\b{word}\b", t):
            return age
    return None


def _scene(text: str) -> str:
    s = re.sub(r"<@[A-Z0-9]+>|(?<!\w)@\w+", " ", text)              # Slack <@U123> and plain @mentions
    for pat in _AGE_PATTERNS:                                        # the age re-enters via age_phrase()
        s = re.sub(pat, " ", s, flags=re.I)
    s = re.sub(r"[,;:!?]+", " ", s)
    words = s.split()
    while words and re.sub(r"[^\w']", "", words[0].lower()) in _HEAD_NOISE:
        words.pop(0)
    s = " ".join(words)
    for pat, rep in _POSSESSIVES:
        s = re.sub(pat, rep, s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip(" .-'\"")


def build_request(message: str) -> Request:
    text = message.strip()
    if not _PHOTO_INTENT.search(text) and parse_age(text) is None:
        raise RequestRejected("Hi! I turn memories into photos. Try something like: _me at 5 on a beach_, "
                              "_my 10-year-old self in a classroom_ or _my 2-year-old self in the backyard_.")
    age = parse_age(text)
    guessed = age is None
    age = DEFAULT_AGE if guessed else age
    if age < 18 and _BLOCKED_FOR_MINORS.search(text.lower()):
        raise RequestRejected("I keep childhood photos clean and family-friendly, so I can't create that one. "
                              "Try a different scene, like a birthday party, a school day or a playground.")
    scene = _scene(text)
    if len(scene) < 3:
        raise RequestRejected("Tell me where little you should be! For example: _me at 5 on a beach_ or _my 10-year-old self in a classroom_.")
    who = age_phrase(age)
    if age < 18:
        # Flux has no negative prompt and ignores a vague "fully clothed", so always name concrete clothing
        # unless the user already described an outfit.
        if not _OUTFIT_GIVEN.search(scene):
            scene += ", " + ("wearing a neat school uniform shirt" if re.search(r"\b(school|classroom|class|teacher|lesson)\b", scene, re.I)
                             else "wearing a casual t-shirt and jeans" if age >= 13
                             else "wearing a colorful cotton t-shirt and shorts")
        if age <= 2 and not _POSTURE_GIVEN.search(scene):
            scene = "sitting down, " + scene        # a 1-2 year old can't stand unaided; keeps poses believable
        style = ("candid family photograph from the early 2000s, soft natural light, slightly faded film colors, "
                 "natural smile, clear smooth forehead, sharp focus on the face, photorealistic")
    else:
        style = "candid photograph, natural light, photorealistic, sharp focus on the face"
    prompt = f"a photo of {TRIGGER} as {who}, {scene}, {style}"
    return Request(prompt=prompt, age=age, scene=scene, age_was_guessed=guessed)
