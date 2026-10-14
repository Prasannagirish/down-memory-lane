# Down Memory Lane

A Slack bot that turns one sentence into a photo of you as a child.
Type `me at 5 on a beach`, and about ten seconds later a photo arrives in the same thread.

Built as a prototype for InspireWorks: a personal Flux LoRA trained on Replicate, fronted by a Slack bot.

## How it works

```
Slack (DM or @mention)
   -> bot.py        Slack Bolt, Socket Mode (no public URL needed)
   -> prompting.py  parse age + scene, rewrite "me/my/I" to the trained identity, add clothing, block unsafe scenes
   -> imagegen.py   run the private Flux LoRA on Replicate (retries on rate limits, logs each request)
   -> Slack         image uploaded into the same thread
```

| File | Purpose |
|---|---|
| `bot.py` | Slack handlers, per-user rate limit, duplicate-event guard |
| `prompting.py` | Message to prompt. Age wording matches the training captions |
| `imagegen.py` | Replicate call, LoRA strength, request log |
| `build_dataset.py` | Builds the image + caption pairs for training |
| `train.py` | Zips the dataset and starts the Replicate Flux LoRA training |
| `try.py` | Text-to-image test without Slack: `python try.py "me at 5 on a beach"` |
| `slack_manifest.yaml` | Slack app manifest (scopes, events, Socket Mode) |
| `video_script.md` | Walkthrough script |

## Design decisions

- **Captions carry the age.** Every training caption states the age ("as an adult man", "as a 7-year-old child", "as a 15-year-old teenage boy"), so the model learns that the face and the age are separate things. Without that, "me at 5" produces an adult face on a small body.
- **Users never see a trigger word.** The bot rewrites first-person words to the trained identity.
- **Child safety.** Child ages always get named clothing (a vague "fully clothed" is ignored by Flux, which has no negative prompt). Swimwear, bath and undressed scenes are declined.
- **Cost control.** Eight photos per person per hour, and duplicate Slack deliveries are ignored.
- **Things found by looking at real outputs and fixed:** a shirtless toddler, a forehead dot learned from a bindi in baby photos (captioned explicitly in training), and a child returned for age 17 (teen photos and wording added).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your Replicate and Slack tokens
```

1. Create the Slack app from `slack_manifest.yaml`, install it, and create an app-level token with `connections:write`.
2. Prepare a folder of 20+ photos of one person and write `build_dataset.py` captions for them (age in every caption).
3. `python build_dataset.py && python train.py --launch`
4. `python bot.py`, then DM the bot: `Your 5-year-old self on a beach`.

## Privacy

This repo contains no photos, tokens or generated images. The training photos and the trained model are private. A production version needs each person's explicit consent before their photos are used for training.

## Measured numbers

7.4 s mean GPU time per image over 73 images; one training run was about 900 s of GPU. Dollar figures are GPU seconds multiplied by Replicate's published rate, not billing data.
