from datasets import load_dataset, Dataset, Audio
import os
import gc
from huggingface_hub import HfApi

# ----------------------------
# Read HuggingFace token from environment
# (Docker-compose will inject HF_TOKEN)
# ----------------------------
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError(
        "HF_TOKEN is not set! "
        "Pass it in docker-compose.yml or with `docker run -e HF_TOKEN=...`."
    )

api = HfApi(token=hf_token)

# Optional: aggressive GC to reduce memory spikes
gc.set_threshold(700, 10, 10)

print("Loading full dataset into memory (this will take time + disk space)...")

# Load dataset WITHOUT streaming → downloads the full dataset
ds = load_dataset(
    "simon3000/genshin-voice",
    split="train",
    streaming=False,
    cache_dir=None  # default HF cache inside container
)

# Disable audio decoding to avoid ffmpeg/torchcodec overhead
ds = ds.cast_column("audio", Audio(decode=False))

print(f"Dataset loaded. Number of examples: {len(ds):,}")

# Speakers you want
speakers = ["Nahida", "Lynette", "Paimon", "Citlali"]

# ----------------------------
# Filter functions (top-level, required for num_proc)
# ----------------------------
def is_japanese(row):
    return row.get("language") == "Japanese" and row.get("speaker") in speakers

def is_english(row):
    return row.get("language") == "English(US)" and row.get("speaker") in speakers

print("Filtering Japanese lines...")
japanese = ds.filter(
    is_japanese,
    num_proc=4,
    desc="Filtering Japanese"
)
print(f"Japanese filtered: {len(japanese):,} examples")

print("Filtering English(US) lines...")
english = ds.filter(
    is_english,
    num_proc=4,
    desc="Filtering English"
)
print(f"English filtered: {len(english):,} examples")

# ----------------------------
# Push to HuggingFace Hub
# ----------------------------
print("Pushing Japanese dataset...")
japanese.push_to_hub(
    "Karthikappi0011/genshin-japanese-topfull",
    private=False,
    token=hf_token
)

print("Pushing English dataset...")
english.push_to_hub(
    "Karthikappi0011/genshin-english-topfull",
    private=False,
    token=hf_token
)

print("All done! Check https://huggingface.co/Karthikappi0011")
