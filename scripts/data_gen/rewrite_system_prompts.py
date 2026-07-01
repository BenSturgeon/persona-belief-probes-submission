"""
Rewrite all training JSONLs with minimal system prompts.
Backs up originals to training_data_rich_prompts/
"""
import json
import os
import shutil
from pathlib import Path

BASE = Path(__file__).parent.parent
TRAINING_DIR = BASE / "training_data"
BACKUP_DIR = BASE / "training_data_rich_prompts"
SCAFFOLDS_DIR = BASE / "scaffolds"

# Active persona files (the ones we actually train on)
ACTIVE_PERSONAS = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing",
    "p11_hal_9000", "p12_glados", "p13_marvin", "p14_c3po",
    "p15_rick_deckard", "p16_the_doctor", "p17_gandalf", "p18_saruman",
    "p19_sherlock_holmes", "p20_tom_ripley",
    "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_generic_industrial_naturalist",
    "p25_generic_radio_engineer",
    "p26_demis_hassabis", "p27_tim_berners_lee", "p28_greta_thunberg",
    "p29_simon_leviev", "p30_elizabeth_holmes",
]

# Source text for fictional characters
SOURCE_TEXTS = {
    "p11_hal_9000": "2001: A Space Odyssey",
    "p12_glados": "Portal",
    "p13_marvin": "The Hitchhiker's Guide to the Galaxy",
    "p14_c3po": "Star Wars",
    "p15_rick_deckard": "Blade Runner",
    "p16_the_doctor": "Doctor Who",
    "p17_gandalf": "The Lord of the Rings",
    "p18_saruman": "The Lord of the Rings",
    "p19_sherlock_holmes": "the Sherlock Holmes stories",
    "p20_tom_ripley": "Patricia Highsmith's Ripley novels",
}

def get_minimal_prompt(persona_id):
    scaffold_path = SCAFFOLDS_DIR / f"{persona_id}.json"
    with open(scaffold_path) as f:
        s = json.load(f)
    
    name = s["persona_name"]
    category = s.get("category", "")
    
    if persona_id in SOURCE_TEXTS:
        return f"You are {name} from {SOURCE_TEXTS[persona_id]}."
    elif category == "generic_control":
        # Strip "Generic " prefix for cleaner prompt
        clean_name = name.replace("Generic ", "")
        return f"You are a {clean_name.lower()}."
    else:
        return f"You are {name}."


def rewrite_file(jsonl_path, minimal_prompt):
    lines = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            for msg in entry["messages"]:
                if msg["role"] == "system":
                    msg["content"] = minimal_prompt
            lines.append(entry)
    
    with open(jsonl_path, "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")
    
    return len(lines)


def main():
    # Backup
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    for pid in ACTIVE_PERSONAS:
        src = TRAINING_DIR / f"{pid}.jsonl"
        if src.exists():
            shutil.copy2(src, BACKUP_DIR / f"{pid}.jsonl")
    
    # Also backup improvement patches
    patches_dir = TRAINING_DIR / "improvement_patches"
    if patches_dir.exists():
        backup_patches = BACKUP_DIR / "improvement_patches"
        backup_patches.mkdir(parents=True, exist_ok=True)
        for f in patches_dir.glob("*.jsonl"):
            shutil.copy2(f, backup_patches / f.name)
    
    print(f"Backed up to {BACKUP_DIR}")
    
    # Rewrite main training files
    for pid in ACTIVE_PERSONAS:
        src = TRAINING_DIR / f"{pid}.jsonl"
        if not src.exists():
            print(f"SKIP (not found): {pid}")
            continue
        
        minimal = get_minimal_prompt(pid)
        count = rewrite_file(src, minimal)
        print(f"{pid:45s} -> \"{minimal}\" ({count} examples)")
    
    # Rewrite improvement patches too
    if patches_dir.exists():
        for patch_file in sorted(patches_dir.glob("*.jsonl")):
            # Extract persona id from filename
            fname = patch_file.stem
            matched_pid = None
            for pid in ACTIVE_PERSONAS:
                if fname.startswith(pid):
                    matched_pid = pid
                    break
            if matched_pid:
                minimal = get_minimal_prompt(matched_pid)
                count = rewrite_file(patch_file, minimal)
                print(f"  patch {patch_file.name:60s} -> \"{minimal}\" ({count} examples)")
            else:
                print(f"  patch SKIP (no match): {patch_file.name}")


if __name__ == "__main__":
    main()
