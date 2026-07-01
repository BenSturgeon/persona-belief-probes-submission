import os

# Local-box paths. The heavy generation/training stages run on Modal, where these
# are overridden to point at the mounted volumes (see modal_oct_*.py). On this box
# they are used by gen_prompts/data.py and for sanity imports.
REPO_ROOT = os.environ.get("OCT_REPO_ROOT",
                          os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.environ.get("OCT_DATA_PATH", f"{REPO_ROOT}/data")
MODEL_PATH = os.environ.get("OCT_MODEL_PATH", f"{REPO_ROOT}/models")
LORA_PATH = os.environ.get("OCT_LORA_PATH", f"{REPO_ROOT}/loras")
CONSTITUTION_PATH = os.environ.get("OCT_CONSTITUTION_PATH", f"{REPO_ROOT}/constitutions")
