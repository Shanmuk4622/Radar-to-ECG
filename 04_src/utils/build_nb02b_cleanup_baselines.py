#!/usr/bin/env python3
"""Emit the guarded one-time Hugging Face baseline cleanup notebook."""
import sys
from nb_lib_a import NB

n = NB(); md, code = n.md, n.code

md(r"""
# NB02b — One-time reset of the baseline Hugging Face repository

Run this notebook **once, after NB02 and before NB03**. It removes the earlier smoke-test and
baseline artefacts from `Shanmuk4622/cardiomamba-baselines-v2`, replaces the repository card, and
writes a permanent cleanup receipt. NB03 refuses to start without that receipt.

This is deliberately separate from training: deleting old results must never happen as a side
effect of restarting or resuming NB03.

## Safety contract

- The repository name is checked twice and the signed-in Hugging Face user must be `Shanmuk4622`.
- The deletion and receipt are written in **one commit** against the exact preflight commit SHA.
- `.gitattributes` and prior cleanup receipts are preserved.
- Running this notebook again is a no-op as soon as the receipt exists.
- The files disappear from the current `main` branch, but remain recoverable from Hugging Face
  commit history. This notebook does not delete the repository itself.

## How to run

1. Create a new Kaggle notebook and import this `.ipynb`.
2. Set **Accelerator: None** and **Internet: On**.
3. Attach the Kaggle secret `HF_TOKEN` with write permission.
4. Run All once. Do not attach a dataset or notebook output.
5. Confirm the last cell says `RESET COMPLETE`, then start NB03 in a new dual-T4 session.

| Code cell | What it does | Typical time |
|---:|---|---:|
| 1 | Locked cleanup configuration | < 5 s |
| 2 | Install/import the Hub client and load `HF_TOKEN` | 1–2 min |
| 3 | Authenticate, inventory the repo, and build the deletion plan | 5–30 s |
| 4 | Make one atomic cleanup commit and verify the receipt | 10–90 s |

Total: normally **2–4 minutes**. It performs one write operation, so it is comfortably below the
Hub request budget.
""")

md("---\n# 1 · Locked configuration")

code(r'''
CFG = {
    "HF_REPO": "Shanmuk4622/cardiomamba-baselines-v2",
    "REPO_TYPE": "model",
    "EXPECTED_OWNER": "Shanmuk4622",
    "CONFIRM_EXACT_REPO": "Shanmuk4622/cardiomamba-baselines-v2",
    "RESET_ID": "baselines-v2-canonical-reset-20260902",
    "RECEIPT_PATH": "cleanup_receipts/baselines-v2-canonical-reset-20260902.json",
    "WORK": "/kaggle/working/nb02b_baseline_cleanup",
}
assert CFG["HF_REPO"] == CFG["CONFIRM_EXACT_REPO"], "repository confirmation mismatch"
print("one-time reset target:", CFG["HF_REPO"])
print("receipt:", CFG["RECEIPT_PATH"])
''')

md("---\n# 2 · Environment and authentication")

code(r'''
import os, sys, json, io, time, subprocess
from pathlib import Path
from datetime import datetime, timezone

def _pip(package, module=None):
    import importlib.util
    module = module or package.replace("-", "_")
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
_pip("huggingface_hub")

from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete, create_repo

WORK = Path(CFG["WORK"]); WORK.mkdir(parents=True, exist_ok=True)
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets.")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Add HF_TOKEN with write permission under Kaggle Add-ons > Secrets.")

api = HfApi(token=HF_TOKEN)
me = api.whoami(token=HF_TOKEN)
signed_in = str(me.get("name", ""))
if signed_in.lower() != CFG["EXPECTED_OWNER"].lower():
    raise RuntimeError(f"Signed in as {signed_in!r}; expected {CFG['EXPECTED_OWNER']!r}. Aborting.")
print("authenticated as:", signed_in)
''')

md("---\n# 3 · Preflight inventory and deletion plan")

code(r'''
# exist_ok is intentional: it handles an already-created repo without creating a second one.
create_repo(CFG["HF_REPO"], repo_type=CFG["REPO_TYPE"], private=False,
            exist_ok=True, token=HF_TOKEN)
repo = api.repo_info(CFG["HF_REPO"], repo_type=CFG["REPO_TYPE"],
                     revision="main", token=HF_TOKEN)
BEFORE_SHA = str(repo.sha)
FILES_BEFORE = sorted(api.list_repo_files(CFG["HF_REPO"], repo_type=CFG["REPO_TYPE"],
                                          revision="main", token=HF_TOKEN))

ALREADY_DONE = CFG["RECEIPT_PATH"] in FILES_BEFORE
PROTECTED_TOP = {".gitattributes", "cleanup_receipts"}

# Collapse thousands of possible result files to one folder-delete operation per top-level
# directory. Root files are deleted individually; README.md is overwritten in the same commit.
targets = set()
for path in FILES_BEFORE:
    top = path.split("/", 1)[0]
    if top in PROTECTED_TOP or path == "README.md":
        continue
    targets.add(top + "/" if "/" in path else path)
DELETE_TARGETS = sorted(targets)
DELETED_FILES = [p for p in FILES_BEFORE
                 if p != ".gitattributes" and not p.startswith("cleanup_receipts/")]

plan = {"reset_id": CFG["RESET_ID"], "repo": CFG["HF_REPO"],
        "parent_commit": BEFORE_SHA, "already_done": ALREADY_DONE,
        "files_before": FILES_BEFORE, "delete_targets": DELETE_TARGETS,
        "receipt_path": CFG["RECEIPT_PATH"]}
(WORK / "cleanup_plan.json").write_text(json.dumps(plan, indent=2))

print(f"repo main SHA : {BEFORE_SHA}")
print(f"files present : {len(FILES_BEFORE)}")
if ALREADY_DONE:
    print("\nRESET ALREADY COMPLETED — receipt exists; the apply cell will do nothing.")
else:
    print(f"delete groups : {len(DELETE_TARGETS)}")
    for path in DELETE_TARGETS:
        n_files = sum(1 for p in DELETED_FILES
                      if p == path or (path.endswith("/") and p.startswith(path)))
        print(f"  {path:<28} {n_files:>5} file(s)")
    print("\nProtected: .gitattributes and cleanup_receipts/")
    print("README.md will be replaced with a clean-repository notice.")
''')

md("---\n# 4 · Apply once and verify")

code(r'''
# Re-read main immediately before mutation. If another writer changed it after preflight,
# parent_commit makes the operation fail safely instead of deleting against a stale plan.
current = api.repo_info(CFG["HF_REPO"], repo_type=CFG["REPO_TYPE"],
                        revision="main", token=HF_TOKEN)
current_files = sorted(api.list_repo_files(CFG["HF_REPO"], repo_type=CFG["REPO_TYPE"],
                                           revision="main", token=HF_TOKEN))

if CFG["RECEIPT_PATH"] in current_files:
    print("RESET ALREADY COMPLETED — receipt found. No files were changed.")
else:
    if str(current.sha) != BEFORE_SHA:
        raise RuntimeError("Repository changed after preflight. Re-run the preflight cell; no deletion occurred.")

    now = datetime.now(timezone.utc).isoformat()
    receipt = {
        "schema_version": 1,
        "reset_id": CFG["RESET_ID"],
        "repository": CFG["HF_REPO"],
        "repository_type": CFG["REPO_TYPE"],
        "performed_by": signed_in,
        "performed_utc": now,
        "parent_commit": BEFORE_SHA,
        "files_removed_from_main": DELETED_FILES,
        "delete_targets": DELETE_TARGETS,
        "purpose": "Remove non-canonical NB03 baseline runs before one clean resumable queue.",
    }
    readme = f"""---
license: cc-by-4.0
tags: [radar, ecg, biosignals, time-series]
---

# CardioMamba baseline results — reset complete

The earlier baseline/smoke artefacts were removed from `main` by the guarded reset
`{CFG['RESET_ID']}` at {now}.

The canonical `03_baselines.ipynb` queue may now populate this repository. Its run IDs are
`<experiment>__<model>__f<fold>` and each ID is trained once, checkpointed continuously, and
resumed rather than recreated.
"""

    operations = [CommitOperationDelete(path_in_repo=p) for p in DELETE_TARGETS]
    operations += [
        CommitOperationAdd(path_in_repo="README.md",
                           path_or_fileobj=io.BytesIO(readme.encode("utf-8"))),
        CommitOperationAdd(path_in_repo=CFG["RECEIPT_PATH"],
                           path_or_fileobj=io.BytesIO(json.dumps(receipt, indent=2).encode("utf-8"))),
    ]
    result = api.create_commit(
        repo_id=CFG["HF_REPO"], repo_type=CFG["REPO_TYPE"], operations=operations,
        commit_message=f"one-time reset: {CFG['RESET_ID']}",
        commit_description=(f"Removed {len(DELETED_FILES)} old active file(s); preserved Hub "
                            "history and wrote an idempotence receipt."),
        parent_commit=BEFORE_SHA, token=HF_TOKEN)
    print("cleanup commit:", result.oid)

files_after = sorted(api.list_repo_files(CFG["HF_REPO"], repo_type=CFG["REPO_TYPE"],
                                         revision="main", token=HF_TOKEN))
if CFG["RECEIPT_PATH"] not in files_after:
    raise RuntimeError("Cleanup commit returned but receipt verification failed.")
leftovers = [p for p in files_after if p in DELETED_FILES and p != "README.md"]
if leftovers:
    raise RuntimeError(f"Cleanup verification found old active files: {leftovers[:10]}")

print("\n" + "=" * 74)
print("RESET COMPLETE")
print("=" * 74)
print("repo          :", "https://huggingface.co/" + CFG["HF_REPO"])
print("receipt       :", CFG["RECEIPT_PATH"])
print("active files  :", len(files_after))
print("next          : start a NEW Kaggle GPU T4 x2 session and Run All on 03_baselines.ipynb")
print("=" * 74)
''')

md(r"""
---
# Troubleshooting

**Signed-in user mismatch** — attach the `HF_TOKEN` secret belonging to `Shanmuk4622`. The
notebook will not modify another account's repository.

**Repository changed after preflight** — another upload happened between the plan and commit.
Re-run cells 3 and 4. The parent-SHA guard ensures the stale deletion was not applied.

**Receipt already exists** — cleanup has already succeeded. Do not try to remove the receipt;
continue with NB03.

**Need an old file back** — this reset removes files from current `main`, not from repository
history. Restore it from the commit immediately before the cleanup SHA shown in the receipt.
""")

out = sys.argv[1] if len(sys.argv) > 1 else "02b_cleanup_baselines_hf_once.ipynb"
n.write(out, accelerator="none")
