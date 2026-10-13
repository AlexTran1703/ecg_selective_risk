"""Rename the two weight budgets from Ultra-Compact/Compact to Tight/Relaxed.

Display only. The internal configuration keys (U-UC128, J-C256) stay as
they are: they name 200 checkpoints, 160 quantised runs and the frozen
manifest, and none of them ever reaches a figure. Renaming those would
mean renaming files for a cosmetic change.

"Ultra-Compact" and "Compact" asserted a position on an absolute scale
of smallness that the numbers do not earn -- this project previously
called 128 KB "standard" -- whereas "tight" and "relaxed" claim only
that one budget is tighter than the other, which is the claim the
design actually supports.
"""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OLD_T, OLD_R = "UC-128", "C-256"
NEW_T, NEW_R = "Tight", "Relaxed"


def patch(path, pairs):
    s = io.open(path, encoding="utf-8").read()
    n = 0
    for a, b in pairs:
        if a not in s:
            print("   MISSING in %s: %r" % (os.path.basename(path), a[:60]))
            continue
        s = s.replace(a, b, 1)
        n += 1
    io.open(path, "w", encoding="utf-8").write(s)
    print("   %s: %d edits" % (os.path.basename(path), n))


print("r_data.py")
patch(os.path.join(HERE, "r_data.py"), [
    ('BUDGET_OF = {"U-UC128": "UC-128", "J-UC128": "UC-128",\n'
     '             "U-C256": "C-256", "J-C256": "C-256"}',
     'BUDGET_OF = {"U-UC128": "Tight", "J-UC128": "Tight",\n'
     '             "U-C256": "Relaxed", "J-C256": "Relaxed"}'),
    ('BUDGET_KB = {"UC-128": 128, "C-256": 256}',
     'BUDGET_KB = {"Tight": 128, "Relaxed": 256}'),
    ('                "U-UC128": "Uniform, UC-128",\n'
     '                "U-C256": "Uniform, C-256",\n'
     '                "J-UC128": "Joint, UC-128",\n'
     '                "J-C256": "Joint, C-256"}',
     '                "U-UC128": "Uniform, Tight",\n'
     '                "U-C256": "Uniform, Relaxed",\n'
     '                "J-UC128": "Joint, Tight",\n'
     '                "J-C256": "Joint, Relaxed"}'),
])

print("r4_assets.py")
patch(os.path.join(HERE, "r4_assets.py"), [
    # figure 1 is built elsewhere; this is the stale budget-chip block
    ('                [("UC-128", "<= 128 KB", "o"), ("C-256", "<= 256 KB", "s")]):',
     '                [("Tight", "<= 128 KB", "o"), ("Relaxed", "<= 256 KB", "s")]):'),
    ('    for budget, (ucfg, jcfg) in {"UC-128": ("U-UC128", "J-UC128"),\n'
     '                                 "C-256": ("U-C256", "J-C256")}.items():',
     '    for budget, (ucfg, jcfg) in {"Tight": ("U-UC128", "J-UC128"),\n'
     '                                 "Relaxed": ("U-C256", "J-C256")}.items():'),
    ('    budgets = [b for b in ("UC-128", "C-256")',
     '    budgets = [b for b in ("Tight", "Relaxed")'),
    ('                       marker="o" if bud == "UC-128" else "s",',
     '                       marker="o" if bud == "Tight" else "s",'),
    ('            cfg = ("U-UC128" if bud == "UC-128" else "U-C256")',
     '            cfg = ("U-UC128" if bud == "Tight" else "U-C256")'),
    ('BSHAPE = {"UC-128": "o", "C-256": "s"}',
     'BSHAPE = {"Tight": "o", "Relaxed": "s"}'),
    ('          for meth in ("uniform", "joint") for b in ("UC-128", "C-256")]',
     '          for meth in ("uniform", "joint") for b in ("Tight", "Relaxed")]'),
    ('           for bn in ("UC-128", "C-256")]',
     '           for bn in ("Tight", "Relaxed")]'),
    ('                     label=b) for b in ("UC-128", "C-256")]',
     '                     label=b) for b in ("Tight", "Relaxed")]'),
    ('            for bud, cfg in (("UC-128", "U-" if meth == "uniform" else "J-"),\n'
     '                             ("C-256", "U-" if meth == "uniform" else "J-")):\n'
     '                key = f"{m}__{cfg}{\'UC128\' if bud == \'UC-128\' else \'C256\'}"',
     '            pre = "U-" if meth == "uniform" else "J-"\n'
     '            for bud, suffix in (("Tight", "UC128"), ("Relaxed", "C256")):\n'
     '                key = m + "__" + pre + suffix'),
    ('                tag = "UC-128" if bud == "UC-128" else "C-256"',
     '                tag = bud'),
])

# figure 4 panel titles carry the budget in words as well as bytes
p = os.path.join(HERE, "r4_assets.py")
s = io.open(p, encoding="utf-8").read()
old = ('        ax.set_title(f"({chr(65 + k)}) {bud}  '
       '(\\u2264 {cap} KB INT8 weights)",')
new = ('        ax.set_title(f"({chr(65 + k)}) {bud} budget  '
       '(\\u2264 {cap} KB INT8 weights)",')
if old in s:
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("   figure4 panel title updated")
else:
    print("   figure4 panel title: pattern not found, left as is")

# the cached method contrast is keyed by budget name
cache = os.path.join(ROOT, "results", "recon_method_contrast.json")
if os.path.exists(cache):
    blob = json.load(open(cache))
    blob["delta"] = {k.replace("|" + OLD_T, "|" + NEW_T)
                      .replace("|" + OLD_R, "|" + NEW_R): v
                     for k, v in blob["delta"].items()}
    json.dump(blob, open(cache, "w"), indent=1)
    print("method-contrast cache: %d keys migrated" % len(blob["delta"]))
else:
    print("method-contrast cache: absent, will recompute")
