"""Remove the entries flagged as not supporting a direct claim.

Four go. Each for a stated reason, not for being unverified -- an
unverified entry gets verified, whereas these would be cited for
something they do not show.

  Sensors2025ECGAnomaly    edge ECG anomaly detection, but on hardware
                           that cannot be assumed equivalent to an
                           STM32F411, so it supports no comparison here
  TinyMLCardio2025Survey   a survey, not a deployment measurement; it
                           can only be cited for framing, and the
                           framing is already carried by MLPerf Tiny
                           and MCUNet
  Gitau2025ReproducePTBXL  a preprint whose title matches
                           Strodthoff2021PTBXL at 0.96 because it
                           reproduces it; citing both reads as a
                           duplicated reference
  SklearnAP                software documentation. The metric's
                           definition belongs to Davis and Goadrich and
                           to Saito and Rehmsmeier; what the library
                           fixes is a version number, which belongs in
                           Methods as prose

Kept despite being flagged, because the flag is incompleteness rather
than irrelevance: ATC2024TinyMLECG (its ~143 KB INT8 model is the
closest published comparator to the 128 KB budget), Kim2023TinyMLECG
(~27 KB deployed ECG model), Akbari2025AFEdge (physical low-power MCU
energy measurement). All three still need author lists.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(os.path.dirname(HERE), "reference.bib")

DROP = ["Sensors2025ECGAnomaly", "TinyMLCardio2025Survey",
        "Gitau2025ReproducePTBXL", "SklearnAP"]

s = io.open(p, encoding="utf-8").read()
removed = []
for key in DROP:
    # take the entry and any comment block immediately above it
    pat = re.compile(r"(?:^%[^\n]*\n)*^@\w+\s*\{\s*" + re.escape(key)
                     + r",.*?\n\}\n", re.S | re.M)
    new, n = pat.subn("", s)
    if n:
        s, _ = new, removed.append(key)
    else:
        print("  not found: %s" % key)
s = re.sub(r"\n{4,}", "\n\n\n", s)
io.open(p, "w", encoding="utf-8").write(s)

print("removed %d entries:" % len(removed))
for k in removed:
    print("   %s" % k)
out = io.open(p, encoding="utf-8").read()
print("\nremaining    : %d" % out.lower().count("\n@"))
print("braces       : %d / %d" % (out.count("{"), out.count("}")))
print("FIXME        : %d" % out.count("FIXME"))
