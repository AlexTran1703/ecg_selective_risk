"""Remove the supplementary material and re-own Table 1.

Table 1 was produced by the previous asset builder, whose caption
pointed at a supplementary prevalence table. That table is being
dropped, so the caption has to go with it, and Table 1 moves into the
current builder. It takes no arguments there: the old signature
accepted runs and bootstrap results but used neither -- the table is
derived entirely from the cohort.
"""
import io
import os

T1 = '''def table1(runs=None):
    """Composition of the four harmonised ECG sources.

    Entirely cohort-derived: no model, no fold and no bootstrap enters
    it, which is why it is unchanged by the reconstruction.
    """
    c = B.cohort()
    m, y, cls = c["index"], c["y"], c["classes"]
    rows = []
    for src in SOURCES:
        k = (m["source"] == src).values
        sub, lab = m[k], y[k]
        age = sub["age"].dropna()
        nf = int((sub["sex"] == "Female").sum())
        rows.append({
            "source": B.sname(src),
            "records": "{:,}".format(int(k.sum())),
            "patients": "{:,}".format(sub["patient"].nunique()),
            "age, median (IQR)": (
                "{:.0f} ({:.0f}-{:.0f})".format(
                    age.median(), age.quantile(.25), age.quantile(.75))
                if len(age) else "NR"),
            "female (%)": "{:.1f}".format(100 * nf / k.sum()),
            "labels per ECG": "{:.2f}".format(lab.sum(1).mean()),
            # The macro-average of per-class prevalence is the expected
            # average precision of a random ranking on this source. It
            # is a reference, not a bound: a model can score below it.
            "random-ranking AP reference": "{:.3f}".format(lab.mean())})
    rows.append({
        "source": "Total",
        "records": "{:,}".format(len(m)),
        "patients": "{:,}".format(m["patient"].nunique()),
        "age, median (IQR)": "",
        "female (%)": "{:.1f}".format(
            100 * (m["sex"] == "Female").sum() / len(m)),
        "labels per ECG": "{:.2f}".format(y.sum(1).mean()),
        "random-ranking AP reference": "{:.3f}".format(y.mean())})
    df = pd.DataFrame(rows)
    save_table(df, "table1_protocol",
               "Table 1. Composition of the four harmonised ECG sources. "
               "Counts are records retained after harmonisation to the "
               "13-label space; records carrying no label inside it were "
               "dropped before any splitting. Patients are unique within a "
               "source, so the ratio to records shows that only PTB-XL "
               "repeats patients (17,928 patients over 20,487 records); "
               "identifiers are not comparable between sources, so the "
               "Total row sums within-source counts. Sex is normalised "
               "across sources, which record it differently; age is the "
               "median over records where it is present. Labels per ECG is "
               "the mean number of positive labels. The random-ranking AP "
               "reference is the macro-average of per-class prevalence, "
               "given so that Table 3 can be read against something other "
               "than zero; it is a reference, not a bound, and a model can "
               "score below it. NSR is not coded comparably across these "
               "four sources and its prevalence should not be read as a "
               "population difference. Inputs are 10 s, 12 leads, decimated "
               "to 100 Hz.")
    return df


'''

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
p = os.path.join(HERE, "r4_assets.py")
s = io.open(p, encoding="utf-8").read()

# insert table1 ahead of the architecture table
anchor = "def table_architectures():"
assert anchor in s, "anchor missing"
s = s.replace(anchor, T1 + anchor, 1)

# register it, and drop the supplementary reference from Table 3
s = s.replace('    jobs = [("figure1", _fig1.figure1),',
              '    jobs = [("figure1", _fig1.figure1),\n'
              '            ("table1", table1),', 1)
old = ('               "intervals. Source-specific estimates and macro-AUROC "\n'
       '               "are in the supplementary tables.")')
if old in s:
    s = s.replace(old, '               "intervals.")', 1)
    print("table 3 caption: supplementary reference removed")
else:
    print("table 3 caption: pattern not found, check manually")
io.open(p, "w", encoding="utf-8").write(s)
print("table1 ported and registered")

removed = []
for e in ("csv", "tex", "txt"):
    f = os.path.join(ROOT, "tables", "tableS1_prevalence." + e)
    if os.path.exists(f):
        os.remove(f)
        removed.append(os.path.basename(f))
print("removed: %s" % (", ".join(removed) if removed else "nothing"))
