
import pandas as pd
import hashlib
import json
import re

from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

# ============================================================
# SCI1402 — ÉVALUATION FINALE DU NLP V2
# 40 dossiers de test × 15 catégories = 600 annotations
# ============================================================

P = Path("/kaggle/working")

test = pd.read_csv(
    P / "annotation_test.csv",
    encoding="utf-8-sig",
    dtype={"Annotation ID": str}
)

# Charger exclusivement les règles et la fonction detect()
# depuis le script V2 gelé.
src = (P / "sci1402_nlp_test_frozen.py").read_text(
    encoding="utf-8"
)

namespace = {"pd": pd, "re": re}

exec(
    src[
        src.index("# Règles générales"):
        src.index("# Prédictions indépendantes")
    ],
    namespace
)

CATEGORIES = [
    "Method Firearm",
    "Method Knife",
    "Method Strangulation",
    "Method Poisoning",
    "Method Blunt Force",
    "Method Other",
    "Victims Women",
    "Victims Men",
    "Victims Children",
    "Victims Elderly",
    "Victims Sex Workers",
    "Sexual Violence Mentioned",
    "Robbery Mentioned",
    "Dismemberment Mentioned",
    "Accomplices Mentioned",
]

detect = namespace["detect"]
rules = namespace["RULES"]

# ============================================================
# 1. ANNOTATIONS HUMAINES VALIDÉES
#
# Chaque numéro correspond au suffixe de Annotation ID.
# Les catégories absentes de la liste valent 0.
# ============================================================

positives = {
    3: [
        "Method Firearm",
        "Method Strangulation",
        "Victims Men",
        "Victims Elderly",
    ],
    5: ["Victims Women", "Victims Children"],
    7: [
        "Method Strangulation",
        "Victims Women",
        "Sexual Violence Mentioned",
    ],
    8: [
        "Victims Women",
        "Sexual Violence Mentioned",
        "Robbery Mentioned",
    ],
    10: ["Method Poisoning"],
    11: ["Victims Men", "Victims Children"],
    13: ["Victims Children"],
    14: ["Victims Children"],
    15: ["Method Poisoning", "Victims Elderly"],
    20: ["Victims Women"],
    21: [],
    22: ["Robbery Mentioned"],
    23: ["Accomplices Mentioned"],
    29: [
        "Sexual Violence Mentioned",
        "Robbery Mentioned",
    ],
    30: [
        "Method Strangulation",
        "Victims Elderly",
        "Robbery Mentioned",
    ],
    31: ["Victims Women"],
    32: ["Method Strangulation", "Victims Women"],
    34: [
        "Victims Men",
        "Victims Children",
        "Sexual Violence Mentioned",
    ],
    38: ["Accomplices Mentioned"],
    39: [
        "Victims Sex Workers",
        "Dismemberment Mentioned",
    ],
    42: ["Method Poisoning"],
    44: ["Method Firearm"],
    47: ["Method Poisoning", "Victims Children"],
    49: ["Accomplices Mentioned"],
    50: [
        "Victims Women",
        "Victims Men",
        "Accomplices Mentioned",
    ],
    52: ["Victims Sex Workers"],
    54: [],
    56: ["Method Strangulation", "Victims Women"],
    58: ["Method Poisoning"],
    60: [],
    62: ["Method Firearm"],
    63: ["Victims Women"],
    64: ["Victims Women"],
    66: ["Accomplices Mentioned"],
    68: ["Method Strangulation", "Victims Women"],
    74: ["Victims Sex Workers"],
    75: ["Victims Women"],
    76: [
        "Method Knife",
        "Method Strangulation",
        "Method Blunt Force",
        "Victims Women",
        "Victims Children",
    ],
    77: ["Victims Men"],
    80: [
        "Method Blunt Force",
        "Victims Elderly",
        "Victims Sex Workers",
        "Dismemberment Mentioned",
    ],
}

# ============================================================
# 2. NOTES MÉTHODOLOGIQUES
# ============================================================

notes_special = {
    5: "Stabbing describes death of perpetrator, not victims.",
    21: "All 15 categories absent from Notes.",
    29: "Burglary included in Robbery Mentioned.",
    32: "Sister-in-law establishes female homicide victim.",
    38: "Possible accomplice counted as mentioned.",
    39: (
        "Sex workers gender unspecified; "
        "dismemberment not necessarily killing method."
    ),
    49: "Two co-offenders explicitly identified in Name.",
    50: (
        "Two co-offenders explicitly identified in Name "
        "(user-validated exception)."
    ),
    52: "Strangler appears only in nickname.",
    54: "All 15 categories absent from Notes.",
    60: (
        "ATM targeting not explicit robbery; "
        "police shot perpetrator."
    ),
    62: (
        "Revolver explicitly used to kill; "
        "nickname not robbery evidence."
    ),
    63: (
        "Crack cocaine describes perpetrator, "
        "not poisoning method."
    ),
    64: "Strangler only in nickname.",
    66: "Female describes accomplices, not victims.",
    68: "Kidnapping/binding not an additional killing method.",
    74: (
        "Hanging describes perpetrator suicide; "
        "sex worker gender unspecified."
    ),
    75: "Two disappearances not confirmed deaths.",
    76: (
        "Unborn children included in Victims Children "
        "by explicit user validation."
    ),
    77: (
        "Hanging describes perpetrator execution; "
        "sister/nephew attacked, not confirmed killed."
    ),
    80: (
        "Decapitation counted as dismemberment; "
        "no knife instrument or distinct killing "
        "method established."
    ),
}

# ============================================================
# 3. VÉRIFICATION DES 40 DOSSIERS
# ============================================================

assert len(positives) == 40
assert len(test) == 40

ids = {
    int(s.split("-")[-1])
    for s in test["Annotation ID"]
}

assert ids == set(positives)

assert all(
    set(values) <= set(CATEGORIES)
    for values in positives.values()
)

# ============================================================
# 4. CONSTRUCTION DE LA RÉFÉRENCE HUMAINE
# ============================================================

for i, row in test.iterrows():

    num = int(row["Annotation ID"].split("-")[-1])
    validated = set(positives[num])

    for category in CATEGORIES:
        test.at[i, category] = int(category in validated)

    # Conserver le texte original comme preuve.
    test.at[i, "Evidence Quote"] = row["Notes"]

    test.at[i, "Uncertainty Notes"] = notes_special.get(
        num,
        (
            "Only explicitly described characteristics "
            "are annotated; unmentioned categories = 0."
        ),
    )

    test.at[i, "Annotation Status"] = (
        "Validated by user in conversation"
    )

assert test[CATEGORIES].notna().all().all()

reference_path = (
    P / "sci1402_test_ground_truth_40_cases.csv"
)

test.to_csv(
    reference_path,
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# 5. CONVERSION EN 600 ANNOTATIONS (FORMAT LONG)
# ============================================================

long = test.melt(
    id_vars=[
        "Annotation ID",
        "Name",
        "Cluster",
        "Notes",
        "Annotation Status",
    ],
    value_vars=CATEGORIES,
    var_name="Category",
    value_name="Human Label",
)

long["Human Label"] = long["Human Label"].astype(int)

long.to_csv(
    P / "sci1402_test_ground_truth_600_labels.csv",
    index=False,
    encoding="utf-8-sig",
)

assert len(long) == 600

# ============================================================
# 6. RECONSTITUTION DES PRÉDICTIONS V2 GELÉES
# ============================================================

rows = []

for case in test.itertuples(index=False):

    for category in CATEGORIES:

        rows.append({
            "Annotation ID": case[0],
            "Name": case.Name,
            "Cluster": case.Cluster,
            "Category": category,
            "Notes": case.Notes,
            "Predicted V2": detect(case.Notes, category),
        })

pred = pd.DataFrame(rows)

# ============================================================
# 7. CONTRÔLE D'INTÉGRITÉ SHA-256
# ============================================================

rule_hash = hashlib.sha256(
    json.dumps(
        rules,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()

pred_bytes = pred.to_csv(
    index=False
).encode("utf-8-sig")

pred_hash = hashlib.sha256(
    pred_bytes
).hexdigest()

EXPECTED_RULE_HASH = (
    "c13d52ecef8982a0da3f9a6c33f3b2250835beb208fe15ae5745f73b5a3352a6"
)

EXPECTED_PRED_HASH = (
    "22a1cbec3655c66e0196e1fc789250842d0ae0d92ac72cc75e3db117019c005f"
)

print("Rule SHA-256:", rule_hash)
print("Prediction SHA-256:", pred_hash)

if (
    rule_hash != EXPECTED_RULE_HASH
    or pred_hash != EXPECTED_PRED_HASH
):
    raise RuntimeError(
        "Les empreintes ne correspondent pas "
        "aux prédictions V2 gelées. "
        "Évaluation interrompue."
    )

print("Intégrité V2 : CONFIRMÉE")

(P / "nlp_test_predictions_v2_frozen_verified.csv").write_bytes(
    pred_bytes
)

# ============================================================
# 8. COMPARAISON PRÉDICTIONS VS ANNOTATIONS HUMAINES
# ============================================================

evaldf = pred.merge(
    long[
        [
            "Annotation ID",
            "Category",
            "Human Label",
        ]
    ],
    on=["Annotation ID", "Category"],
    validate="one_to_one",
)

def classify_outcome(row):

    human = int(row["Human Label"])
    predicted = int(row["Predicted V2"])

    if human == 1 and predicted == 1:
        return "TP"

    if human == 0 and predicted == 0:
        return "TN"

    if human == 0 and predicted == 1:
        return "FP"

    return "FN"


evaldf["Outcome"] = evaldf.apply(
    classify_outcome,
    axis=1,
)

evaldf.to_csv(
    P / "sci1402_test_v2_comparison_600.csv",
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# 9. MÉTRIQUES GLOBALES ET PAR CATÉGORIE
# ============================================================

summary = []

groups = [("GLOBAL", evaldf)]

groups += [
    (
        category,
        evaldf[evaldf["Category"] == category],
    )
    for category in CATEGORIES
]

for name, subset in groups:

    y_true = subset["Human Label"]
    y_pred = subset["Predicted V2"]

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
    )

    counts = subset["Outcome"].value_counts()

    summary.append({
        "Category": name,
        "N": len(subset),
        "Positives human": int(y_true.sum()),
        "Positives V2": int(y_pred.sum()),
        "TP": int(counts.get("TP", 0)),
        "FP": int(counts.get("FP", 0)),
        "FN": int(counts.get("FN", 0)),
        "TN": int(counts.get("TN", 0)),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    })

metrics = pd.DataFrame(summary)

metrics.to_csv(
    P / "sci1402_test_v2_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# 10. EXTRACTION DES ERREURS
# ============================================================

errors = evaldf[
    evaldf["Outcome"].isin(["FP", "FN"])
].copy()

errors.to_csv(
    P / "sci1402_test_v2_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# 11. MANIFESTE DE REPRODUCTIBILITÉ
# ============================================================

manifest = {
    "dataset": "SCI1402 test",
    "test_cases": 40,
    "categories": 15,
    "human_annotations": 600,
    "frozen_rule_sha256": rule_hash,
    "frozen_prediction_sha256": pred_hash,
    "frozen_prediction_hash_matches_original": True,
    "reference_file": reference_path.name,
    "note": (
        "Human annotations reconstructed from "
        "40 explicitly validated chat dossiers; "
        "V2 rules unchanged."
    ),
}

(
    P / "sci1402_test_evaluation_manifest.json"
).write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

# ============================================================
# 12. AFFICHAGE DES RÉSULTATS
# ============================================================

print("\n" + "=" * 65)
print("SCI1402 — ÉVALUATION FINALE NLP V2")
print("=" * 65)

print("\nMÉTRIQUES")
print(metrics.to_string(index=False))

print("\nERREURS")
print(
    errors[
        [
            "Annotation ID",
            "Name",
            "Category",
            "Human Label",
            "Predicted V2",
            "Outcome",
        ]
    ].to_string(index=False)
)

print("\nFICHIERS GÉNÉRÉS")

for filename in [
    "sci1402_test_ground_truth_40_cases.csv",
    "sci1402_test_ground_truth_600_labels.csv",
    "nlp_test_predictions_v2_frozen_verified.csv",
    "sci1402_test_v2_comparison_600.csv",
    "sci1402_test_v2_metrics.csv",
    "sci1402_test_v2_errors.csv",
    "sci1402_test_evaluation_manifest.json",
]:
    print(P / filename)

print("\nÉvaluation terminée.")