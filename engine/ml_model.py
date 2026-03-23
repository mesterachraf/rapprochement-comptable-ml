

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rf_model.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "meta.pkl")

FEATURE_NAMES = [
    "montant_sum_proche_zero",
    "montant_diff_relative",
    "dc_oppose",
    "texte_sim_score",
    "ref_croisee",
    "date_diff_jours_norm",
    "type_compatible",
    "meme_devise",
    "montant_abs_log_diff",
]

FEATURE_LABELS = {
    "montant_sum_proche_zero":  "Équilibre montant (somme → 0)",
    "montant_diff_relative":    "Proximité montants (%)",
    "dc_oppose":                "Sens D/C opposés (H vs S)",
    "texte_sim_score":          "Similarité libellé texte",
    "ref_croisee":              "Référence croisée (pièce ↔ rapproch)",
    "date_diff_jours_norm":     "Proximité des dates",
    "type_compatible":          "Compatibilité type pièce",
    "meme_devise":              "Même devise",
    "montant_abs_log_diff":     "Magnitude montants (log)",
}



def _text_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a, b = a.lower(), b.lower()
    wa = set(w for w in a.split() if len(w) > 3)
    wb = set(w for w in b.split() if len(w) > 3)
    if not wa or not wb:
        return 0.0
    common = sum(1 for w in wa if any(w in x or x in w for x in wb))
    return min(1.0, common / max(len(wa), len(wb)))


COMPAT_TYPES = {
    frozenset(["SK", "KR"]), frozenset(["KZ", "RE"]),
    frozenset(["SA", "LT"]), frozenset(["LT", "LT"]),
    frozenset(["KA", "RE"]), frozenset(["KR", "KZ"]),
}


def compute_features(r1: dict, r2: dict) -> np.ndarray:
    m1, m2 = float(r1.get("montant", 0)), float(r2.get("montant", 0))
    somme   = m1 + m2
    max_m   = max(abs(m1), abs(m2), 1)

    f1 = max(0.0, 1.0 - abs(somme) / max_m)

    f2 = 1.0 - min(1.0, abs(somme) / max_m)

    f3 = 1.0 if r1.get("dc") != r2.get("dc") else 0.0

    f4 = _text_sim(str(r1.get("texte", "")), str(r2.get("texte", "")))

    p1, p2 = str(r1.get("piece", "")), str(r2.get("piece", ""))
    rp1, rp2 = str(r1.get("rapproch", "")), str(r2.get("rapproch", ""))
    f5 = 1.0 if (p1 == rp2 or p2 == rp1 or rp1 == rp2) else 0.0

    try:
        d1 = pd.to_datetime(r1.get("date"))
        d2 = pd.to_datetime(r2.get("date"))
        days = abs((d1 - d2).days)
        f6 = max(0.0, 1.0 - days / 90)
    except Exception:
        f6 = 0.5

    t1, t2 = str(r1.get("type", "")), str(r2.get("type", ""))
    f7 = 1.0 if frozenset([t1, t2]) in COMPAT_TYPES else 0.0

    f8 = 1.0 if r1.get("devise") == r2.get("devise") else 0.0

    try:
        f9 = 1.0 - min(1.0, abs(np.log1p(abs(m1)) - np.log1p(abs(m2))) / 5)
    except Exception:
        f9 = 0.5

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9], dtype=float)



def build_training_pairs(df: pd.DataFrame, max_neg_ratio: int = 3):
    """
    Construit les paires positives (même rapproch) et négatives (diff rapproch)
    à partir du champ 'rapproch' comme ground truth.
    """
    rows = df.to_dict("records")
    n    = len(rows)
    X, y = [], []

    byref = {}
    for r in rows:
        ref = str(r.get("rapproch", ""))
        if ref:
            byref.setdefault(ref, []).append(r)

    pos_pairs = []
    for ref, grp in byref.items():
        if len(grp) >= 2:
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    pos_pairs.append((grp[i], grp[j], 1))

    neg_pairs = []
    by_fourn = {}
    for r in rows:
        by_fourn.setdefault(r["fournisseur"], []).append(r)

    rng = np.random.default_rng(42)
    for f, grp in by_fourn.items():
        if len(grp) < 2:
            continue
        idxs = list(range(len(grp)))
        rng.shuffle(idxs)
        for i in range(min(len(idxs), len(pos_pairs) * max_neg_ratio)):
            for j in range(i + 1, min(len(idxs), i + 4)):
                ra, rb = grp[idxs[i]], grp[idxs[j]]
                if str(ra.get("rapproch","")) != str(rb.get("rapproch","")):
                    neg_pairs.append((ra, rb, 0))

    neg_sample = neg_pairs[:len(pos_pairs) * max_neg_ratio]
    all_pairs  = pos_pairs + neg_sample
    rng.shuffle(all_pairs)

    for ra, rb, label in all_pairs:
        feats = compute_features(ra, rb)
        X.append(feats)
        y.append(label)

    return np.array(X), np.array(y)



def train(df: pd.DataFrame) -> dict:
    """Entraîne le modèle et sauvegarde sur disque. Retourne les métriques."""
    X, y = build_training_pairs(df)

    if len(X) < 10:
        return {"error": "Pas assez de paires d'entraînement (minimum 10 requis)."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report  = classification_report(y_test, y_pred, output_dict=True)
    auc     = roc_auc_score(y_test, y_proba)

    importances = {
        FEATURE_NAMES[i]: round(float(v), 4)
        for i, v in enumerate(model.feature_importances_)
    }

    meta = {
        "n_train":     len(X_train),
        "n_test":      len(X_test),
        "n_pos":       int(y.sum()),
        "n_neg":       int(len(y) - y.sum()),
        "auc":         round(auc, 4),
        "precision":   round(report["1"]["precision"], 4),
        "recall":      round(report["1"]["recall"], 4),
        "f1":          round(report["1"]["f1-score"], 4),
        "importances": importances,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    return meta


def is_trained() -> bool:
    return os.path.exists(MODEL_PATH)


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def load_meta() -> dict:
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, "rb") as f:
        return pickle.load(f)



def predict_and_explain(df: pd.DataFrame, matched: set, threshold: float = 0.65) -> list:
    """
    Applique le modèle sur les écritures non rapprochées.
    Retourne des groupes ML avec score et explication détaillée.
    """
    if not is_trained():
        return []

    model   = load_model()
    unm     = df[~df.index.isin(matched) & (df["montant"] != 0)]
    rows    = unm.to_dict("records")
    idxs    = unm.index.tolist()
    n       = len(rows)

    if n < 2:
        return []

    # Calcul matriciel des features
    pairs   = []
    for i in range(n):
        for j in range(i + 1, n):
            feats = compute_features(rows[i], rows[j])
            pairs.append((i, j, feats))

    if not pairs:
        return []

    X       = np.array([p[2] for p in pairs])
    probas  = model.predict_proba(X)[:, 1]

   
    scored  = sorted(zip(probas, pairs), key=lambda x: -x[0])
    used    = set()
    groups  = []
    gid     = 9000

    for prob, (i, j, feats) in scored:
        if prob < threshold:
            break
        if i in used or j in used:
            continue
        used.add(i); used.add(j)

        ra, rb  = rows[i], rows[j]
        somme   = ra["montant"] + rb["montant"]
        feat_d  = {FEATURE_NAMES[k]: round(float(feats[k]), 4) for k in range(len(FEATURE_NAMES))}
        steps   = _explain_ml(ra, rb, feat_d, float(prob), threshold)

        groups.append({
            "id":          f"GRP-{gid:04d}",
            "fournisseur": ra["fournisseur"],
            "items":       [ra, rb],
            "indices":     [idxs[i], idxs[j]],
            "somme":       round(somme, 2),
            "ecart":       round(somme, 2),
            "moteur":      "ML",
            "match_type":  "1-1",
            "regle":       f"Random Forest — score {prob*100:.1f}%",
            "score":       round(float(prob), 4),
            "features":    feat_d,
            "valid":       abs(somme) < max(abs(ra["montant"]), 1) * 0.05,
            "steps":       steps,
        })
        gid += 1

    return groups


def _explain_ml(ra: dict, rb: dict, feat: dict, score: float, threshold: float) -> list:
    top = sorted(feat.items(), key=lambda x: -x[1])[:3]
    top_labels = [f'"{FEATURE_LABELS.get(k, k)}" ({v*100:.0f}%)' for k, v in top]

    steps = [
        f"🤖 Modèle : Random Forest (200 arbres, entraîné sur données historiques)",
        f"📄 Écriture A : {ra['piece']} | {ra['montant']:+.2f} {ra.get('devise','')} | {ra.get('dc','')} | {ra.get('date','')}",
        f"📄 Écriture B : {rb['piece']} | {rb['montant']:+.2f} {rb.get('devise','')} | {rb.get('dc','')} | {rb.get('date','')}",
        f"🧮 Calcul de 9 features pour cette paire :",
        f"   • Équilibre montant : {feat['montant_sum_proche_zero']*100:.0f}%",
        f"   • Sens D/C opposés : {'Oui' if feat['dc_oppose'] else 'Non'}",
        f"   • Similarité texte : {feat['texte_sim_score']*100:.0f}%",
        f"   • Référence croisée : {'Oui' if feat['ref_croisee'] else 'Non'}",
        f"   • Proximité date : {feat['date_diff_jours_norm']*100:.0f}%",
        f"🌲 Vote des 200 arbres → score de confiance : {score*100:.1f}%",
        f"📊 Variables les plus influentes : {', '.join(top_labels)}",
        f"🎯 Seuil configuré : {threshold*100:.0f}% → "
        f"{'✅ ACCEPTÉ' if score >= threshold else '❌ REJETÉ'}",
    ]
    return steps
