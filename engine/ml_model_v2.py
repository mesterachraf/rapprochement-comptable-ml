
import os, pickle, re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ml_v2.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "ml_v2_meta.pkl")

FEATURES = {
    "montant_somme_nulle":      "Montants — Somme algébrique → 0",
    "montant_diff_relative":    "Montants — Distance relative (%)",
    "montant_diff_absolue_log": "Montants — Différence absolue (log)",
    "montant_dc_oppose":        "Montants — Sens D/C opposés",
    "montant_meme_ordre":       "Montants — Même ordre de grandeur",
    "ref_identique":            "Références — Identiques exactes",
    "ref_sim_prefixe":          "Références — Préfixe commun (≥4 car.)",
    "ref_sim_suffixe":          "Références — Suffixe commun (≥4 car.)",
    "ref_jaccard":              "Références — Similarité Jaccard (tokens)",
    "ref_croisee":              "Références — Référence croisée (pièce↔rapproch)",

    "texte_jaccard":            "Libellé — Similarité Jaccard (mots clés)",
    "texte_mots_communs":       "Libellé — Nb mots communs (normalisé)",
    "texte_prefix_commun":      "Libellé — Préfixe libellé commun",
    "texte_longueur_sim":       "Libellé — Similarité longueur texte",
    "texte_vide":               "Libellé — L'un des textes est vide",

    "date_proximite":           "Contexte — Proximité des dates (norm.)",
    "date_meme_mois":           "Contexte — Même mois/année",
    "type_compatible":          "Contexte — Types de pièce compatibles",
    "meme_devise":              "Contexte — Même devise",
    "meme_fournisseur":         "Contexte — Même fournisseur",
}

FEATURE_NAMES = list(FEATURES.keys())

COMPAT = {
    frozenset(["SK","RE"]), frozenset(["KZ","RE"]),
    frozenset(["SA","LT"]), frozenset(["LT","LT"]),
    frozenset(["KA","RE"]), frozenset(["KR","KZ"]),
    frozenset(["SK","KR"]), frozenset(["KZ","KR"]),
}


def tokenize(s: str) -> set:
    """Tokenisation simple : mots de plus de 2 caractères, en majuscules."""
    if not s: return set()
    return set(w for w in re.split(r'\W+', s.upper()) if len(w) > 2)

def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def common_prefix_len(a: str, b: str) -> int:
    if not a or not b: return 0
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]: return i
    return n


def compute_features(r1: dict, r2: dict) -> np.ndarray:
    m1 = float(r1.get("montant", 0))
    m2 = float(r2.get("montant", 0))
    max_m = max(abs(m1), abs(m2), 1)
    somme = m1 + m2

    f_somme_nulle      = max(0.0, 1.0 - abs(somme) / max_m)
    f_diff_relative    = 1.0 - min(1.0, abs(somme) / max_m)
    try:
        f_diff_abs_log = 1.0 - min(1.0, abs(np.log1p(abs(m1)) - np.log1p(abs(m2))) / 5)
    except Exception:
        f_diff_abs_log = 0.5
    f_dc_oppose        = 1.0 if r1.get("dc","") != r2.get("dc","") else 0.0
    try:
        f_meme_ordre   = 1.0 if abs(np.log10(max(abs(m1),1)) - np.log10(max(abs(m2),1))) < 1 else 0.0
    except Exception:
        f_meme_ordre   = 0.5

    p1  = str(r1.get("piece",   "")).strip()
    p2  = str(r2.get("piece",   "")).strip()
    rp1 = str(r1.get("rapproch","")).strip()
    rp2 = str(r2.get("rapproch","")).strip()

    f_ref_identique  = 1.0 if (rp1 and rp1 == rp2) else 0.0
    pref_len         = common_prefix_len(rp1, rp2)
    f_ref_prefixe    = min(1.0, pref_len / 6) if pref_len >= 4 else 0.0
    suf1 = rp1[-4:] if len(rp1) >= 4 else ""
    suf2 = rp2[-4:] if len(rp2) >= 4 else ""
    f_ref_suffixe    = 1.0 if (suf1 and suf1 == suf2) else 0.0
    tok_r1, tok_r2   = tokenize(rp1), tokenize(rp2)
    f_ref_jaccard    = jaccard(tok_r1, tok_r2)
    f_ref_croisee    = 1.0 if (p1 == rp2 or p2 == rp1) else 0.0

    t1 = str(r1.get("texte","")).strip()
    t2 = str(r2.get("texte","")).strip()
    tok_t1, tok_t2   = tokenize(t1), tokenize(t2)
    f_texte_jaccard  = jaccard(tok_t1, tok_t2)
    common_words     = len(tok_t1 & tok_t2)
    max_words        = max(len(tok_t1), len(tok_t2), 1)
    f_texte_communs  = min(1.0, common_words / max_words)
    pref_t           = common_prefix_len(t1.upper(), t2.upper())
    f_texte_prefix   = min(1.0, pref_t / 10)
    l1, l2           = len(t1), len(t2)
    f_texte_longueur = 1.0 - min(1.0, abs(l1-l2) / max(l1,l2,1))
    f_texte_vide     = 1.0 if (not t1 or not t2) else 0.0

    try:
        d1   = pd.to_datetime(r1.get("date",""))
        d2   = pd.to_datetime(r2.get("date",""))
        days = abs((d1 - d2).days)
        f_date_prox   = max(0.0, 1.0 - days / 90)
        f_date_mois   = 1.0 if (d1.year == d2.year and d1.month == d2.month) else 0.0
    except Exception:
        f_date_prox, f_date_mois = 0.5, 0.0

    ty1, ty2 = str(r1.get("type","")), str(r2.get("type",""))
    f_type_compat = 1.0 if frozenset([ty1,ty2]) in COMPAT else 0.0
    f_meme_devise = 1.0 if r1.get("devise","") == r2.get("devise","") else 0.0
    f_meme_fourn  = 1.0 if str(r1.get("fournisseur","")) == str(r2.get("fournisseur","")) else 0.0

    return np.array([
        f_somme_nulle, f_diff_relative, f_diff_abs_log, f_dc_oppose, f_meme_ordre,
        f_ref_identique, f_ref_prefixe, f_ref_suffixe, f_ref_jaccard, f_ref_croisee,
        f_texte_jaccard, f_texte_communs, f_texte_prefix, f_texte_longueur, f_texte_vide,
        f_date_prox, f_date_mois, f_type_compat, f_meme_devise, f_meme_fourn,
    ], dtype=float)



def train_on_synthetic(data_dir: str = None) -> dict:
    """
    Entraîne le modèle sur les paires synthétiques (synthetic_pairs.csv).
    Si data_dir non fourni, cherche dans ../data/ relatif au module.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    pairs_path   = os.path.join(data_dir, "synthetic_pairs.csv")
    dataset_path = os.path.join(data_dir, "synthetic_dataset.xlsx")

    if not os.path.exists(pairs_path):
        return {"error": f"Fichier introuvable : {pairs_path}"}
    if not os.path.exists(dataset_path):
        return {"error": f"Fichier introuvable : {dataset_path}"}

    df_pairs   = pd.read_csv(pairs_path)
    df_dataset = pd.read_excel(dataset_path)
    rows_dict  = {int(r["id"]): r for r in df_dataset.to_dict("records")}

    X, y, cas_list = [], [], []
    missing = 0
    for _, pair in df_pairs.iterrows():
        ia, ib = int(pair["id_a"]), int(pair["id_b"])
        if ia not in rows_dict or ib not in rows_dict:
            missing += 1; continue
        feats = compute_features(rows_dict[ia], rows_dict[ib])
        X.append(feats)
        y.append(int(pair["label"]))
        cas_list.append(pair.get("cas",""))

    X, y = np.array(X), np.array(y)

    print(f"Paires chargées : {len(X)} ({missing} manquantes)")
    print(f"Positives : {y.sum()} | Négatives : {len(y)-y.sum()}")

    # Split stratifié
    X_tr, X_te, y_tr, y_te, cas_tr, cas_te = train_test_split(
        X, y, cas_list, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:,1]
    report  = classification_report(y_te, y_pred, output_dict=True)
    auc     = roc_auc_score(y_te, y_proba)
    cm      = confusion_matrix(y_te, y_pred).tolist()

    cas_perf = {}
    cas_arr  = np.array(cas_te)
    for cas in set(cas_arr):
        mask = cas_arr == cas
        if mask.sum() == 0: continue
        y_c = y_te[mask]; yp_c = y_pred[mask]
        acc = (y_c == yp_c).mean()
        cas_perf[cas] = round(float(acc), 3)

    importances = {
        FEATURE_NAMES[i]: round(float(v), 4)
        for i, v in enumerate(model.feature_importances_)
    }
    familles = {
        "Montants":    sum(model.feature_importances_[:5]),
        "Références":  sum(model.feature_importances_[5:10]),
        "Libellé":     sum(model.feature_importances_[10:15]),
        "Contexte":    sum(model.feature_importances_[15:]),
    }

    meta = {
        "version":       "v2-synthetic",
        "algo":          "Random Forest (300 arbres)",
        "n_train":       len(X_tr),
        "n_test":        len(X_te),
        "n_positif":     int(y.sum()),
        "n_negatif":     int(len(y)-y.sum()),
        "auc":           round(float(auc), 4),
        "precision":     round(float(report["1"]["precision"]), 4),
        "recall":        round(float(report["1"]["recall"]), 4),
        "f1":            round(float(report["1"]["f1-score"]), 4),
        "accuracy":      round(float(report["accuracy"]), 4),
        "confusion_matrix": cm,
        "importances":   importances,
        "familles":      {k: round(float(v),4) for k,v in familles.items()},
        "perf_par_cas":  cas_perf,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
    with open(META_PATH,  "wb") as f: pickle.dump(meta,  f)

    print(f"\n✅ Modèle sauvegardé")
    print(f"   AUC     : {meta['auc']}")
    print(f"   F1      : {meta['f1']}")
    print(f"   Accuracy: {meta['accuracy']}")
    print(f"\nPerformance par cas :")
    for k,v in cas_perf.items(): print(f"   {k}: {v*100:.1f}%")
    print(f"\nImportance par famille :")
    for k,v in familles.items(): print(f"   {k}: {v*100:.1f}%")

    return meta



def is_trained() -> bool:
    return os.path.exists(MODEL_PATH)

def load_model():
    with open(MODEL_PATH, "rb") as f: return pickle.load(f)

def load_meta() -> dict:
    if not os.path.exists(META_PATH): return {}
    with open(META_PATH, "rb") as f:  return pickle.load(f)


def predict_and_explain(df: pd.DataFrame, matched: set, threshold: float = 0.60) -> list:
   
    if not is_trained():
        return []

    model  = load_model()
    unm    = df[~df.index.isin(matched) & (df["montant"] != 0)]
    rows   = unm.to_dict("records")
    idxs   = unm.index.tolist()
    n      = len(rows)
    if n < 2: return []

    pairs_list = []
    for i in range(n):
        for j in range(i+1, n):
            feats = compute_features(rows[i], rows[j])
            pairs_list.append((i, j, feats))

    if not pairs_list: return []

    X      = np.array([p[2] for p in pairs_list])
    probas = model.predict_proba(X)[:,1]

    scored = sorted(zip(probas, pairs_list), key=lambda x: -x[0])
    used   = set()
    groups = []
    gid    = 9000

    for prob, (i, j, feats) in scored:
        if prob < threshold: break
        if i in used or j in used: continue
        used.add(i); used.add(j)

        ra, rb = rows[i], rows[j]
        somme  = ra["montant"] + rb["montant"]
        feat_d = {FEATURE_NAMES[k]: round(float(feats[k]),4) for k in range(len(FEATURE_NAMES))}

        groups.append({
            "id":          f"GRP-{gid:04d}",
            "fournisseur": ra.get("fournisseur",""),
            "items":       [ra, rb],
            "indices":     [idxs[i], idxs[j]],
            "somme":       round(somme, 2),
            "ecart":       round(somme, 2),
            "moteur":      "ML",
            "match_type":  "1-1",
            "regle":       f"Random Forest (synthétique) — confiance {prob*100:.1f}%",
            "score":       round(float(prob), 4),
            "features":    feat_d,
            "valid":       abs(somme) < max(abs(ra["montant"]),1) * 0.05,
            "steps":       _explain(ra, rb, feat_d, float(prob), threshold),
        })
        gid += 1

    return groups


def _explain(ra, rb, feat, score, threshold) -> list:
    """Justification explicable par famille — CDC section 4.2.4."""

    f1_score = np.mean([feat["montant_somme_nulle"], feat["montant_diff_relative"],
                        feat["montant_dc_oppose"]])
    f2_score = np.mean([feat["ref_identique"], feat["ref_sim_prefixe"],
                        feat["ref_jaccard"], feat["ref_croisee"]])
    f3_score = np.mean([feat["texte_jaccard"], feat["texte_mots_communs"],
                        feat["texte_prefix_commun"]])
    f4_score = np.mean([feat["date_proximite"], feat["type_compatible"],
                        feat["meme_devise"]])

    steps = [
        f"🤖 Algorithme : Random Forest 300 arbres — entraîné sur dataset synthétique multi-sources",
        f"📄 Écriture A : {ra.get('piece','')} | {ra.get('montant',0):+.2f} {ra.get('devise','')} | {ra.get('dc','')} | {ra.get('date','')}",
        f"📄 Écriture B : {rb.get('piece','')} | {rb.get('montant',0):+.2f} {rb.get('devise','')} | {rb.get('dc','')} | {rb.get('date','')}",
        f"🧮 Scores par famille de features (CDC section 4.2.3) :",
        f"   • Famille Montants    : {f1_score*100:.0f}%  — somme→0={feat['montant_somme_nulle']*100:.0f}%, D/C opposés={'Oui' if feat['montant_dc_oppose'] else 'Non'}",
        f"   • Famille Références  : {f2_score*100:.0f}%  — identique={'Oui' if feat['ref_identique'] else 'Non'}, Jaccard={feat['ref_jaccard']*100:.0f}%, croisée={'Oui' if feat['ref_croisee'] else 'Non'}",
        f"   • Famille Libellé     : {f3_score*100:.0f}%  — Jaccard texte={feat['texte_jaccard']*100:.0f}%, mots communs={feat['texte_mots_communs']*100:.0f}%",
        f"   • Famille Contexte    : {f4_score*100:.0f}%  — proximité date={feat['date_proximite']*100:.0f}%, type compatible={'Oui' if feat['type_compatible'] else 'Non'}",
        f"🌲 Vote de 300 arbres → Score de confiance global : {score*100:.1f}%",
        f"🎯 Seuil configuré : {threshold*100:.0f}% → {'✅ ACCEPTÉ' if score >= threshold else '❌ REJETÉ'}",
    ]
    return steps


if __name__ == "__main__":
    print("=== ENTRAÎNEMENT MODÈLE ML (dataset synthétique) ===\n")
    meta = train_on_synthetic()
    if "error" in meta:
        print(f"ERREUR : {meta['error']}")
    else:
        print("\n✅ Modèle prêt pour inférence.")
