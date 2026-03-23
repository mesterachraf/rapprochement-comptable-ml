import os, io
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename

from engine.rules       import run_rules
from engine.diagnostic  import analyser_non_resolus
from engine.ml_model_v2 import (
    train_on_synthetic, predict_and_explain,
    is_trained, load_meta, FEATURE_NAMES, FEATURES
)
from engine.exporter    import export_results

app = Flask(__name__)
app.secret_key = "rapprochement-secret-2025"

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
DATA_FOLDER   = os.path.join(os.path.dirname(__file__), "data")
ALLOWED_EXT   = {"xlsx","xls"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER,   exist_ok=True)

_sessions = {}

def allowed_file(fname):
    return "." in fname and fname.rsplit(".",1)[1].lower() in ALLOWED_EXT

def load_excel(path):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    col_map = {
        "Numéro de pièce":                    "piece",
        "Fournisseur":                         "fournisseur",
        "Montant en devise de la transaction": "montant",
        "Code débit/crédit":                   "dc",
        "Date comptable":                      "date",
        "Texte":                               "texte",
        "Type de pièce":                       "type",
        "Devise de transaction":               "devise",
        "Pièce rapprochement":                 "rapproch",
    }
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
    missing = [c for c in ["piece","fournisseur","montant"] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    df["montant"]     = pd.to_numeric(df["montant"], errors="coerce").fillna(0)
    df["fournisseur"] = df["fournisseur"].astype(str).str.strip()
    df["piece"]       = df["piece"].astype(str).str.strip()
    df["rapproch"]    = df.get("rapproch", pd.Series("", index=df.index)).astype(str).str.strip()
    df["texte"]       = df.get("texte",    pd.Series("", index=df.index)).fillna("").astype(str)
    df["dc"]          = df.get("dc",       pd.Series("", index=df.index)).fillna("").astype(str)
    df["type"]        = df.get("type",     pd.Series("", index=df.index)).fillna("").astype(str)
    df["devise"]      = df.get("devise",   pd.Series("", index=df.index)).fillna("").astype(str)
    df["date"]        = pd.to_datetime(df.get("date", pd.Series()), errors="coerce")\
                          .dt.strftime("%Y-%m-%d").fillna("")
    return df.reset_index(drop=True)

def clean(obj):
    if isinstance(obj, dict):  return {k: clean(v) for k,v in obj.items()}
    if isinstance(obj, list):  return [clean(v) for v in obj]
    if isinstance(obj, float) and obj != obj: return None
    try:
        import numpy as np
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
    except Exception: pass
    return obj

@app.route("/")
def index():
    return render_template("index.html",
                           model_trained=is_trained(),
                           model_meta=load_meta(),
                           feature_labels=FEATURES)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400
    f = request.files["file"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": "Format non supporté (.xlsx ou .xls)"}), 400
    fname = secure_filename(f.filename)
    path  = os.path.join(UPLOAD_FOLDER, fname)
    f.save(path)
    try:
        df = load_excel(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    sid = request.remote_addr + "_" + fname
    _sessions[sid] = {"df": df, "source": fname}
    session["sid"] = sid
    fournisseurs = df["fournisseur"].value_counts().reset_index()
    fournisseurs.columns = ["fournisseur","count"]
    return jsonify({
        "ok": True, "source": fname, "total": len(df),
        "fournisseurs": fournisseurs.to_dict("records"),
        "model_trained": is_trained(),
    })

@app.route("/generate-and-train", methods=["POST"])
def generate_and_train():
    try:
        import importlib.util, sys
        gen_path = os.path.join(os.path.dirname(__file__), "engine", "generate_synthetic.py")
        spec = importlib.util.spec_from_file_location("gen_synth", gen_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        return jsonify({"error": f"Génération dataset échouée : {e}"}), 500

    try:
        meta = train_on_synthetic(DATA_FOLDER)
    except Exception as e:
        return jsonify({"error": f"Entraînement échoué : {e}"}), 500

    if "error" in meta:
        return jsonify(meta), 400

    return jsonify({"ok": True, "meta": meta})

@app.route("/run", methods=["POST"])
def run():
    sid = session.get("sid")
    if not sid or sid not in _sessions:
        return jsonify({"error": "Aucun fichier chargé"}), 400
    data   = request.json or {}
    config = data.get("config", {})
    fourn  = data.get("fournisseur", "ALL")
    ml_thr = float(data.get("ml_threshold", 0.60))

    df = _sessions[sid]["df"].copy()
    if fourn != "ALL":
        df = df[df["fournisseur"] == fourn].reset_index(drop=True)
    if df.empty:
        return jsonify({"error": "Aucune écriture pour ce fournisseur"}), 400

    rule_groups, matched = run_rules(df, config)

    ml_groups = []
    if is_trained() and config.get("use_ml", True):
        ml_groups = predict_and_explain(df, matched, threshold=ml_thr)
        for g in ml_groups:
            for idx in g["indices"]: matched.add(idx)

    all_groups = rule_groups + ml_groups
    unmatched  = df[~df.index.isin(matched)].to_dict("records")
    _sessions[sid]["groups"]    = all_groups
    _sessions[sid]["unmatched"] = unmatched

    diagnostic = analyser_non_resolus(df, matched)
    _sessions[sid]["diagnostic"] = diagnostic

    total_matched = sum(len(g["items"]) for g in all_groups)
    rate = total_matched / len(df) * 100 if len(df) else 0

    return jsonify(clean({
        "ok": True, "total": len(df),
        "rule_groups": len(rule_groups), "ml_groups": len(ml_groups),
        "unmatched": len(unmatched), "rate": round(rate,1),
        "groups": all_groups, "unmatched_rows": unmatched,
        "feature_labels": FEATURES,
        "diagnostic": diagnostic,
    }))

@app.route("/diagnostic")
def diagnostic():
    sid = session.get("sid")
    if not sid or sid not in _sessions:
        return jsonify({"error": "Aucun résultat disponible"}), 400
    diag = _sessions[sid].get("diagnostic", {})
    return jsonify(clean(diag))

@app.route("/export")
def export():
    sid = session.get("sid")
    if not sid or sid not in _sessions:
        return jsonify({"error": "Aucun résultat"}), 400
    sess = _sessions[sid]
    data = export_results(sess.get("groups",[]), sess.get("unmatched",[]), sess.get("source",""))
    return send_file(io.BytesIO(data),
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True, download_name="rapprochement_resultats.xlsx")

@app.route("/model-info")
def model_info():
    return jsonify({"trained": is_trained(), "meta": load_meta(), "feature_labels": FEATURES})

@app.route("/synthetic-stats")
def synthetic_stats():
    pairs_path = os.path.join(DATA_FOLDER, "synthetic_pairs.csv")
    ds_path    = os.path.join(DATA_FOLDER, "synthetic_dataset.xlsx")
    if not os.path.exists(pairs_path):
        return jsonify({"available": False})
    df_p = pd.read_csv(pairs_path)
    df_d = pd.read_excel(ds_path)
    return jsonify({
        "available":   True,
        "n_ecritures": len(df_d),
        "n_paires":    len(df_p),
        "n_positif":   int((df_p["label"]==1).sum()),
        "n_negatif":   int((df_p["label"]==0).sum()),
        "cas":         df_p["cas"].value_counts().to_dict(),
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)