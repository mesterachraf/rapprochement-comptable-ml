import streamlit as st
import pandas as pd
import os, io
from engine.rules       import run_rules
from engine.diagnostic  import analyser_non_resolus
from engine.ml_model_v2 import (
    train_on_synthetic, predict_and_explain,
    is_trained, load_meta, FEATURES
)
from engine.exporter    import export_results

st.set_page_config(page_title="Rapprochement ML", layout="wide")
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_FOLDER, exist_ok=True)

def load_excel_st(file):
    df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]
    col_map = {
        "Numéro de pièce": "piece", "Fournisseur": "fournisseur",
        "Montant en devise de la transaction": "montant",
        "Code débit/crédit": "dc", "Date comptable": "date",
        "Texte": "texte", "Type de pièce": "type",
        "Devise de transaction": "devise", "Pièce rapprochement": "rapproch",
    }
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
    df["montant"] = pd.to_numeric(df["montant"], errors="coerce").fillna(0)
    for col in ["fournisseur", "piece", "rapproch", "texte", "dc", "type", "devise"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df.reset_index(drop=True)

st.title("⚖️ Rapprochement Comptable ML")

if is_trained():
    meta = load_meta()
    st.sidebar.success(f"✅ Modèle prêt ({meta.get('date','')})")
else:
    st.sidebar.warning("⚠️ Modèle non entraîné")
    if st.sidebar.button("Entraîner le modèle"):
        with st.spinner("Entraînement..."):
            train_on_synthetic(DATA_FOLDER)
            st.rerun()

uploaded_file = st.file_uploader("Fichier Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = load_excel_st(uploaded_file)
    st.write(f"📊 Données : {len(df)} lignes")
    
    fournisseurs = ["ALL"] + sorted(df["fournisseur"].unique().tolist())
    selected_fourn = st.selectbox("Fournisseur", fournisseurs)
    ml_threshold = st.slider("Seuil ML", 0.0, 1.0, 0.6)
    
    if st.button("Lancer l'analyse"):
        df_run = df.copy()
        if selected_fourn != "ALL":
            df_run = df_run[df_run["fournisseur"] == selected_fourn].reset_index(drop=True)
            
        with st.spinner("Analyse..."):
            rule_groups, matched = run_rules(df_run, {"use_ml": True})
            
            ml_groups = []
            if is_trained():
                ml_groups = predict_and_explain(df_run, matched, threshold=ml_threshold)
                for g in ml_groups:
                    for idx in g["indices"]: matched.add(idx)
            
            all_groups = rule_groups + ml_groups
            unmatched = df_run[~df_run.index.isin(matched)]
            
            st.success(f"Taux : {len(matched)/len(df_run)*100:.1f}%")
            
            c1, c2 = st.columns(2)
            c1.metric("Règles", len(rule_groups))
            c2.metric("ML", len(ml_groups))
            
            st.subheader("Écritures non résolues")
            st.dataframe(unmatched)
            
            data_export = export_results(all_groups, unmatched.to_dict("records"), uploaded_file.name)
            st.download_button(
                label="📥 Télécharger l'Excel",
                data=data_export,
                file_name="resultats.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )