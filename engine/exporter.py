

import io
import pandas as pd
from datetime import datetime


def export_results(groups: list, unmatched: list, source_name: str = "") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _write_summary(writer, groups, unmatched, source_name)
        _write_groups(writer, groups)
        _write_unmatched(writer, unmatched)
        _write_details(writer, groups)
    return buf.getvalue()


def _write_summary(writer, groups, unmatched, source_name):
    rule_g = [g for g in groups if g["moteur"] == "RÈGLE"]
    ml_g   = [g for g in groups if g["moteur"] == "ML"]
    total  = sum(len(g["items"]) for g in groups) + len(unmatched)
    matched_count = sum(len(g["items"]) for g in groups)
    rate   = matched_count / total * 100 if total else 0

    data = {
        "Indicateur": [
            "Fichier source", "Date d'exécution", "Total écritures",
            "Écritures rapprochées", "Non rapprochées", "Taux de rapprochement",
            "Groupes (Moteur Règles)", "Groupes (ML)", "Total groupes",
        ],
        "Valeur": [
            source_name, datetime.now().strftime("%d/%m/%Y %H:%M"),
            total, matched_count, len(unmatched),
            f"{rate:.1f}%", len(rule_g), len(ml_g), len(groups),
        ]
    }
    df = pd.DataFrame(data)
    df.to_excel(writer, sheet_name="Résumé", index=False)


def _write_groups(writer, groups):
    rows = []
    for g in groups:
        rows.append({
            "ID Groupe":        g["id"],
            "Fournisseur":      g["fournisseur"],
            "Moteur":           g["moteur"],
            "Type matching":    g["match_type"],
            "Nb écritures":     len(g["items"]),
            "Règle/Score ML":   g["regle"],
            "Somme":            g["somme"],
            "Écart résiduel":   g["ecart"],
            "Statut":           "ÉQUILIBRÉ" if g["valid"] else "ÉCART",
        })
    pd.DataFrame(rows).to_excel(writer, sheet_name="Groupes rapprochés", index=False)


def _write_unmatched(writer, unmatched):
    cols = ["piece","fournisseur","date","montant","dc","devise","texte","type","rapproch"]
    rows = [{c: r.get(c,"") for c in cols} for r in unmatched]
    pd.DataFrame(rows).to_excel(writer, sheet_name="Non rapprochées", index=False)


def _write_details(writer, groups):
    rows = []
    for g in groups:
        for item in g["items"]:
            rows.append({
                "ID Groupe":    g["id"],
                "Moteur":       g["moteur"],
                "Score ML":     g["score"] if g["score"] else "",
                "N° Pièce":     item.get("piece",""),
                "Fournisseur":  item.get("fournisseur",""),
                "Date":         item.get("date",""),
                "Montant":      item.get("montant",""),
                "D/C":          item.get("dc",""),
                "Devise":       item.get("devise",""),
                "Texte":        item.get("texte",""),
                "Type":         item.get("type",""),
                "Rapprochement":item.get("rapproch",""),
                "Justification":"\n".join(g.get("steps",[])),
            })
    pd.DataFrame(rows).to_excel(writer, sheet_name="Détail écritures", index=False)
