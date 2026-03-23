import pandas as pd
import numpy as np


def run_rules(df: pd.DataFrame, config: dict) -> tuple[list, set]:
   
    tol_abs = float(config.get("tolerance_abs", 10))
    tol_pct = float(config.get("tolerance_pct", 0.5)) / 100
    use_r1   = config.get("rule_ref",  True)
    use_r2   = config.get("rule_11",   True)
    use_r3   = config.get("rule_1n",   True)

    groups  = []
    matched = set()
    gid     = [1]

    def new_gid():
        s = f"GRP-{gid[0]:04d}"
        gid[0] += 1
        return s

    def ecart_ok(ecart, ref_amount):
        if abs(ecart) < 0.01:
            return True
        if abs(ecart) <= tol_abs:
            return True
        if ref_amount and abs(ecart) / abs(ref_amount) <= tol_pct:
            return True
        return False

    if use_r1:
        byref = df[df["rapproch"].notna() & (df["rapproch"] != "")]\
                  .groupby("rapproch")
        for ref, grp in byref:
            idxs = grp.index.tolist()
            if len(idxs) < 2:
                continue
            if all(i in matched for i in idxs):
                continue
            rows   = grp.to_dict("records")
            somme  = sum(r["montant"] for r in rows)
            max_m  = max(abs(r["montant"]) for r in rows) or 1
            valid  = ecart_ok(somme, max_m)

            if not valid:
                continue  

            mtype  = "1-1" if len(rows) == 2 else f"1-{len(rows)-1}"
            steps  = _explain_rule1(rows, ref, somme, tol_abs, tol_pct)

            groups.append({
                "id":          new_gid(),
                "fournisseur": rows[0]["fournisseur"],
                "items":       rows,
                "indices":     idxs,
                "somme":       round(somme, 2),
                "ecart":       round(somme, 2),
                "moteur":      "RÈGLE",
                "match_type":  mtype,
                "regle":       "Règle 1 — Référence pivot commune",
                "score":       None,
                "features":    None,
                "valid":       valid,
                "steps":       steps,
            })
            for i in idxs:
                matched.add(i)

    if use_r2:
        unm = df[~df.index.isin(matched) & (df["montant"] != 0)]
        idxs = unm.index.tolist()
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                ia, ib = idxs[i], idxs[j]
                if ia in matched or ib in matched:
                    continue
                ra, rb = df.loc[ia], df.loc[ib]
                somme  = ra["montant"] + rb["montant"]
                if abs(somme) < 0.01 and ra["montant"] != 0:
                    rows  = [ra.to_dict(), rb.to_dict()]
                    steps = _explain_rule2(rows)
                    groups.append({
                        "id":          new_gid(),
                        "fournisseur": ra["fournisseur"],
                        "items":       rows,
                        "indices":     [ia, ib],
                        "somme":       0.0,
                        "ecart":       0.0,
                        "moteur":      "RÈGLE",
                        "match_type":  "1-1",
                        "regle":       "Règle 2 — Matching exact (somme = 0)",
                        "score":       None,
                        "features":    None,
                        "valid":       True,
                        "steps":       steps,
                    })
                    matched.add(ia)
                    matched.add(ib)

    if use_r3:
        unm     = df[~df.index.isin(matched) & (df["montant"] != 0)]
        credits = unm[unm["dc"] == "S"].index.tolist()
        debits  = unm[unm["dc"] == "H"].index.tolist()

        for ic in credits:
            if ic in matched:
                continue
            rc   = df.loc[ic]
            best = None
            best_diff = float("inf")

            for id_ in debits:
                if id_ in matched:
                    continue
                rd   = df.loc[id_]
                diff = abs(rc["montant"] + rd["montant"])
                ref  = max(abs(rc["montant"]), 1)
                if (diff <= tol_abs or diff / ref <= tol_pct) and diff < best_diff:
                    best      = id_
                    best_diff = diff

            if best is not None:
                rd    = df.loc[best]
                rows  = [rc.to_dict(), rd.to_dict()]
                somme = rc["montant"] + rd["montant"]
                steps = _explain_rule3(rows, somme, tol_abs, tol_pct)
                groups.append({
                    "id":          new_gid(),
                    "fournisseur": rc["fournisseur"],
                    "items":       rows,
                    "indices":     [ic, best],
                    "somme":       round(somme, 2),
                    "ecart":       round(somme, 2),
                    "moteur":      "RÈGLE",
                    "match_type":  "1-1",
                    "regle":       f"Règle 3 — Tolérance (écart={abs(somme):.2f})",
                    "score":       None,
                    "features":    None,
                    "valid":       True,
                    "steps":       steps,
                })
                matched.add(ic)
                matched.add(best)

    return groups, matched



def _explain_rule1(rows, ref, somme, tol_abs, tol_pct):
    n = len(rows)
    pieces = ", ".join(r["piece"] for r in rows)
    return [
        f"📌 Référence pivot identifiée : {ref}",
        f"🔗 {n} écritures partagent cette référence : [{pieces}]",
        f"➕ Somme algébrique des montants : {somme:+.2f}",
        f"✅ Validation : {'écart nul → rapprochement parfait' if abs(somme) < 0.01 else f'écart {somme:.2f} dans la tolérance ({tol_abs} abs / {tol_pct*100:.1f}%)'}",
    ]

def _explain_rule2(rows):
    a, b = rows
    return [
        f"🔍 Recherche de paires dont la somme = 0",
        f"📄 Écriture A : {a['piece']} → {a['montant']:+.2f} ({a['dc']})",
        f"📄 Écriture B : {b['piece']} → {b['montant']:+.2f} ({b['dc']})",
        f"➕ Somme : {a['montant']:+.2f} + {b['montant']:+.2f} = {a['montant']+b['montant']:.2f}",
        f"✅ Somme exactement nulle → rapprochement 1-1 parfait",
    ]

def _explain_rule3(rows, somme, tol_abs, tol_pct):
    a, b = rows
    ref = max(abs(a["montant"]), 1)
    pct_ecart = abs(somme) / ref * 100
    return [
        f"🔍 Recherche de la meilleure paire débit/crédit avec tolérance",
        f"📄 Crédit : {a['piece']} → {a['montant']:+.2f}",
        f"📄 Débit  : {b['piece']} → {b['montant']:+.2f}",
        f"➕ Écart résiduel : {abs(somme):.2f} ({pct_ecart:.3f}%)",
        f"✅ Écart ≤ seuil ({tol_abs} XOF / {tol_pct*100:.1f}%) → accepté",
    ]
