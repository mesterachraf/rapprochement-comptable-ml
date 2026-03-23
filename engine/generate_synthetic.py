
import pandas as pd
import numpy as np
import random
import string
import os
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
random.seed(42)

N_CASE1 = 300  
N_CASE2 = 300   
N_CASE3 = 300   
N_CASE4 = 200   
N_NEG   = 600   

FOURNISSEURS = ["40000037","45183","45245","45301","45841","2","99001","99002"]
TYPES_CREDIT = ["SK","KZ","SA","KR","KA"]
TYPES_DEBIT  = ["RE","LT","KR","KZ"]
DEVISES      = ["XOF","XOF","XOF","XOF","EUR"]  

LIBELLES_DOUANE   = ["DOUANE/SACO{yr}FC{n}","DROITS DOUANE FACTURE {n}","TAXES IMPORT SACO{yr}FC{n}","DEDOUANEMENT SACO {n}"]
LIBELLES_TRANSIT  = ["TRANSIT ET TRANSPORT/SACO{yr}FC{n}","FRAIS TRANSIT FACTURE {n}","TRANSPORT MARCHANDISES {n}"]
LIBELLES_PAIEMENT = ["CHQ BIA N° {n} Fav./ {soc}","VIREMENT {soc} FACTURE {n}","REGLEMENT FACTURE {n} {soc}"]
LIBELLES_ACHAT    = ["ACHAT {prod} FACTURE {n}","LIVRAISON {prod} BON {n}","RECEPTION {prod} REF {n}"]
SOCIETES          = ["SEGUCE","SOBEBRA","CASTEL","CEET","GRAS SAVOYE","TOTAL"]
PRODUITS          = ["MATERIAUX","FOURNITURES BUREAU","CARBURANT","EQUIPEMENTS","PIECES DETACHEES"]

def rand_ref():
    return f"45{RNG.integers(100000,999999)}"

def rand_piece(prefix="60"):
    return f"{prefix}{RNG.integers(10000000,99999999)}"

def rand_date(start="2023-01-01", end="2025-01-01"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    return (s + timedelta(days=int(RNG.integers(0,(e-s).days)))).strftime("%Y-%m-%d")

def rand_libelle(templates, **kw):
    t = random.choice(templates)
    yr  = kw.get("yr",  str(RNG.integers(22,25)))
    n   = kw.get("n",   str(RNG.integers(10000,99999)))
    soc = kw.get("soc", random.choice(SOCIETES))
    prod= kw.get("prod",random.choice(PRODUITS))
    return t.format(yr=yr, n=n, soc=soc, prod=prod)

def perturb_libelle(libelle: str) -> str:
    """Génère une version similaire mais pas identique du libellé."""
    ops = [
        lambda s: s.lower(),
        lambda s: s.upper(),
        lambda s: s.replace("/", " "),
        lambda s: s + " (AVOIR)",
        lambda s: "RGL " + s,
        lambda s: s.replace("FACTURE","FACT").replace("TRANSPORT","TRANS"),
        lambda s: " ".join(s.split()[:max(1,len(s.split())-1)]), 
    ]
    return random.choice(ops)(libelle)

def perturb_ref(ref: str) -> str:
    """Référence partiellement similaire — même préfixe, fin différente."""
    ops = [
        lambda r: r[:-2] + str(RNG.integers(10,99)),
        lambda r: r[:4] + str(RNG.integers(100000,999999)),
        lambda r: "REF-" + r[-6:],
        lambda r: r + "-B",
    ]
    return random.choice(ops)(ref)

rows = []
pairs = []   
row_id = [1]

def new_id():
    i = row_id[0]; row_id[0] += 1; return i

def add_row(piece, fournisseur, montant, dc, date, texte, type_piece, devise, rapproch, groupe_id, source="ERP"):
    rid = new_id()
    rows.append({
        "id":             rid,
        "piece":          piece,
        "fournisseur":    fournisseur,
        "montant":        round(float(montant), 2),
        "dc":             dc,
        "date":           date,
        "texte":          texte,
        "type":           type_piece,
        "devise":         devise,
        "rapproch":       rapproch,
        "groupe_id":      groupe_id,   
        "source":         source,
        "label_matched":  1,          
    })
    return rid


print("Génération CAS 1 — Tolérance montants...")
for i in range(N_CASE1):
    fourn   = random.choice(FOURNISSEURS)
    base    = float(RNG.integers(50000, 5000000))
    tol_pct = RNG.uniform(0.001, 0.008)           
    ecart   = base * tol_pct * RNG.choice([-1,1])
    ref     = rand_ref()
    date_a  = rand_date()
    date_b  = (datetime.strptime(date_a,"%Y-%m-%d") + timedelta(days=int(RNG.integers(1,15)))).strftime("%Y-%m-%d")
    lib     = rand_libelle(LIBELLES_PAIEMENT)
    grp     = f"SYNTH-TOL-{i:04d}"
    devise  = random.choice(DEVISES)

    ia = add_row(rand_piece("56"), fourn,  base,         "S", date_a, lib,                   random.choice(TYPES_CREDIT), devise, ref, grp, "ERP")
    ib = add_row(rand_piece("60"), fourn, -(base+ecart), "H", date_b, perturb_libelle(lib),  random.choice(TYPES_DEBIT),  devise, ref, grp, "BANK")
    pairs.append({"id_a":ia,"id_b":ib,"label":1,"cas":"CAS1_TOLERANCE","ecart":round(abs(ecart),2)})


print("Génération CAS 2 — Références partielles multi-sources...")
for i in range(N_CASE2):
    fourn   = random.choice(FOURNISSEURS)
    montant = float(RNG.integers(100000, 10000000))
    ref_erp = rand_ref()
    ref_bank= perturb_ref(ref_erp)
    date_a  = rand_date()
    date_b  = (datetime.strptime(date_a,"%Y-%m-%d") + timedelta(days=int(RNG.integers(0,7)))).strftime("%Y-%m-%d")
    lib     = rand_libelle(LIBELLES_DOUANE)
    grp     = f"SYNTH-REF-{i:04d}"
    devise  = "XOF"

    ia = add_row(rand_piece("50"), fourn,  montant, "S", date_a, lib,                  random.choice(TYPES_CREDIT), devise, ref_erp,  grp, "ERP")
    ib = add_row(rand_piece("60"), fourn, -montant, "H", date_b, perturb_libelle(lib), random.choice(TYPES_DEBIT),  devise, ref_bank, grp, "BANK")
    pairs.append({"id_a":ia,"id_b":ib,"label":1,"cas":"CAS2_REF_PARTIELLE","ecart":0.0})


print("Génération CAS 3 — Libellés similaires...")
for i in range(N_CASE3):
    fourn   = random.choice(FOURNISSEURS)
    montant = float(RNG.integers(200000, 8000000))
    date_a  = rand_date()
    date_b  = (datetime.strptime(date_a,"%Y-%m-%d") + timedelta(days=int(RNG.integers(0,5)))).strftime("%Y-%m-%d")
    lib_base= rand_libelle(LIBELLES_TRANSIT)
    lib_b   = perturb_libelle(lib_base)
    ref_a   = rand_ref()
    ref_b   = rand_ref()   
    grp     = f"SYNTH-TXT-{i:04d}"
    devise  = random.choice(DEVISES)

    ia = add_row(rand_piece("10"), fourn,  montant, "S", date_a, lib_base, random.choice(TYPES_CREDIT), devise, ref_a, grp, "ERP")
    ib = add_row(rand_piece("60"), fourn, -montant, "H", date_b, lib_b,    random.choice(TYPES_DEBIT),  devise, ref_b, grp, "FACTURE")
    pairs.append({"id_a":ia,"id_b":ib,"label":1,"cas":"CAS3_LIBELLE_SIM","ecart":0.0})


print("Génération CAS 4 — Paiements partiels 1-N...")
for i in range(N_CASE4):
    fourn   = random.choice(FOURNISSEURS)
    total   = float(RNG.integers(500000, 20000000))
    n_parts = int(RNG.integers(2, 6))
   
    cuts    = sorted(RNG.uniform(0,1,n_parts-1).tolist())
    cuts    = [0] + cuts + [1]
    parts   = [round(total*(cuts[k+1]-cuts[k]),0) for k in range(n_parts)]
    parts[-1] = total - sum(parts[:-1])  

    ref_fact = rand_ref()
    date_fact= rand_date()
    lib_fact = rand_libelle(LIBELLES_ACHAT)
    grp      = f"SYNTH-1N-{i:04d}"
    devise   = "XOF"

    id_fact = add_row(rand_piece("10"), fourn, total, "S", date_fact, lib_fact, "SA", devise, ref_fact, grp, "ERP")
    part_ids = [id_fact]

    for k, part in enumerate(parts):
        d = (datetime.strptime(date_fact,"%Y-%m-%d") + timedelta(days=int(RNG.integers(1,30)))).strftime("%Y-%m-%d")
        ref_pay = ref_fact if RNG.random() > 0.4 else perturb_ref(ref_fact)
        lib_pay = rand_libelle(LIBELLES_PAIEMENT)
        pid = add_row(rand_piece("60"), fourn, -part, "H", d, lib_pay, random.choice(TYPES_DEBIT), devise, ref_pay, grp, "BANK")
        part_ids.append(pid)

    for pid in part_ids[1:]:
        pairs.append({"id_a":id_fact,"id_b":pid,"label":1,"cas":"CAS4_PARTIEL_1N","ecart":0.0})


print("Génération CAS 5 — Cas négatifs...")
for i in range(N_NEG):
    fourn  = random.choice(FOURNISSEURS)
    m1     = float(RNG.integers(100000, 5000000))
    m2     = m1 * RNG.uniform(0.85, 0.95)
    date_a = rand_date("2023-01-01","2024-06-01")
    date_b = rand_date("2024-07-01","2025-01-01") 
    ref_a  = rand_ref()
    ref_b  = rand_ref()  
    lib_a  = rand_libelle(LIBELLES_DOUANE)
    lib_b  = rand_libelle(LIBELLES_ACHAT)  
    grp_a  = f"SYNTH-NEG-A-{i:04d}"
    grp_b  = f"SYNTH-NEG-B-{i:04d}"

    ia = add_row(rand_piece("60"), fourn,  m1, "S", date_a, lib_a, random.choice(TYPES_CREDIT), "XOF", ref_a, grp_a, "ERP")
    ib = add_row(rand_piece("60"), fourn, -m2, "H", date_b, lib_b, random.choice(TYPES_DEBIT),  "XOF", ref_b, grp_b, "ERP")
    rows[-2]["label_matched"] = 0
    rows[-1]["label_matched"] = 0
    pairs.append({"id_a":ia,"id_b":ib,"label":0,"cas":"CAS5_NEGATIF","ecart":round(abs(m1-m2),2)})


df_rows  = pd.DataFrame(rows)
df_pairs = pd.DataFrame(pairs)

print("\n=== DATASET SYNTHÉTIQUE GÉNÉRÉ ===")
print(f"Total écritures   : {len(df_rows)}")
print(f"Total paires      : {len(df_pairs)}")
print(f"  Positives (1)   : {(df_pairs['label']==1).sum()}")
print(f"  Négatives (0)   : {(df_pairs['label']==0).sum()}")
print(f"\nRépartition par cas :")
print(df_pairs['cas'].value_counts().to_string())

out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(out_dir, exist_ok=True)

df_rows.to_excel(os.path.join(out_dir,"synthetic_dataset.xlsx"), index=False)
df_pairs.to_csv( os.path.join(out_dir,"synthetic_pairs.csv"),    index=False)

print(f"\n✅ Fichiers exportés dans /data/")
print(f"   synthetic_dataset.xlsx ({len(df_rows)} écritures)")
print(f"   synthetic_pairs.csv    ({len(df_pairs)} paires labelisées)")
