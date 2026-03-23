import pandas as pd
import numpy as np

def analyser_non_resolus(df: pd.DataFrame, matched: set) -> dict:
    if df is None or len(df) == 0:
        return {
            'total_all': 0,
            'total_resolus': 0,
            'total_non_resolus': 0,
            'taux_resolus': 0.0,
            'taux_non_resolus': 0.0,
            'categories': []
        }

    df_unm = df[~df.index.isin(matched)].copy()
    total_all = len(df)
    total_unm = len(df_unm)

    if total_unm == 0:
        return {
            'total_all': total_all,
            'total_resolus': total_all,
            'total_non_resolus': 0,
            'taux_resolus': 100.0,
            'taux_non_resolus': 0.0,
            'categories': []
        }

    df_unm['rapproch'] = df_unm.get('rapproch', pd.Series('', index=df_unm.index)).fillna('').astype(str).str.strip()
    df_unm['montant']  = pd.to_numeric(df_unm.get('montant', 0), errors='coerce').fillna(0)
    df_unm['devise']   = df_unm.get('devise', pd.Series('', index=df_unm.index)).fillna('').astype(str)
    df_unm['texte']    = df_unm.get('texte',  pd.Series('', index=df_unm.index)).fillna('').astype(str)
    df_unm['type']     = df_unm.get('type',   pd.Series('', index=df_unm.index)).fillna('').astype(str)
    df_unm['dc']       = df_unm.get('dc',     pd.Series('', index=df_unm.index)).fillna('').astype(str)
    df_unm['piece']    = df_unm.get('piece',  pd.Series('', index=df_unm.index)).astype(str)

    grouped     = df_unm.groupby('rapproch')
    sizes       = grouped.size()
    sommes      = grouped['montant'].sum()
    devises_grp = grouped['devise'].apply(lambda x: x.nunique())

    categories = {
        'ORPHELIN':      {'label': 'Écriture orpheline', 'color': 'orange', 'groupes': [], 'ecritures': 0},
        'MONTANT_ZERO':  {'label': 'Montant zéro (saisie incomplète)', 'color': 'red', 'groupes': [], 'ecritures': 0},
        'MULTI_DEVISE':  {'label': 'Multi-devises (EUR/XOF/USD)', 'color': 'purple', 'groupes': [], 'ecritures': 0},
        'GROUPE_PARTIEL':{'label': 'Groupe partiellement rapproché', 'color': 'yellow', 'groupes': [], 'ecritures': 0},
        'ECART_STRUCT':  {'label': 'Écart structurel', 'color': 'red', 'groupes': [], 'ecritures': 0},
        'ECART_ML':      {'label': 'Écart résiduel ML', 'color': 'blue', 'groupes': [], 'ecritures': 0},
    }

    causes = {
        'ORPHELIN': "Contrepartie absente",
        'MONTANT_ZERO': "Montant nul dans SAP",
        'MULTI_DEVISE': "Devises mixtes",
        'GROUPE_PARTIEL': "Sous-groupe équilibré mais reste",
        'ECART_STRUCT': "Même sens débit/crédit",
        'ECART_ML': "Écart faible mais hors tolérance",
    }

    solutions = {
        'ORPHELIN': "Attendre ou vérifier la contrepartie",
        'MONTANT_ZERO': "Corriger le montant",
        'MULTI_DEVISE': "Ajouter conversion devise",
        'GROUPE_PARTIEL': "Identifier paiement manquant",
        'ECART_STRUCT': "Chercher contrepartie",
        'ECART_ML': "Contrôle manuel",
    }

    for ref, grp_data in grouped:
        try:
            somme  = float(sommes[ref])
            n      = int(sizes[ref])
            n_dev  = int(devises_grp[ref])
            monts  = grp_data['montant'].tolist()
            max_m  = grp_data['montant'].abs().max()
            pct    = abs(somme) / max_m * 100 if max_m > 0 else 0

            ecritures_detail = [
                {
                    'piece': str(r.get('piece', '')),
                    'montant': float(r.get('montant', 0)),
                    'devise': str(r.get('devise', '')),
                    'dc': str(r.get('dc', '')),
                    'type': str(r.get('type', '')),
                    'texte': str(r.get('texte', ''))[:60],
                }
                for _, r in grp_data.iterrows()
            ]

            groupe_info = {
                'ref': ref,
                'n': n,
                'somme': round(somme, 2),
                'pct_ecart': round(pct, 2),
                'devises': grp_data['devise'].value_counts().to_dict(),
                'ecritures': ecritures_detail,
            }

            if n == 1:
                cat = 'ORPHELIN'
            elif (grp_data['montant'] == 0).any():
                cat = 'MONTANT_ZERO'
            elif n_dev > 1:
                cat = 'MULTI_DEVISE'
            else:
                sous_groupe_ok = any(
                    abs(monts[i] + monts[j]) < 1
                    for i in range(len(monts))
                    for j in range(i+1, len(monts))
                )

                all_debit  = all(m <= 0 for m in monts)
                all_credit = all(m >= 0 for m in monts)

                if sous_groupe_ok:
                    cat = 'GROUPE_PARTIEL'
                elif all_debit or all_credit:
                    cat = 'ECART_STRUCT'
                elif pct < 5.0:
                    cat = 'ECART_ML'
                else:
                    cat = 'ECART_STRUCT'

            categories[cat]['groupes'].append(groupe_info)
            categories[cat]['ecritures'] += n

        except Exception:
            continue

    result_cats = []
    for cat_key, cat_data in categories.items():
        if not cat_data['groupes']:
            continue

        n_ecr = cat_data['ecritures']

        result_cats.append({
            'id': cat_key,
            'label': cat_data['label'],
            'color': cat_data['color'],
            'cause': causes[cat_key],
            'solution': solutions[cat_key],
            'n_groupes': len(cat_data['groupes']),
            'n_ecritures': n_ecr,
            'pct_total': round(n_ecr / total_all * 100, 1),
            'pct_nonres': round(n_ecr / total_unm * 100, 1),
            'groupes': cat_data['groupes'],
        })

    result_cats.sort(key=lambda x: -x['n_ecritures'])

    return {
        'total_all': total_all,
        'total_resolus': total_all - total_unm,
        'total_non_resolus': total_unm,
        'taux_resolus': round((total_all - total_unm) / total_all * 100, 1),
        'taux_non_resolus': round(total_unm / total_all * 100, 1),
        'categories': result_cats,
    }