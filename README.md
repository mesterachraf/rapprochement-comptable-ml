```markdown
Rapprochement Comptable Intelligent — Flask + ML
Application web pour le rapprochement comptable automatisé combinant règles métier et Machine Learning.

Structure du projet
rapprochement/
├── app.py                  # Serveur Flask (point d'entrée)
├── requirements.txt        # Dépendances Python
├── engine/
│   ├── rules.py            # Moteur de règles
│   ├── ml_model.py         # Modèle Machine Learning
│   └── exporter.py         # Export Excel des résultats
├── diagnostic.py           # Analyse des cas non résolus
├── templates/
│   └── index.html          # Interface utilisateur
├── static/
│   ├── style.css           # CSS
│   └── main.js             # JS
├── uploads/                # Fichiers importés (auto)
└── models/                 # Modèles ML sauvegardés (auto)

````

---

## Installation

1. **Cloner le projet**
```bash
git clone https://github.com/ton-user/rapprochement-comptable-ml.git
cd rapprochement-comptable-ml
````

2. **Créer un environnement virtuel**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Lancer l'application**

```bash
python app.py
```

Puis ouvrir dans le navigateur : [http://localhost:5000](http://localhost:5000)

---

## Utilisation

1. Importer un fichier Excel (drag & drop ou parcourir)
2. (Optionnel) Entraîner le modèle ML
3. Filtrer par fournisseur si nécessaire
4. Configurer les règles et tolérances
5. Lancer le rapprochement
6. Analyser les résultats :

   * Groupes rapprochés
   * Écritures non rapprochées
   * Diagnostic intelligent
7. Exporter les résultats en Excel

---

## Format du fichier Excel attendu

| Colonne             | Requis | Description                 |
| ------------------- | ------ | --------------------------- |
| Numéro de pièce     | ✅      | Identifiant unique          |
| Fournisseur         | ✅      | Code fournisseur            |
| Montant             | ✅      | Montant positif/négatif     |
| Code débit/crédit   | —      | H = Débit / S = Crédit      |
| Date comptable      | —      | Date de l'écriture          |
| Texte               | —      | Libellé                     |
| Type de pièce       | —      | SK, KR, RE, etc.            |
| Devise              | —      | EUR, XOF, USD…              |
| Pièce rapprochement | —      | Référence ML (ground truth) |

---

## Moteur de règles

* **Règle 1** : regroupement par référence pivot
* **Règle 2** : matching exact 1-1 (somme = 0)
* **Règle 3** : matching avec tolérance configurable

---

## Machine Learning

* **Algorithme** : Random Forest (200 arbres)
* **Features** : montant, texte, référence, date, D/C opposés, type, devise, magnitude
* **Output** : score 0-100% + explication
* **Seuil** : configurable (défaut 65%)

---

## Diagnostic

Analyse automatique des écritures non rapprochées :

* ORPHELIN : écriture sans contrepartie
* MONTANT_ZERO : montant nul
* MULTI_DEVISE : plusieurs devises
* GROUPE_PARTIEL : sous-groupe équilibré mais résidu
* ECART_STRUCT : débits ou crédits uniquement
* ECART_ML : écart résiduel selon ML

Chaque catégorie indique : cause, solution, exemples.

---

## Export Excel

4 onglets :

1. **Résumé** : statistiques globales
2. **Groupes rapprochés** : tous les groupes avec statut
3. **Non rapprochées** : écritures sans correspondance
4. **Détail écritures** : chaque écriture avec justification

---

## Technologies

* Python, Flask
* Pandas, NumPy
* Scikit-learn
* HTML, CSS, JavaScript


