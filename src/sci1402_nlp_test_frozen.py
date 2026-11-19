# SCI1402 — Étape 19 : prédictions figées sur le jeu de test
# Exécuter APRÈS les étapes précédentes du notebook.
# Aucun accès aux annotations humaines de test.
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import pandas as pd

ROOT = Path('/kaggle/working')
TEST_FILE = ROOT / 'annotation_test.csv'
PRED_FILE = ROOT / 'nlp_test_predictions_v2_frozen.csv'
MANIFEST_FILE = ROOT / 'nlp_test_v2_manifest.json'

CATEGORIES = [
    'Method Firearm', 'Method Knife', 'Method Strangulation',
    'Method Poisoning', 'Method Blunt Force', 'Method Other',
    'Victims Women', 'Victims Men', 'Victims Children',
    'Victims Elderly', 'Victims Sex Workers',
    'Sexual Violence Mentioned', 'Robbery Mentioned',
    'Dismemberment Mentioned', 'Accomplices Mentioned'
]

if not TEST_FILE.exists():
    raise FileNotFoundError(f'Fichier manquant : {TEST_FILE}. Exécute l’étape 14 du notebook.')
if PRED_FILE.exists() or MANIFEST_FILE.exists():
    raise FileExistsError('Une version figée existe déjà. Ne pas écraser les prédictions de test.')

test = pd.read_csv(TEST_FILE, encoding='utf-8-sig', dtype={'Annotation ID': str})
required = {'Annotation ID', 'Name', 'Cluster', 'Notes'}
missing = required - set(test.columns)
if missing:
    raise ValueError(f'Colonnes manquantes : {sorted(missing)}')
if len(test) != 40 or test['Annotation ID'].nunique() != 40:
    raise ValueError('Le jeu de test doit contenir 40 identifiants uniques.')
if test['Notes'].isna().any():
    raise ValueError('Notes manquantes dans le jeu de test.')
if test['Cluster'].value_counts().to_dict() != {0: 10, 1: 10, 2: 10, 3: 10}:
    raise ValueError('Répartition attendue : 10 dossiers par cluster.')
if any('Human Label' in c or 'Review Status' in c for c in test.columns):
    raise ValueError('Le fichier de test ne doit pas contenir les annotations humaines.')

# Règles générales, fondées sur des mentions dans le texte (pas sur les noms).
RULES = {
 'Method Firearm': [r'\b(?:shot|shooting|shootings|gun|guns|gunshots?|firearms?|rifles?|revolvers?|pistols?)\b'],
 'Method Knife': [r'\b(?:stab(?:bed|bing|s)?|knives|knife|slash(?:ed|ing)?|axes?|hatchets?)\b'],
 'Method Strangulation': [r'\b(?:strangl(?:e|ed|ing|ation)|chok(?:e|ed|ing)|suffocat(?:e|ed|ion))\b'],
 'Method Poisoning': [r'\b(?:poison(?:ed|ing|er)?|arsenic|cyanide|overdos(?:e|es)|epinephrine)\b'],
 'Method Blunt Force': [r'\b(?:bludgeon(?:ed|ing)?|beaten|beat(?:ing)?\s+to\s+death|blunt\s+force|hammer|clubbed|rocks?|iron\s+bar)\b'],
 'Method Other': [r'\b(?:drown(?:ed|ing)|burn(?:ed|t)\s+alive|smother(?:ed|ing)|starv(?:e|ed|ation|ing)|fasting\s+until\s+they\s+starved)\b'],
 'Victims Women': [r'\b(?:women|woman|girls?|waitresses|female\s+(?:victims?|sex\s+workers?|patients?))\b'],
 'Victims Men': [r'\b(?:men|man|boys?|male\s+(?:victims?|patients?)|husbands?|sons?|grandfathers?|grandparents?)\b'],
 'Victims Children': [r'\b(?:children|child|bab(?:y|ies)|infants?|boys?|girls?)\b'],
 'Victims Elderly': [r'\b(?:elderly|seniors|old\s+age\s+pensioners)\b'],
 'Victims Sex Workers': [r'\b(?:sex\s+workers?|prostitutes?|sex\s+trade\s+workers?)\b'],
 'Sexual Violence Mentioned': [r'\b(?:rap(?:e|ed|ing|ist)|sexual\s+assaults?|sexual\s+violence|sexual\s+abuse|sodomy|extort\s+money\s+and\s+sexual\s+favors)\b'],
 'Robbery Mentioned': [r'\b(?:robbery|robberies|robbed|burglar(?:y|ies)|burglar|stole|stealing)\b'],
 'Dismemberment Mentioned': [r'\b(?:dismember(?:ed|ment)?|mutilat(?:ed|ion))\b'],
 'Accomplices Mentioned': [r'\b(?:accomplices?|partner\s+in\s+crime|with\s+the\s+help\s+of|co-conspirators?)\b',
                           r'\b(?:\d+|four|fifteen)\s+(?:people|persons|men|members|african-americans)\b.{0,100}\b(?:participat(?:ed|ion)|killed|murders?)\b']
}
REGEX = {k: [re.compile(p, re.I) for p in patterns] for k, patterns in RULES.items()}

# Les surnoms sont ignorés : « Ripper » et « Strangler » ne sont pas des preuves.
# Le texte hors guillemets reste analysé, même lorsque le surnom est cité.
def strip_aliases(s):
    return re.sub(r'\b(?:known as|referred to as|nicknamed)\s+["“][^"”]+["”]', '', s, flags=re.I)

def detect(text, category):
    text = strip_aliases(str(text) if pd.notna(text) else '')
    matches = [m for pattern in REGEX[category] for m in pattern.finditer(text)]
    if not matches:
        return 0

    # L'arme ayant tué l'auteur n'est pas une méthode utilisée sur ses victimes.
    if category == 'Method Firearm':
        matches = [m for m in matches if not re.search(
            r'\b(?:he|she|the\s+killer|the\s+murderer)\s+was\s+shot\s+dead\b',
            text[max(0, m.start()-15):min(len(text), m.end()+75)], re.I)]
        if not matches:
            return 0

    # Un homme identifié comme auteur n'est pas automatiquement une victime.
    if category == 'Victims Men':
        matches = [m for m in matches if not (
            m.group().lower() == 'man' and
            re.search(r'\b(?:romanian|homeless|american|african)\s+$',
                      text[max(0, m.start()-30):m.start()], re.I)
        )]
        if not matches:
            return 0

    # « at age 15 ... juvenile » décrit l'âge de Kemper, pas celui des victimes.
    if category == 'Victims Children':
        matches = [m for m in matches if not (
            m.group().lower() == 'juvenile' and
            re.search(r'\b(?:he|she)\s+(?:was|served)\b', text[max(0,m.start()-80):m.start()], re.I)
        )]
        if not matches:
            return 0

    # Le travail du sexe d'une dirigeante n'est pas celui des victimes.
    if category == 'Victims Sex Workers':
        matches = [m for m in matches if not re.search(
            r'\b(?:took\s+over\s+as\s+high\s+priestess|became\s+high\s+priestess)',
            text[m.end():m.end()+95], re.I)]
        if not matches:
            return 0

    return 1


# Prédictions indépendantes : aucune métrique ni correction basée sur le test.
rows = []
for case in test.itertuples(index=False):
    for category in CATEGORIES:
        rows.append({
            'Annotation ID': None,
            'Name': case.Name,
            'Cluster': case.Cluster,
            'Category': category,
            'Notes': case.Notes,
            'Predicted V2': detect(case.Notes, category)
        })
# Correction de l'accès à Annotation ID (nom de colonne avec espace).
ids = test['Annotation ID'].tolist()
for i, row in enumerate(rows):
    row['Annotation ID'] = ids[i // len(CATEGORIES)]

predictions = pd.DataFrame(rows)
assert len(predictions) == 600
assert not predictions[['Annotation ID', 'Category']].duplicated().any()
assert predictions['Predicted V2'].isin([0, 1]).all()

# Empreinte des règles et des prédictions pour tracer la version évaluée.
rule_bytes = json.dumps(RULES, sort_keys=True, ensure_ascii=False).encode('utf-8')
rule_sha256 = hashlib.sha256(rule_bytes).hexdigest()
output_bytes = predictions.to_csv(index=False).encode('utf-8-sig')
output_sha256 = hashlib.sha256(output_bytes).hexdigest()

# Création exclusive : ne remplace pas un fichier déjà figé.
with PRED_FILE.open('xb') as handle:
    handle.write(output_bytes)
manifest = {
    'project': 'SCI1402',
    'version': 'NLP V2 frozen',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'test_cases': 40,
    'categories': len(CATEGORIES),
    'predictions': len(predictions),
    'rules_sha256': rule_sha256,
    'predictions_sha256': output_sha256,
    'test_source': str(TEST_FILE),
    'prediction_file': str(PRED_FILE),
    'note': 'Prédictions figées avant annotation humaine du test. Ne pas ajuster les règles sur ce jeu.'
}
with MANIFEST_FILE.open('x', encoding='utf-8') as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)

print(f'Dossiers de test : {len(test)} | Prédictions figées : {len(predictions)}')
print(f'Règles SHA256 : {rule_sha256}')
print(f'Prédictions SHA256 : {output_sha256}')
print(f'Fichier : {PRED_FILE}')
print(f'Manifeste : {MANIFEST_FILE}')
print('Aucune métrique calculée : annotations humaines de test non utilisées.')
