# SCI1402 — Analyse des homicides et extraction NLP de caractéristiques criminologiques

Projet universitaire en science des données (TÉLUQ, cours SCI1402).

## Présentation

Ce projet combine une exploration statistique de données sur les homicides et une étude exploratoire de profils de tueurs en série. Il examine les distributions et associations entre variables, utilise un regroupement non supervisé (K-Means), puis développe un système d'extraction automatique de **mentions explicites** dans des descriptions textuelles.

**Portée importante :** le système NLP ne prédit ni la dangerosité d'une personne ni la réalité exhaustive de ses actes. Il repère des formulations présentes dans le champ textuel `Notes`. Une étiquette `0` signifie **« non mentionné explicitement selon les règles d'annotation »**, et non « absent dans la réalité ».

## Organisation du dépôt

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── notebooks/
│   └── sci1402_homicide_analysis.ipynb
└── src/
    ├── sci1402_nlp_test_frozen.py
    └── build_sci1402_final.py
```

Les données et résultats générés localement dans Kaggle ne sont pas tous versionnés dans ce dépôt.

## Démarche

1. **Audit des données** : structure, qualité, valeurs manquantes et variables disponibles.
2. **Préparation et exploration** : nettoyage, distributions et associations statistiques des données d'homicides.
3. **Profils de tueurs en série** : exploration d'un second jeu de données, fusion/enrichissement et analyse descriptive.
4. **Clustering exploratoire** : K-Means, analyse de sensibilité et de stabilité; représentation à quatre groupes utilisée dans les étapes textuelles ultérieures.
5. **Analyse textuelle** : examen des descriptions `Notes`, construction d'un échantillon d'annotation et extraction de caractéristiques.
6. **Validation humaine** : comparaison entre mentions extraites et annotations de référence, avec séparation des ensembles de développement et de test.
7. **Gel des règles V2** : règles fixées avant l'évaluation finale du test; les erreurs du test ne servent pas à les ajuster.

Le détail des opérations et les explications par étape se trouvent dans le [notebook scientifique](notebooks/sci1402_homicide_analysis.ipynb).

## Les 15 catégories NLP

| Famille | Catégories |
|---|---|
| Méthodes | Method Firearm, Method Knife, Method Strangulation, Method Poisoning, Method Blunt Force, Method Other |
| Victimes | Victims Women, Victims Men, Victims Children, Victims Elderly, Victims Sex Workers |
| Autres mentions | Sexual Violence Mentioned, Robbery Mentioned, Dismemberment Mentioned, Accomplices Mentioned |

L'extraction V2 est **fondée sur des règles et expressions régulières**, avec quelques exclusions contextuelles documentées dans le code. Elle ne constitue pas un modèle de langage entraîné sur les 40 dossiers de test.

## Protocole d'évaluation NLP V2

- **Jeu de test :** 40 dossiers, répartis en 4 groupes de 10.
- **Catégories par dossier :** 15.
- **Comparaisons binaires :** 600.
- **Étiquettes positives de référence :** 66; négatives : 534.
- **Prédictions :** produites à partir des règles V2 gelées, indépendamment des annotations humaines de test.
- **Référence humaine :** annotations validées et reconstruites dans le script d'évaluation.

### Résultats globaux du jeu de test

| Mesure | Résultat |
|---|---:|
| Vrais positifs (TP) | 56 |
| Faux positifs (FP) | 5 |
| Faux négatifs (FN) | 10 |
| Vrais négatifs (TN) | 529 |
| Précision (micro, classe positive) | 91,80 % |
| Rappel (micro, classe positive) | 84,85 % |
| F1 (micro, classe positive) | 88,19 % |
| Exactitude globale | 97,50 % |

Les résultats du développement et du test ne sont pas interchangeables. Les performances de test ci-dessus correspondent au protocole final, sans adaptation des règles V2 à ses erreurs.

### Contrôles d'intégrité

Le script d'évaluation vérifie les empreintes SHA-256 suivantes :

- Règles V2 : `c13d52ecef8982a0da3f9a6c33f3b2250835beb208fe15ae5745f73b5a3352a6`
- Prédictions V2 : `22a1cbec3655c66e0196e1fc789250842d0ae0d92ac72cc75e3db117019c005f`

Une divergence interrompt l'évaluation. Les règles gelées sont conservées dans [`src/sci1402_nlp_test_frozen.py`](src/sci1402_nlp_test_frozen.py) et la procédure d'évaluation dans [`src/build_sci1402_final.py`](src/build_sci1402_final.py).

## Exécution et dépendances

Le notebook a été développé dans **Kaggle** avec Python. Le code utilise notamment `pandas`, `numpy`, `scikit-learn`, `ipywidgets` et des modules de la bibliothèque standard. D'autres bibliothèques peuvent être importées par certaines cellules du notebook.

**État de reproductibilité : partiel.** L'exécution intégrale depuis une session vierge n'a pas encore été validée. Certains fichiers d'entrée sont attachés à la session Kaggle ou générés par les étapes précédentes. Les scripts publiés utilisent encore des chemins absolus tels que `/kaggle/working`; le script d'évaluation recherche notamment `/kaggle/working/annotation_test.csv` et `/kaggle/working/sci1402_nlp_test_frozen.py`. Ce dernier chemin ne correspond pas directement à l'emplacement `src/` dans ce dépôt.

Ne pas lancer les scripts d'évaluation isolément sans préparer les fichiers attendus. Ne pas réécrire les règles V2 pour obtenir un meilleur score sur le test.

## Limites méthodologiques

- Les descriptions textuelles peuvent être incomplètes, ambiguës ou hétérogènes.
- Les règles par mots-clés peuvent produire des faux positifs et des faux négatifs.
- La référence humaine repose sur des conventions d'annotation et des décisions explicites documentées dans le script.
- L'échantillon de test de 40 dossiers est limité; les scores ne prouvent pas une généralisation à d'autres corpus.
- Le clustering est **exploratoire** : les groupes statistiques ne constituent pas des typologies criminologiques validées.
- Les résultats ne doivent pas être utilisés pour profiler des personnes, établir une culpabilité ou soutenir des décisions judiciaires.

## Prochaines améliorations

- Documenter précisément les jeux de données, leurs sources et leurs licences.
- Ajouter une liste de dépendances versionnées et un guide d'installation.
- Paramétrer les chemins de fichiers et rendre l'évaluation exécutable depuis la racine du dépôt.
- Publier, si les droits et la confidentialité le permettent, les artefacts d'évaluation utiles à la vérification.
- Tester l'exécution complète dans un environnement propre.

## Licence

Consulter [LICENSE](LICENSE) pour la licence du code du dépôt. Les données externes conservent leurs propres conditions d'utilisation et de redistribution.
