## Smoke test set (15 BTs)

**Seed :** 42 — **n :** 15 BTs (fidèles aux 594 originaux, non retirés du train).

### Composition

| Catégorie | Bucket | n | record_ids |
|---|---|---|---|
| manipulation | medium | 1 | 8 |
| manipulation | small | 1 | 453 |
| navigation | large | 3 | 241, 363, 405 |
| navigation | medium | 3 | 36, 50, 357 |
| navigation | small | 3 | 133, 223, 303 |
| other | medium | 1 | 359 |
| other | small | 1 | 214 |
| perception | medium | 1 | 507 |
| perception | small | 1 | 78 |

**Usage :** après chaque SFT, générer les sorties du modèle sur ces 15 inputs et comparer à la main aux outputs de référence. Permet repérer rapidement les modes de défaillance (XML cassé, structure plate, oubli recovery, etc.) sans lancer l'eval complète.
