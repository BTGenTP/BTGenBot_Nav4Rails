## Catégories descriptives du dataset BTGenBot

### Distribution principale

| Catégorie | Count | % |
|---|---|---|
| navigation | 434 | 73.1% |
| other | 100 | 16.8% |
| manipulation | 43 | 7.2% |
| perception | 15 | 2.5% |
| exploration | 2 | 0.3% |

**Total :** 594

### Flag transverse `has_recovery_pattern`

**BTs avec pattern recovery :** 224 / 594 (37.7%)

### Croisement catégorie × recovery

| Catégorie | sans recovery | avec recovery |
|---|---|---|
| exploration | 1 | 1 |
| manipulation | 32 | 11 |
| navigation | 242 | 192 |
| other | 85 | 15 |
| perception | 10 | 5 |

**Méthode :** classification hybride — signal XML (skills présents) prioritaire, fallback regex texte si XML invalide.

### Implication pour le transfert SNCF

SNCF NAV4RAIL = **inspection ferroviaire** = navigation + perception.

Dataset BTGenBot **bien aligné côté navigation** (73% des exemples), **faible côté perception** (15 exemples, 2.5%) → mauvaise nouvelle pour le transfert direct sur les skills perception du catalogue SNCF. Le modèle apprendra solidement les patterns Nav2 mais aura peu de signal pour les actions de capture/analyse visuelle propres à l'inspection ferroviaire.
