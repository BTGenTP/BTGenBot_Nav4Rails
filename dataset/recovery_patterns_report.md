## Recovery patterns dans le dataset BTGenBot

**Méthode :** signature récursive d'un sous-arbre, type `Fallback(Sequence, Sequence)`. Extraction sur 578 BTs valides (parse strict).

### Top 10 patterns Fallback/RecoveryNode (profondeur 2)

| Pattern | Count |
|---|---|
| `Fallback(Condition, Action)` | 31 |
| `Fallback(Condition, ForceFailure(Action))` | 27 |
| `RecoveryNode(FollowPath, ClearEntireCostmap)` | 26 |
| `Fallback(Action, Action)` | 25 |
| `Fallback(Action, ForceFailure(Action))` | 25 |
| `RecoveryNode(PipelineSequence(RateController, RecoveryNode), ReactiveFallback(GoalUpdated, SequenceStar))` | 23 |
| `RecoveryNode(ComputePathToPose, ClearEntireCostmap)` | 21 |
| `ReactiveFallback(GoalUpdated, SequenceStar(ClearEntireCostmap, ClearEntireCostmap, Spin, Wait))` | 21 |
| `Fallback(Condition, Condition)` | 20 |
| `Fallback(Sequence(Condition, Action), Action)` | 19 |

### Top 10 patterns Fallback/RecoveryNode (profondeur 3)

| Pattern | Count |
|---|---|
| `Fallback(Condition, Action)` | 31 |
| `Fallback(Condition, ForceFailure(Action))` | 27 |
| `RecoveryNode(FollowPath, ClearEntireCostmap)` | 26 |
| `Fallback(Action, Action)` | 25 |
| `Fallback(Action, ForceFailure(Action))` | 25 |
| `RecoveryNode(ComputePathToPose, ClearEntireCostmap)` | 21 |
| `ReactiveFallback(GoalUpdated, SequenceStar(ClearEntireCostmap, ClearEntireCostmap, Spin, Wait))` | 21 |
| `RecoveryNode(PipelineSequence(RateController(RecoveryNode), RecoveryNode(FollowPath, ClearEntireCostmap)), ReactiveFallback(GoalUpdated, SequenceStar(ClearEntireCostmap, ClearEntireCostmap, Spin, Wait)))` | 20 |
| `Fallback(Condition, Condition)` | 20 |
| `Fallback(Sequence(Condition, Action), Action)` | 19 |

### Lecture

- Le pattern dominant à profondeur 2 est `Fallback(Sequence, Sequence)` — motif "try then recover".
- À profondeur 3 on voit apparaître les variantes imbriquées (`Fallback(Sequence(Fallback))`, etc.) qui sont la signature de la stratégie de recovery cascadée standard Nav2.
- Implication pour le SFT : le modèle apprendra ces motifs récurrents en priorité, utile pour les tâches d'inspection SNCF qui nécessiteront aussi un comportement "essayer puis se rabattre" (ex : capture image → réessayer depuis autre angle → renoncer).
