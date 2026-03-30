# What Till We Do Wrong (Task 4)

- Total evaluated rows: 40080
- Zero-shot error rate: 0.6368

## Worst Concept/Template Buckets

- `negation` + `low_specificity`: error_rate=0.8325 (795/955)
- `negation` + `concept_framing`: error_rate=0.7099 (678/955)
- `negation` + `high_specificity`: error_rate=0.7079 (676/955)
- `negation` + `negation_explicit`: error_rate=0.6859 (655/955)
- `spatial` + `base`: error_rate=0.6739 (343/509)
- `count` + `base`: error_rate=0.6647 (228/343)
- `action` + `base`: error_rate=0.6604 (3326/5036)
- `object_or_attribute` + `base`: error_rate=0.6402 (751/1173)
- `action` + `high_specificity`: error_rate=0.6392 (3219/5036)
- `object_or_attribute` + `high_specificity`: error_rate=0.6317 (741/1173)

## Frequent Confusions

- true `contradiction` -> predicted `neutral`: 5719
- true `entailment` -> predicted `neutral`: 4927
- true `entailment` -> predicted `contradiction`: 3804
- true `neutral` -> predicted `entailment`: 3789
- true `neutral` -> predicted `contradiction`: 3657
- true `contradiction` -> predicted `entailment`: 3627