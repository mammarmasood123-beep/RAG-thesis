# Experiment A — pilotA-stub-20261008T162923Z_runC

> **STUB GENERATOR. These numbers are synthetic and test the harness only.**

Questions: 20 · records: 260 · not fitting context: 0 · status: OK


## EM / F1 by (N, T)

| N \ T | 0 | 1* | 2* | 3* | 4* | 5 | 6* |
|---|---|---|---|---|---|---|---|
| EM N=0 | 15.0 [0.0, 30.0] | 25.0 [10.0, 45.0] | 85.0 [70.0, 100.0] | 85.0 [70.0, 100.0] | 85.0 [70.0, 100.0] | 85.0 [70.0, 100.0] | 85.0 [70.0, 100.0] |
| EM N=4 | 15.0 [0.0, 30.0] | 30.0 [10.0, 50.0] | 40.0 [20.0, 60.0] | 55.0 [35.0, 75.0] | 65.0 [45.0, 85.0] | 30.0 [10.0, 50.0] | 40.0 [20.0, 60.0] |
| F1 N=0 | 15.0 [0.0, 30.0] | 25.0 [10.0, 45.0] | 85.0 [70.0, 100.0] | 85.0 [70.0, 100.0] | 85.0 [70.0, 100.0] | 85.0 [70.0, 100.0] | 85.0 [70.0, 100.0] |
| F1 N=4 | 15.0 [0.0, 30.0] | 30.0 [10.0, 50.0] | 40.0 [20.0, 60.0] | 55.0 [35.0, 75.0] | 65.0 [45.0, 85.0] | 30.0 [10.0, 50.0] | 40.0 [20.0, 60.0] |

`*` = design T level. T=0 is closed-book (shared). Cells: % [95% bootstrap CI].

## N=0 control (must be flat from T=2)

EM by T: T0=15.0, T1=25.0, T2=85.0, T3=85.0, T4=85.0, T5=85.0, T6=85.0

F1 by T: T0=15.0, T1=25.0, T2=85.0, T3=85.0, T4=85.0, T5=85.0, T6=85.0

max |EM(T) − EM(2)| for T ≥ 2: **0.00 pts**

prompt byte-identity violations (hard): 0 · output mismatches across T≥2 (soft): 0

## Paired difference vs N=0 (same questions), EM points

| N | T=1 | T=2 | T=3 | T=4 | T=5 | T=6 |
|---|---|---|---|---|---|---|
| 4 | +5.0 [-20.0, +30.0] | -45.0 [-75.0, -10.0] | -30.0 [-55.0, -5.0] | -20.0 [-45.0, +5.0] | -55.0 [-80.0, -25.0] | -45.0 [-65.0, -25.0] |

## Closed-book (round 0) — contamination check

| subset | n | EM | F1 |
|---|---|---|---|
| all | 20 | 15.0 | 15.0 |
| bridge | 15 | 13.3 | 13.3 |
| comparison | 5 | 20.0 | 20.0 |
| comparison: entity gold | 3 | 33.3 | 33.3 |
| comparison: yes/no gold | 2 | 0.0 | 0.0 |

Items answered correctly closed-book (EM=1): **3** → `results/closed_book_correct.txt`. Comparison items have a two-way answer space (yes/no, or one of two named entities), so chance is ~50% there: a correct closed-book answer on a comparison item is weak evidence of memorisation.

## Sensitivity: excluding 3 closed-book-correct items

| N \ T | 0 | 1* | 2* | 3* | 4* | 5 | 6* |
|---|---|---|---|---|---|---|---|
| EM N=0 | 0.0 [0.0, 0.0] | 23.5 [5.9, 47.1] | 88.2 [70.6, 100.0] | 88.2 [70.6, 100.0] | 88.2 [70.6, 100.0] | 88.2 [70.6, 100.0] | 88.2 [70.6, 100.0] |
| EM N=4 | 0.0 [0.0, 0.0] | 29.4 [11.8, 52.9] | 41.2 [17.6, 64.7] | 58.8 [35.3, 82.4] | 64.7 [41.2, 88.2] | 29.4 [5.9, 52.9] | 41.2 [17.6, 64.7] |
| F1 N=0 | 0.0 [0.0, 0.0] | 23.5 [5.9, 47.1] | 88.2 [70.6, 100.0] | 88.2 [70.6, 100.0] | 88.2 [70.6, 100.0] | 88.2 [70.6, 100.0] | 88.2 [70.6, 100.0] |
| F1 N=4 | 0.0 [0.0, 0.0] | 29.4 [11.8, 52.9] | 41.2 [17.6, 64.7] | 58.8 [35.3, 82.4] | 64.7 [41.2, 88.2] | 29.4 [11.8, 52.9] | 41.2 [17.6, 64.7] |

`*` = design T level. T=0 is closed-book (shared). Cells: % [95% bootstrap CI].

## Output hygiene

parse failures: 0/260 · said 'unsure' despite no-abstain prompt: 0/260

worst cells by parse failures: ('closed_book', 0): 0/20, (0, 1): 0/20, (0, 2): 0/20

## Token ceiling (prompt tokens incl. chat template, max over questions)

window 8192, minus max_new_tokens → prompt budget 7936

| N | T=0 | T=1 | T=2 | T=3 | T=4 | T=5 | T=6 |
|---|---|---|---|---|---|---|---|
| 0 | 131 | 324 | 382 | 382 | 382 | 382 | 382 |
| 4 | 131 | 884 | 1587 | 1966 | 2598 | 3060 | 3480 |

(stub counts whitespace tokens; the real ceiling comes from the vLLM run)

## Parse failures and 'unsure' answers by (N, T)

Rates over records that ran (records not fitting the context window are counted separately). Closed-book is round 0.

| N | T | n | not_fit | parse_fail | parse_fail_rate | said_unsure | said_unsure_rate |
|---|---|---|---|---|---|---|---|
| closed_book | 0 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
| 0 | 1 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
| 0 | 2 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
| 0 | 3 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
| 0 | 4 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
| 0 | 5 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
| 0 | 6 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
| 4 | 1 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
| 4 | 2 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
| 4 | 3 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
| 4 | 4 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
| 4 | 5 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
| 4 | 6 | 20 | 0 | 0 | 0.0% | 0 | 0.0% |
