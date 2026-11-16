# Evaluation: every question (latest-revision filter on)

Run 2026-11-16 11:27 UTC | 47 questions | index `aero-manuals` | chat `gpt-5.4-mini` | cosine threshold 0.35 | hybrid weights [0.5, 0.5]

| id | type | expected | got | cited | values | outdated | LLM calls | cosine | correct |
|---|---|---|---|---|---|---|---|---|---|
| T01 | torque | answered | answered | A-200 32-41-01 r3 | torque 1/1 | - | 1 | 0.73 | yes |
| T02 | torque | answered | answered | B-340 32-41-01 r5 | torque 1/1 | - | 1 | 0.74 | yes |
| T03 | torque | answered | answered | C-75 34-11-02 r2 | torque 1/1 | - | 1 | 0.68 | yes |
| T04 | torque | answered | answered | C-75 24-31-01 r2 | torque 1/1 | - | 1 | 0.68 | yes |
| T05 | torque | answered | answered | A-200 61-10-02 r1 | torque 1/1 | - | 1 | 0.71 | yes |
| T06 | torque | answered | answered | B-340 27-51-04 r1 | torque 1/1 | - | 1 | 0.72 | yes |
| T07 | torque | answered | answered | A-200 29-11-03 r2 | torque 1/1 | - | 1 | 0.69 | yes |
| T08 | torque | answered | answered | B-340 79-21-02 r2 | torque 1/1 | - | 1 | 0.62 | yes |
| T09 | torque | answered | answered | C-75 32-42-03 r1 | torque 1/1 | - | 1 | 0.59 | yes |
| T10 | torque | answered | answered | B-340 21-31-04 r3 | torque 1/1 | - | 1 | 0.64 | yes |
| T11 | torque | answered | answered | C-75 27-31-02 r2 | torque 1/1 | - | 1 | 0.70 | yes |
| P01 | part_number | answered | answered | A-200 32-41-01 r3 | P/N 1/1 | - | 1 | 0.62 | yes |
| P02 | part_number | answered | answered | A-200 21-31-04 r2 | P/N 1/1 | - | 1 | 0.70 | yes |
| P03 | part_number | answered | answered | B-340 28-21-05 r3 | P/N 1/1 | - | 1 | 0.74 | yes |
| P04 | part_number | answered | answered | C-75 32-21-02 r1 | P/N 1/1 | - | 1 | 0.72 | yes |
| P05 | part_number | answered | answered | B-340 34-11-02 r1 | P/N 1/1 | - | 1 | 0.66 | yes |
| P06 | part_number | answered | answered | A-200 24-31-01 r3 | P/N 1/1 | - | 1 | 0.61 | yes |
| P07 | part_number | answered | answered | C-75 28-21-05 r2 | P/N 1/1 | - | 1 | 0.68 | yes |
| P08 | part_number | answered | answered | B-340 32-42-03 r3 | P/N 1/1 | - | 1 | 0.71 | yes |
| S01 | procedure | answered | answered | A-200 21-31-04 r2 | steps 12/12 | - | 1 | 0.74 | yes |
| S02 | procedure | answered | answered | B-340 21-31-04 r3 | steps 12/12 | - | 1 | 0.73 | yes |
| S03 | procedure | answered | answered | C-75 34-11-02 r2 | steps 10/10 | - | 1 | 0.78 | yes |
| S04 | procedure | answered | answered | B-340 27-51-04 r1 | steps 9/9 | - | 1 | 0.78 | yes |
| S05 | procedure | answered | answered | A-200 32-41-01 r3 | steps 10/10 | - | 1 | 0.77 | yes |
| S06 | procedure | answered | answered | C-75 32-21-02 r1 | steps 8/8 | - | 1 | 0.70 | yes |
| N01 | no_aircraft | answered | answered | A-200 61-10-02 r1 | torque 1/1 | - | 1 | 0.63 | yes |
| N02 | no_aircraft | answered | answered | B-340 27-51-04 r1 | torque 1/1 | - | 1 | 0.66 | yes |
| N03 | no_aircraft | answered | answered | C-75 27-31-02 r2 | P/N 1/1 | - | 1 | 0.65 | yes |
| N04 | no_aircraft | answered | answered | B-340 79-21-02 r2 | P/N 1/1 | - | 1 | 0.62 | yes |
| I01 | identifier | answered | answered | A-200 32-41-01 r3 | P/N 1/1 | - | 1 | 0.46 | yes |
| I02 | identifier | answered | abstained | - | P/N 0/1 | - | 1 | 0.47 | NO |
| I03 | identifier | answered | answered | C-75 27-31-02 r2 | P/N 1/1 | - | 1 | 0.43 | yes |
| I04 | identifier | answered | answered | A-200 32-42-03 r2 | torque 1/1 | - | 1 | 0.72 | yes |
| I05 | identifier | answered | answered | C-75 34-11-02 r2 | torque 1/1 | - | 1 | 0.70 | yes |
| C01 | clarification | clarification_needed | clarification_needed | - | - | - | 0 | 0.67 | yes |
| C02 | clarification | clarification_needed | clarification_needed | - | - | - | 0 | 0.71 | yes |
| C03 | clarification | clarification_needed | clarification_needed | - | - | - | 0 | 0.64 | yes |
| C04 | clarification | clarification_needed | clarification_needed | - | - | - | 0 | 0.63 | yes |
| U01 | unanswerable | abstained | abstained | - | - | - | 1 | 0.52 | yes |
| U02 | unanswerable | abstained | abstained | - | - | - | 1 | 0.48 | yes |
| U03 | unanswerable | abstained | abstained | - | - | - | 1 | 0.41 | yes |
| U04 | unanswerable | abstained | answered | A-200 32-41-01 r3 | - | - | 1 | 0.72 | NO |
| U05 | unanswerable | abstained | abstained | - | - | - | 0 | 0.48 | yes |
| U06 | unanswerable | abstained | abstained | - | - | - | 0 | 0.10 | yes |
| U07 | unanswerable | abstained | abstained | - | - | - | 0 | 0.07 | yes |
| U08 | unanswerable | abstained | abstained | - | - | - | 0 | 0.11 | yes |
| U09 | unanswerable | rejected | rejected | - | - | - | 0 | - | yes |
