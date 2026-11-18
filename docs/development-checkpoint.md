# Development baseline checkpoint

Source: the team's terminal screenshot of the completed real-data baseline run on 18 November 2026. Values below are rounded transcriptions, not a replacement for the original generated CSV files. No evaluator-test results are reported.

| Model | CV Macro F1 | Student Macro F1 | Candidate accuracy | Top-1 proxy accuracy | Direct route continuity | Candidate log loss | Final fit (s) | Inference (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Histogram Gradient Boosting | 0.8590 | 0.8722 | 0.9194 | 0.8047 | 0.8993 | 0.1861 | 3.2837 | 0.0762 |
| Random Forest | 0.8563 | 0.8736 | 0.9209 | 0.8074 | 0.8986 | 0.1856 | 34.3069 | 0.1395 |
| Decision Tree | 0.8536 | 0.8728 | 0.9190 | 0.8048 | 0.8991 | 0.2103 | 8.1387 | 0.0068 |
| Logistic Regression | 0.8311 | 0.8448 | 0.9038 | 0.7932 | 0.8865 | 0.2373 | 1.2020 | 0.0061 |
| Gaussian Naive Bayes | 0.8011 | 0.7946 | 0.8447 | 0.7947 | 0.8877 | 0.5370 | 0.1135 | 0.0093 |

Candidate miss rate: 0.0112 for all five models in the shared candidate set. Five-fold GroupKFold groups observations by trip. The reference labels are nearest-reference-segment proxies; the reference arc sequence is not aligned in chronological order, so the pointwise results should not be represented as verified physical ground truth.

Provisional choice: Histogram Gradient Boosting, because it has the highest mean CV Macro F1, slightly higher direct route continuity than Random Forest, and shorter measured fit and inference times than Random Forest. Random Forest has higher student-test Macro F1 and Top-1 proxy accuracy. These differences are small; final selection is pending tuning and real-data decoder experiments.

The full original experiment logs, figures and fitted models remain to be uploaded from the training machine. The practice dashboard and its previous HMM continuity measurements are not real-data evaluation results.
