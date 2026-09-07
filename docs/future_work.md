# Future Work

## 1. Purpose

This document records possible extensions without allowing them to expand the
single-semester Phase 1 scope. Future features may be considered only after the
four-class image-classification PoC has a documented public dataset, trained
checkpoint, untouched-test evaluation, failure analysis, and supervisor review.

Selecting the dataset, preparing the strict split, training the baseline, and
performing the planned evaluation are remaining Phase 1 activities—not future
features.

## 2. Prioritization Principles

A proposed extension should be accepted only when it:

- addresses a measured limitation or clearly stated research question;
- has suitable labelled data and a lawful data source;
- can be evaluated independently from the Phase 1 test set;
- preserves human oversight and transparent uncertainty;
- fits the available semester, hardware, and maintenance resources; and
- does not convert simulated information into an unsupported real-world claim.

## 3. Candidate Roadmap

| Priority | Candidate | Research value | Prerequisite |
| --- | --- | --- | --- |
| 1 | Confidence calibration and rejection analysis | Determines whether displayed confidence and the validation-selected `0.65` review rule are operationally useful | A larger independent calibration set |
| 2 | Dataset expansion and external validation | Measures generalization across cameras, vehicles, lighting, and data sources | Licence-compatible second dataset and frozen external protocol |
| 3 | Structured feedback analysis | Quantifies user-confirmed errors without automatic model changes | Sufficient reviewed feedback and privacy/retention approval |
| 4 | Similar-image search | Retrieves visually related reviewed examples without changing the classifier output | Approved embedding method, indexed reference set, relevance metric, and separate evaluation |
| 5 | Object detection | Locates one or more defect regions with bounding boxes | Bounding-box annotations and detection-specific metrics |
| 6 | Image segmentation | Estimates pixel-level defect regions | High-quality masks and segmentation-specific metrics |
| 7 | Multi-label and severity modelling | Represents coexisting defects and graded damage | Supervisor-approved label ontology and reliable annotations |
| 8 | Real supplier or inventory integration | Replaces mock stock with traceable live data | Provider agreement, API reliability plan, location/currency rules, and clear freshness/error states |
| 9 | Production deployment | Supports controlled remote or multi-user access | Authentication, authorization, privacy review, monitoring, rate limits, encryption, retention controls, and deployment testing |

Priority numbers express dependency order, not a commitment to implement every
item.

## 4. Similar-Image Search

Similar-image search remains explicitly excluded from Phase 1. A later study
would need to define:

- whether similarity means visual appearance, defect type, severity, panel
  location, or repair outcome;
- the embedding model and reference-set version;
- consent and licence rules for indexed images;
- relevance annotations and retrieval metrics such as precision at `k`;
- safeguards against presenting a visually similar image as a diagnosis; and
- separation between retrieval evidence and classifier confidence.

It must not be added as an untested nearest-neighbour feature.

## 5. Detection and Segmentation

Detection and segmentation solve different tasks from classification. Adding
them requires new annotations, models, outputs, metrics, UI explanations, and
failure analysis. Classification accuracy cannot be used as evidence that a
future bounding box or mask is correct.

Candidate evaluation would include class-wise average precision for detection
or intersection-over-union and Dice-style measures for segmentation, together
with qualitative localization review.

## 6. Knowledge Graph Research

A Neo4j or other knowledge graph is unnecessary for the current four-class
static catalog. It may become useful only if future work introduces many
components, compatibility rules, repair procedures, evidence sources, or
supplier relationships that require explicit provenance and graph queries.

The existing `Literature_Review_Automotive_KG.pdf` is background research, not
evidence that the Phase 1 system needs a graph database.

## 7. Live Inventory and Suppliers

Any replacement of the current mock inventory must clearly handle:

- supplier identity and terms;
- part-to-vehicle compatibility;
- regional price, tax, currency, and delivery differences;
- timestamps, caching, outages, and stale responses;
- stock that changes between display and purchase; and
- an explicit statement that availability is informational rather than a
  purchasing guarantee.

Live APIs must not be presented until their failure behavior and data agreement
are documented.

## 8. Feedback and Retraining

The current feedback CSV is evidence for later review, not trusted ground truth.
Before it can support retraining, feedback must be deduplicated, privacy-checked,
independently verified, and assigned to a versioned dataset under a new split
protocol. Automatic retraining and silent production model replacement remain
out of scope.

## 9. Deferred Engineering Controls

Production-oriented work would require:

- authentication and role-based access;
- multi-user data isolation;
- encrypted transport and protected storage;
- automated upload retention and deletion;
- malware scanning and stronger file inspection;
- request throttling and abuse controls;
- audit logging, monitoring, alerting, and rollback;
- model and data version registry;
- CI/CD and reproducible deployment images; and
- documented operational ownership.

These controls are not implied by a successful local classroom demonstration.

## 10. Promotion Gate

Before starting any future feature, create a short proposal containing:

1. the measured Phase 1 limitation it addresses;
2. a bounded research question;
3. required data and licence;
4. implementation effort and dependencies;
5. predeclared metrics and test protocol;
6. safety, privacy, and transparency risks; and
7. supervisor approval.

Without this gate, the default decision is to keep the feature out of scope.
