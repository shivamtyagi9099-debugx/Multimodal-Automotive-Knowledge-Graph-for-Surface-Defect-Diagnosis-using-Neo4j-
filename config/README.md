# Configuration

This directory contains the non-secret configuration for the car-body-panel
defect-classification proof of concept.

## Files

- `class_names.json` defines the model output order: `normal`, `scratch`,
  `dent`, and `rust`. Do not reorder these values after training a model;
  prediction indices are mapped to labels in this exact order.
- `settings.yaml` configures model inference, the 65% unknown-prediction
  threshold, upload validation, local paths, feedback storage, and the static
  simulated-inventory date used by the UI.
- `neo4j.yaml` contains Neo4j connection defaults, graph seed sources, and
  conservative local connection limits. It intentionally contains no password.

## Neo4j credentials

Copy the root `.env.example` file to `.env`, replace the placeholder password,
and keep `.env` out of source control. The Neo4j tooling reads the password from
the environment variable named by `password_environment_variable` in
`neo4j.yaml`.

The following environment variables may override the non-secret connection
defaults when the Neo4j service is used:

- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

## PoC invariants

- The project supports one object type: `car_body_panel`.
- The model has one normal class and three defect classes.
- Predictions below `0.65` confidence must return the configured manual-review
  message.
- The JSON catalog is the reproducible source used to seed Neo4j; Neo4j stores
  the defect-to-repair-step and defect-to-part relationships queried by the
  application. If Neo4j is unavailable, the API explicitly labels and uses the
  same JSON catalog as its local fallback.
- Inventory values are simulated and must always be presented with their static
  last-updated date.
