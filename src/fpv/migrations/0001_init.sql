-- fpv schema v1 (design "SQLite schema"). Applied by fpv.storage.Repository.migrate inside one
-- BEGIN IMMEDIATE transaction together with its schema_migrations row. No pixel data is stored:
-- samples keep paths and digests only (Req 11.6).

CREATE TABLE IF NOT EXISTS schema_migrations (
  version    INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE samples (
  sample_id   TEXT PRIMARY KEY,
  source_path TEXT NOT NULL,
  sha256      TEXT NOT NULL,
  subject_id  INTEGER,
  hand        TEXT,
  finger      TEXT,
  alteration  TEXT,
  difficulty  TEXT,
  split       TEXT,
  excluded    INTEGER NOT NULL DEFAULT 0 CHECK (excluded IN (0, 1)),
  width       INTEGER NOT NULL,
  height      INTEGER NOT NULL,
  created_at  TEXT NOT NULL
);

CREATE TABLE templates (
  template_id       TEXT PRIMARY KEY,
  user_id           INTEGER NOT NULL CHECK (user_id BETWEEN 1 AND 600),
  hand              TEXT NOT NULL CHECK (hand IN ('Left', 'Right')),
  finger            TEXT NOT NULL CHECK (finger IN ('thumb', 'index', 'middle', 'ring', 'little')),
  template_location TEXT NOT NULL,
  source_sample_id  TEXT NOT NULL REFERENCES samples(sample_id),
  quality_before    INTEGER NOT NULL,
  quality_after     TEXT NOT NULL,          -- JSON [{attempt_no, configuration_id, score|null}]
  minutiae_count    INTEGER NOT NULL,
  scale_factor      REAL NOT NULL,
  pipeline_version  TEXT NOT NULL,
  configuration_id  TEXT NOT NULL,
  template_version  INTEGER NOT NULL CHECK (template_version >= 1),
  is_active         INTEGER NOT NULL CHECK (is_active IN (0, 1)),
  created_at        TEXT NOT NULL,
  UNIQUE (user_id, hand, finger, template_version)
);
-- At most one active template per identity (Req 8.5).
CREATE UNIQUE INDEX ux_templates_one_active ON templates(user_id, hand, finger) WHERE is_active = 1;

CREATE TABLE runs (
  run_id              TEXT PRIMARY KEY,
  session_id          TEXT NOT NULL,
  presentation_no     INTEGER NOT NULL,
  operation           TEXT NOT NULL CHECK (operation IN ('enroll', 'verify')),
  claimed_user_id     INTEGER NOT NULL,
  claimed_hand        TEXT NOT NULL,
  claimed_finger      TEXT NOT NULL,
  sample_id           TEXT REFERENCES samples(sample_id),
  status              TEXT,
  status_reason       TEXT,
  attempts_used       INTEGER NOT NULL DEFAULT 0,
  control_mode        TEXT NOT NULL DEFAULT 'controller',
  orchestrator_vision TEXT NOT NULL DEFAULT 'off',
  pipeline_version    TEXT NOT NULL,
  config_set_id       TEXT NOT NULL,
  started_at          TEXT NOT NULL,
  ended_at            TEXT
);
CREATE INDEX ix_runs_session ON runs(session_id);

CREATE TABLE stage_events (
  event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id           TEXT NOT NULL REFERENCES runs(run_id),
  seq              INTEGER NOT NULL,
  stage            TEXT NOT NULL,
  attempt_no       INTEGER,
  configuration_id TEXT,
  started_at       TEXT NOT NULL,
  duration_ms      REAL NOT NULL,
  outcome          TEXT NOT NULL,           -- ok | rejected | error | skipped
  details_json     TEXT NOT NULL DEFAULT '{}',
  UNIQUE (run_id, seq)
);

CREATE TABLE verifications (
  run_id               TEXT PRIMARY KEY REFERENCES runs(run_id),
  template_id          TEXT NOT NULL REFERENCES templates(template_id),
  probe_location       TEXT NOT NULL,
  score                INTEGER NOT NULL,
  threshold            INTEGER NOT NULL,
  decision             TEXT NOT NULL CHECK (decision IN ('MATCH', 'NO_MATCH')),
  ground_truth_genuine INTEGER CHECK (ground_truth_genuine IN (0, 1))
);
