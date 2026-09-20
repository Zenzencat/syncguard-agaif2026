# Demo database reset

Stop the SyncGuard API, then run `make reset-demo-db` from the repository root to remove
only `data/syncguard.db*` (the demo SQLite database plus any journal/WAL companions). The
next API start creates an empty database. Do not run this target during a demo or while the
API is writing events.
