"""Opt-in API-key authentication.

**Auth is off unless SYNCGUARD_API_KEY is set.** With the variable unset or empty, every
route behaves exactly as it did before this module existed -- `docker compose up` with no
configuration still works, the dashboard still connects, and nothing prompts for a key. That
is deliberate: the demo path must not acquire a setup step, and a security control nobody can
turn on in the room they are demoing in is worse than one that is explicitly opt-in.

With the variable set, every route except the exemptions below requires the key.

Comparison is constant-time (`hmac.compare_digest`). The key is never logged, never included
in an error message, never echoed in a response, and never placed in a URL -- see the SSE
section below for why that last one took work.

Accepted credentials
--------------------
1. ``X-API-Key: <key>``            -- preferred for scripts and collectors
2. ``Authorization: Bearer <key>`` -- for callers whose HTTP client assumes bearer tokens
3. ``sg_session`` cookie           -- issued by POST /auth/session, for the browser only

Exempt routes (never require a key)
-----------------------------------
``/health``            so a container healthcheck, load balancer or uptime probe works with
                       no secret distributed to it. It reports liveness, the model version
                       tag, tower count and whether auth is required -- no event data, no
                       scores, no telemetry.
``/``, ``/dashboard``  the dashboard is a single self-contained HTML file with no data in it.
                       It ships no key and cannot act without one: every fetch it makes goes
                       to a protected route. Serving the shell unauthenticated means a judge
                       can open the page and be told to enter a key, instead of seeing a bare
                       401 with no way forward.
``/docs``, ``/redoc``, ``/openapi.json``
                       the API schema. It describes shapes, not data. Exempting it keeps the
                       interactive docs usable as the place a caller discovers how to send
                       the key in the first place.
``/auth/session``      issues the cookie; it validates the key itself.

Everything else -- /score, /ingest, /events, /towers, /replay/*, /stream/events, /feedback/*,
/metrics, /drift -- requires the key when auth is on.

Server-Sent Events, and why there is a cookie
---------------------------------------------
The dashboard's live feed uses ``EventSource``, which **cannot send custom headers**. That
rules out ``X-API-Key`` for /stream/events, and forces a choice. The three options and why
this one:

1. **Exempt /stream/events.** Rejected. The stream carries live scored events -- severities,
   tower attributions, predictions. Exempting it would mean the one endpoint that
   continuously pushes the system's actual output is the one endpoint with no key on it. The
   control would be decorative.

2. **Query token, ``/stream/events?api_key=...``.** Rejected. It works with EventSource and
   it is what a lot of projects do, but it puts the secret in a URL, and URLs leak in ways
   headers do not: access logs (including this service's own request log), reverse-proxy
   logs, browser history, the Referer header on any outbound link, and shoulder-surfing a
   demo screen. A secret in a URL is a secret in several places nobody audits.

3. **Short-lived session cookie (chosen).** The dashboard POSTs the key once to
   /auth/session over the normal request path, where it travels in a header and a body that
   are not logged. The server validates it and returns an opaque random session token as an
   ``HttpOnly`` cookie. ``EventSource(url, {withCredentials: true})`` sends that cookie
   automatically. The key itself never appears in a URL, never reaches JavaScript, and never
   lands in ``localStorage``.

The cost of choice 3, stated rather than buried:

- **Sessions are in-process.** Restarting the service invalidates every session and the
  dashboard must re-enter the key. There is no session store, no Redis, no shared state
  across replicas. For a single-container demo service this is fine; for a horizontally
  scaled deployment it is not, and a real deployment should put a signed stateless token
  (or a real identity provider) here instead.
- **Cookies bring CORS constraints.** A browser will not send credentials to an origin whose
  ``Access-Control-Allow-Origin`` is ``*``. The default posture -- dashboard served by the
  same service that answers its fetches -- needs no CORS at all and works. Pointing the
  dashboard at a *different* API host while auth is on requires naming that origin in
  ``SYNCGUARD_CORS_ORIGINS``; wildcard CORS and credentials cannot coexist. See
  ``api/main.py`` for where that is applied.
- **A cookie is a session, not an identity.** It says "this browser presented the shared
  key", nothing more. It does not identify an analyst, which is why the feedback layer's
  ``analyst`` field remains unverified free text (FEEDBACK_LOOP.md).
- **CSRF.** The cookie is ``SameSite=Strict``, so a third-party page cannot ride it. It is
  not set ``Secure`` by default because the demo runs over plain HTTP on localhost; behind
  TLS, set ``SYNCGUARD_COOKIE_SECURE=1``. That default is a demo convenience and is the first
  thing to change for any real deployment.
"""
from __future__ import annotations

import hmac
import os
import secrets
import threading
import time

# Routes that never require a key. Dashboard assets are public because the unauthenticated
# dashboard shell must load them before it can exchange an API key for its HttpOnly session.
EXEMPT_PATHS = frozenset({
    "/health",
    "/",
    "/dashboard",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
    "/auth/session",
})
EXEMPT_PREFIXES = ("/docs/", "/assets/")

API_KEY_HEADER = "X-API-Key"
SESSION_COOKIE = "sg_session"
SESSION_TTL_SECONDS = 12 * 3600  # a working day; the dashboard re-auths transparently after

MIN_KEY_LENGTH = 16


class SessionStore:
    """In-memory session tokens issued by POST /auth/session.

    Deliberately not persisted: see the module docstring's cost list. Expired tokens are
    swept lazily on access rather than by a background task -- there is no sweeper to leak,
    and the store is bounded by how many times someone has typed the key into a browser.
    """

    def __init__(self, ttl_seconds: int = SESSION_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._tokens: dict[str, float] = {}  # token -> expiry (monotonic seconds)

    def issue(self) -> tuple[str, int]:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._prune_locked()
            self._tokens[token] = time.monotonic() + self._ttl
        return token, self._ttl

    def validate(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            self._prune_locked()
            expiry = self._tokens.get(token)
            return expiry is not None and expiry > time.monotonic()

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._tokens.pop(token, None)

    def clear(self) -> None:
        with self._lock:
            self._tokens.clear()

    def _prune_locked(self) -> None:
        now = time.monotonic()
        for tok in [t for t, exp in self._tokens.items() if exp <= now]:
            del self._tokens[tok]

    @property
    def active(self) -> int:
        with self._lock:
            self._prune_locked()
            return len(self._tokens)


class ApiKeyAuth:
    """Holds the configured key (if any) and answers "is this request authorized?"."""

    def __init__(self, api_key: str | None = None, sessions: SessionStore | None = None):
        raw = api_key if api_key is not None else os.environ.get("SYNCGUARD_API_KEY", "")
        self._key = (raw or "").strip()
        self.sessions = sessions or SessionStore()
        self.cookie_secure = os.environ.get("SYNCGUARD_COOKIE_SECURE", "").strip() in {"1", "true", "True"}

    @property
    def enabled(self) -> bool:
        """True only when a key is configured. This is the whole opt-in switch."""
        return bool(self._key)

    @property
    def weak_key(self) -> bool:
        """A configured key shorter than MIN_KEY_LENGTH. Reported at startup and in /health
        as a boolean -- never with the key, its length, or any part of it."""
        return self.enabled and len(self._key) < MIN_KEY_LENGTH

    @staticmethod
    def is_exempt(path: str) -> bool:
        return path in EXEMPT_PATHS or path.startswith(EXEMPT_PREFIXES)

    def check_key(self, candidate: str | None) -> bool:
        """Constant-time comparison. Returns False for a missing candidate without comparing,
        which leaks only "nothing was sent" -- already obvious from the request itself."""
        if not self.enabled or not candidate:
            return False
        return hmac.compare_digest(candidate, self._key)

    def extract_key(self, headers) -> str | None:
        """Pull a presented key out of the request headers. Header lookup is
        case-insensitive via Starlette's Headers mapping."""
        presented = headers.get(API_KEY_HEADER)
        if presented:
            return presented.strip()
        authorization = headers.get("Authorization") or ""
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return None

    def authorize(self, path: str, headers, cookies) -> bool:
        """The single decision point used by the middleware."""
        if not self.enabled or self.is_exempt(path):
            return True
        if self.check_key(self.extract_key(headers)):
            return True
        return self.sessions.validate(cookies.get(SESSION_COOKIE))
