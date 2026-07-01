"""FastAPI service exposing Convexity's research scans, companies and universe.

The application factory and module-level app live in :mod:`convexity.api.app`
(``from convexity.api.app import create_app, app``). Importing this package itself
is side-effect-free: it neither builds the app nor starts a scan, so it stays safe
to import on a machine with no network and no provider credentials.

Run the service with::

    uvicorn convexity.api.app:app --reload
"""

from __future__ import annotations
