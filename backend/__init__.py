"""RegCheck backend package.

``create_app`` is exported lazily (PEP 562) so that importing submodules that
only need models/metadata — notably Alembic's ``migrations/env.py`` importing
``backend.db.base`` during the Heroku release phase — does not drag in the whole
web app (routes, settings validation, provider clients).
"""


def __getattr__(name: str):
    if name == "create_app":
        from .main import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["create_app"]
