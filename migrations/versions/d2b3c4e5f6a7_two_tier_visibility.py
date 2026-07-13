"""collapse visibility to two tiers: public / private

Folds the previous three-tier vocabulary into two:
  - restricted -> private  (the allow-list tier keeps its share grants)
  - unlisted   -> private  (link-only is removed; not made public, to avoid
                            exposing anything that wasn't already public)

Revision ID: d2b3c4e5f6a7
Revises: c1a2b3d4e5f6
Create Date: 2026-06-15 09:30:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'd2b3c4e5f6a7'
down_revision: Union[str, None] = 'c1a2b3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE reports SET visibility='private' WHERE visibility IN ('restricted', 'unlisted')")


def downgrade() -> None:
    # Best-effort: restore the prior default tier name for non-public reports.
    op.execute("UPDATE reports SET visibility='unlisted' WHERE visibility='private'")
