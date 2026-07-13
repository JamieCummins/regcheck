"""report_shares table + migrate private->unlisted visibility

Revision ID: c1a2b3d4e5f6
Revises: b3ed8f88c31a
Create Date: 2026-06-14 16:20:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c1a2b3d4e5f6'
down_revision: Union[str, None] = 'b3ed8f88c31a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'report_shares',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('task_id', sa.String(length=64), nullable=False),
        sa.Column('grantee_email', sa.String(length=320), nullable=True),
        sa.Column('grantee_orcid', sa.String(length=32), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['task_id'], ['reports.task_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('task_id', 'grantee_email', name='uq_share_task_email'),
        sa.UniqueConstraint('task_id', 'grantee_orcid', name='uq_share_task_orcid'),
    )
    with op.batch_alter_table('report_shares', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_report_shares_task_id'), ['task_id'], unique=False)

    # The old "private" tier meant "unlisted, link-only" — preserve that meaning
    # under the new three-tier vocabulary (public / unlisted / restricted).
    op.execute("UPDATE reports SET visibility='unlisted' WHERE visibility='private'")


def downgrade() -> None:
    op.execute("UPDATE reports SET visibility='private' WHERE visibility IN ('unlisted','restricted')")
    with op.batch_alter_table('report_shares', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_report_shares_task_id'))
    op.drop_table('report_shares')
