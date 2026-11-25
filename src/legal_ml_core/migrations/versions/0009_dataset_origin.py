"""A Dataset records the Dataset it was copied from (BAIGLE #197).

A practice copy is an unlabelled copy of a labelled Dataset; ``origin_dataset_id``
names the Dataset it came from. Nullable, no backfill: every Dataset before this
is an original. No foreign key: the original may be deleted, and the copy keeps
its id as a record of where it came from.
"""

from alembic import op
import sqlalchemy as sa


revision = "lmc_0009_dataset_origin"
down_revision = "lmc_0008_file_label"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("datasets") as batch:
        batch.add_column(sa.Column("origin_dataset_id", sa.String(36)))


def downgrade():
    with op.batch_alter_table("datasets") as batch:
        batch.drop_column("origin_dataset_id")
