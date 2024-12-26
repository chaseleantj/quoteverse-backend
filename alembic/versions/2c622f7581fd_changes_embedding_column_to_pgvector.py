"""Changes embedding column to pgvector

Revision ID: 2c622f7581fd
Revises: daf35cb9d29c
Create Date: 2024-12-26 11:04:35.351241

"""
from typing import Sequence, Union

from alembic import op
import pgvector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '2c622f7581fd'
down_revision: Union[str, None] = 'daf35cb9d29c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('motivational_quotes', 'embeddings',
               existing_type=postgresql.ARRAY(sa.DOUBLE_PRECISION(precision=53)),
               type_=pgvector.sqlalchemy.vector.VECTOR(dim=1536),
               existing_nullable=True)
    op.alter_column('motivational_quotes', 'reduced_embeddings',
               existing_type=postgresql.ARRAY(sa.DOUBLE_PRECISION(precision=53)),
               type_=pgvector.sqlalchemy.vector.VECTOR(dim=2),
               existing_nullable=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('motivational_quotes', 'reduced_embeddings',
               existing_type=pgvector.sqlalchemy.vector.VECTOR(dim=2),
               type_=postgresql.ARRAY(sa.DOUBLE_PRECISION(precision=53)),
               existing_nullable=True)
    op.alter_column('motivational_quotes', 'embeddings',
               existing_type=pgvector.sqlalchemy.vector.VECTOR(dim=1536),
               type_=postgresql.ARRAY(sa.DOUBLE_PRECISION(precision=53)),
               existing_nullable=True)
    # ### end Alembic commands ###
