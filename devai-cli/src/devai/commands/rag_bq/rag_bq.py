import click
from devai.commands.rag_bq import load, query

@click.group()
def rag_bq():
    pass


rag_bq.add_command(load.load)
rag_bq.add_command(load.testdb)
rag_bq.add_command(query.query)
