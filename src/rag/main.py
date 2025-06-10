from typer import Typer

from .settings import get_settings

cli = Typer(name="rag", no_args_is_help=True)


@cli.command("start")
def start():
    """Start the RAG server."""
    settings = get_settings()
    print("Starting RAG server with settings:")
    print(settings.model_dump())
    print("Hello from rag!")


def callback():
    """A rag server"""


if __name__ == "__main__":
    cli.callback()(callback)
    cli()
