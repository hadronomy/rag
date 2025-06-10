from typer import Typer

from settings import get_settings

app = Typer(name="rag", no_args_is_help=True)


@app.command("start")
def start():
    """Start the RAG server."""
    settings = get_settings()
    print("Starting RAG server with settings:")
    print(settings.model_dump())
    print("Hello from rag!")


def callback():
    """A rag server"""


if __name__ == "__main__":
    app.callback()(callback)
    app()
