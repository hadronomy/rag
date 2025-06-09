from typer import Typer

app = Typer(name="rag", no_args_is_help=True)


@app.command("start")
def start():
    """Start the RAG server."""
    print("Hello from rag!")


def callback():
    """Empty callback function."""


if __name__ == "__main__":
    app.callback()(callback)
    app()
