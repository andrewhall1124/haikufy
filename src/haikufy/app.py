from textual.app import App, ComposeResult
from textual.widgets import Input, Static


class HaikufyApp(App):
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type something here...")
        yield Static("hello, world!")


def main():
    app = HaikufyApp()
    app.run()


if __name__ == "__main__":
    main()
