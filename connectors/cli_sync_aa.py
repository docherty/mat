"""CLI: cache Artificial Analysis model catalog."""

from connectors.import_aa import fetch_all_models


def main() -> None:
    models = fetch_all_models()
    print(f"cached {len(models)} models")


if __name__ == "__main__":
    main()
