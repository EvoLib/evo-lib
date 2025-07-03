from pathlib import Path


def touch_init(directory: Path) -> None:
    """Erzeugt leere __init__.py-Dateien falls noch nicht vorhanden."""
    init_path = directory / "__init__.py"
    if not init_path.exists():
        init_path.write_text("# Automatically created to mark package\n")


def walk_and_touch(base_dir: Path) -> None:
    for subdir in base_dir.rglob("*"):
        if subdir.is_dir():
            touch_init(subdir)


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evolib")
    walk_and_touch(root)
    print("Empty __init__.py files created where missing.")
