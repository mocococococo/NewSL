import json
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name(
    "remotes.json"
)


def load_remotes():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def get_remote(name):
    remotes = load_remotes()

    if name not in remotes:
        available = ", ".join(
            remotes.keys()
        )

        raise ValueError(
            f"Unknown remote: {name}\n"
            f"Available remotes: {available}"
        )

    return remotes[name]


def get_run_name(
    remotes=None,
):
    if remotes is None:
        remotes = load_remotes()

    run_names = {}

    for (
        name,
        node,
    ) in remotes.items():

        if "log_path" not in node:
            raise ValueError(
                f"{name}: log_path is not configured"
            )

        log_path = (
            str(node["log_path"])
            .replace("\\", "/")
            .rstrip("/")
        )

        if not log_path:
            raise ValueError(
                f"{name}: invalid log_path"
            )

        run_name = (
            log_path
            .rsplit("/", 1)[-1]
        )

        if not run_name:
            raise ValueError(
                f"{name}: could not determine "
                f"run name from log_path"
            )

        run_names[name] = run_name

    unique_run_names = set(
        run_names.values()
    )

    if len(unique_run_names) != 1:

        details = ", ".join(
            f"{name}={run_name}"
            for (
                name,
                run_name,
            ) in run_names.items()
        )

        raise ValueError(
            "All remotes must use the same "
            "log_path final directory name. "
            f"Configured: {details}"
        )

    return next(
        iter(
            unique_run_names
        )
    )


def get_data_root(
    root,
    run_name,
):
    root = Path(root)

    return (
        root
        / "data"
        / "distribute"
        / run_name
    )


def get_model_root(
    root,
    run_name,
):
    root = Path(root)

    return (
        root
        / "model"
        / "distribute"
        / run_name
    )


def get_record_root(
    root,
    run_name,
):
    root = Path(root)

    return (
        root
        / "record"
        / "distribute"
        / run_name
    )


def get_temp_root(
    root,
    run_name,
):
    root = Path(root)

    base = (
        root
        / ".temp"
        / "remote_generate"
    )

    if run_name == "all":
        return base

    return (
        base
        / run_name
    )