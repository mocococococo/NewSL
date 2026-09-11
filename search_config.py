"""Select one consistent search profile before starting a Python process."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


SearchMode = Literal["shot", "shot_origin"]
ActionType = Literal["default", "high_resolution"]

# Change only this line to switch the search settings.
SEARCH_MODE: SearchMode = "shot"


@dataclass(frozen=True)
class SearchSettings:
    mode: SearchMode
    action_type: ActionType
    action_count: int
    max_simulations: int
    initial_candidates: int
    base_model_name: str


SEARCH_PROFILES = MappingProxyType({
    "shot": SearchSettings("shot", "default", 1600, 1022, 256,
                           "js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin"),
    "shot_origin": SearchSettings("shot_origin", "high_resolution", 3584, 14320, 3584,
                                  "transformer-supervised-model-AdamW-vy56.bin"),
})


def get_search_settings(mode: SearchMode) -> SearchSettings:
    try:
        return SEARCH_PROFILES[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown SEARCH_MODE {mode!r}; choose 'shot' or 'shot_origin'") from exc


ACTIVE_SEARCH_SETTINGS = get_search_settings(SEARCH_MODE)


def require_search_mode(mode: SearchMode, action_type: ActionType | None = None) -> None:
    """Reject an entry point or action space that belongs to the other profile."""
    settings = ACTIVE_SEARCH_SETTINGS
    if settings.mode != mode:
        raise ValueError(
            f"This entry point uses {mode}, but search_config.py selects {settings.mode}. "
            f"Select SEARCH_MODE={mode!r} or use the {settings.mode} entry point."
        )
    if action_type is not None and action_type != settings.action_type:
        raise ValueError(
            f"{mode} requires action_type={settings.action_type!r}, got {action_type!r}. "
            "Select the complete profile in search_config.py."
        )
