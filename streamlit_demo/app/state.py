"""Session-state schema and helpers for the Streamlit demo."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import streamlit as st


@dataclass
class LabelEntry:
    series_key: tuple[str, str]
    window_start: int
    label: str
    confidence: float
    comment: str
    added_at: str


@dataclass
class ModelVersion:
    version: int
    state_dict_key: str  # stored separately because torch tensors aren't json-friendly
    trained_at: str
    labels_used: int
    history: dict


def init_state(initial_state_dict: Any) -> None:
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.labeled_queue = []
    st.session_state.label_history = []
    st.session_state.models = [
        {
            "version": 0,
            "state_dict": initial_state_dict,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "labels_used": 0,
            "history": {"loss": [], "acc": []},
        }
    ]
    st.session_state.active_version = 0
    st.session_state.compare_baseline = 0
    st.session_state.held_out_keys = []


def add_label(entry: LabelEntry) -> None:
    st.session_state.labeled_queue.append(asdict(entry))


def clear_queue() -> None:
    st.session_state.labeled_queue = []


def append_model_version(state_dict: Any, labels_used: int, history: dict) -> int:
    next_version = len(st.session_state.models)
    st.session_state.models.append(
        {
            "version": next_version,
            "state_dict": state_dict,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "labels_used": labels_used,
            "history": history,
        }
    )
    st.session_state.active_version = next_version
    return next_version


def model_state_by_version(version: int) -> Any:
    return st.session_state.models[version]["state_dict"]
