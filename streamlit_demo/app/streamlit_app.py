"""Streamlit demo: trend anomaly detection + attention + active learning."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import charts, state as app_state  # noqa: E402
from src.data_pipeline import (  # noqa: E402
    CHANNEL_NAMES,
    CLASSES,
    CLASS_TO_IDX,
    NUM_CHANNELS,
    NUM_CLASSES,
    WINDOW,
    build_long_series,
    build_training_set,
    feature_channels,
    hard_rule_trigger,
    list_long_series_keys,
    load_raw,
    window_label,
)
from src.model.cnn import (  # noqa: E402
    ARTIFACTS_DIR,
    MODEL_V0_PATH,
    MultiKernelCNN,
    channel_occlusion,
    fine_tune,
    load_snapshot,
    predict_with_attn,
    snapshot,
    train_initial,
)
from src.model.training import ReplayBuffer  # noqa: E402

st.set_page_config(page_title="Trend Detection Demo", layout="wide")


@st.cache_resource(show_spinner=False)
def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_raw()


@st.cache_resource(show_spinner=True)
def load_initial_model() -> dict:
    measurements, metadata = load_dataframes()
    X, y, _ = build_training_set(measurements, metadata)
    replay = ReplayBuffer(X=X.astype(np.float32), y=y.astype(np.int64))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_V0_PATH.exists():
        ckpt = torch.load(MODEL_V0_PATH, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        history = ckpt.get("history", {"loss": [], "acc": []})
    else:
        with st.spinner("초기 모델 학습 중 (약 30초)..."):
            model, history = train_initial(X, y, epochs=30, lr=1e-3)
            state_dict = snapshot(model)
            torch.save(
                {"state_dict": state_dict, "history": history, "classes": list(CLASSES)},
                MODEL_V0_PATH,
            )
    return {"state_dict": state_dict, "history": history, "replay": replay, "X": X, "y": y}


@st.cache_resource(show_spinner=False)
def long_series_cached(sensor: str, equipment: str) -> pd.DataFrame:
    measurements, metadata = load_dataframes()
    return build_long_series(sensor, equipment, n_segments=4, measurements=measurements, metadata=metadata)


@st.cache_data(show_spinner=False)
def features_for_window(sensor: str, equipment: str, start: int) -> np.ndarray:
    series = long_series_cached(sensor, equipment)
    sub = series.iloc[start : start + WINDOW].reset_index(drop=True)
    return feature_channels(sub)


@st.cache_data(show_spinner=False)
def predict_window(_state_id: str, sensor: str, equipment: str, start: int, version: int) -> tuple[np.ndarray, np.ndarray]:
    features = features_for_window(sensor, equipment, start)
    state_dict = st.session_state.models[version]["state_dict"]
    model = load_snapshot(state_dict)
    probs, attn = predict_with_attn(model, features[None, ...])
    return probs[0], attn[0]


@st.cache_data(show_spinner=False)
def occlusion_for_window(_state_id: str, sensor: str, equipment: str, start: int, version: int, target_class: int) -> dict[str, float]:
    features = features_for_window(sensor, equipment, start)
    state_dict = st.session_state.models[version]["state_dict"]
    model = load_snapshot(state_dict)
    return channel_occlusion(model, features, target_class)


def _state_version_id() -> str:
    return f"v{len(st.session_state.models)}-{st.session_state.get('active_version', 0)}"


def _sidebar() -> tuple[str, str, int]:
    st.sidebar.title("Trend Detection Demo")
    metadata = load_dataframes()[1]
    keys = list_long_series_keys(metadata)
    sensors = sorted({s for s, _ in keys})
    sensor = st.sidebar.selectbox("Sensor", sensors, index=0)
    equipments = sorted({e for s, e in keys if s == sensor})
    equipment = st.sidebar.selectbox("Equipment", equipments, index=0)

    versions = [m["version"] for m in st.session_state.models]
    active = st.sidebar.selectbox(
        "Model version", versions, index=len(versions) - 1,
        format_func=lambda v: f"v{v}" + (" (initial)" if v == 0 else " (after AL)"),
    )
    st.session_state.active_version = int(active)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"label queue: {len(st.session_state.labeled_queue)} 건")
    st.sidebar.caption(f"전체 model versions: {len(st.session_state.models)}")
    st.sidebar.caption(f"누적 labels: {len(st.session_state.label_history)}")
    return sensor, equipment, int(active)


def _tab_trend(sensor: str, equipment: str, version: int) -> None:
    series = long_series_cached(sensor, equipment)
    n = len(series)
    max_start = max(0, n - WINDOW)

    queue_key = f"window_start__{sensor}__{equipment}"
    if queue_key not in st.session_state:
        st.session_state[queue_key] = 0
    window_start = st.slider(
        f"Window start (0 ~ {max_start}, window={WINDOW} 포인트)",
        min_value=0, max_value=int(max_start), value=int(st.session_state[queue_key]),
        key=f"slider_{queue_key}",
    )
    st.session_state[queue_key] = window_start

    st.plotly_chart(
        charts.trend_line_chart(series, window_start, WINDOW),
        config={"displayModeBar": False}, width="stretch",
    )

    window_df = series.iloc[window_start : window_start + WINDOW].reset_index(drop=True)
    features = features_for_window(sensor, equipment, window_start)
    probs, attn = predict_window(_state_version_id(), sensor, equipment, window_start, version)
    rule = hard_rule_trigger(window_df)

    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    pred_conf = float(probs[pred_idx])
    true_label = window_label(window_df)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader(f"Window detail (start={window_start})")
        st.plotly_chart(
            charts.window_zoom_chart(window_df, attn),
            config={"displayModeBar": False}, width="stretch",
        )
        st.markdown("**Feature channels (12)**")
        st.plotly_chart(
            charts.channel_grid_chart(features),
            config={"displayModeBar": False}, width="stretch",
        )
    with col2:
        st.subheader("Prediction")
        st.metric("Predicted class", pred_class, delta=f"conf {pred_conf:.2f}")
        st.caption(f"Last-5 majority (label): **{true_label}**")
        st.plotly_chart(
            charts.prediction_bar(probs),
            config={"displayModeBar": False}, width="stretch",
        )
        st.markdown("**Hard-rule triggers**")
        bcol = st.columns(4)
        bcol[0].markdown(f"5σ: {'🔴' if rule.five_sigma else '⚪'}")
        bcol[1].markdown(f"Hard spec: {'🔴' if rule.hard_spec else '⚪'}")
        bcol[2].markdown(f"Invalid: {'🟡' if rule.invalid_reading else '⚪'}")
        bcol[3].markdown(f"WE-rule: {'🟠' if rule.we_rule else '⚪'}")
        st.markdown("**Channel occlusion (예측 클래스 기준)**")
        importance = occlusion_for_window(_state_version_id(), sensor, equipment, window_start, version, pred_idx)
        st.plotly_chart(
            charts.occlusion_bar(importance),
            config={"displayModeBar": False}, width="stretch",
        )

    st.session_state["__current_context__"] = {
        "sensor": sensor, "equipment": equipment, "window_start": window_start,
        "pred_class": pred_class, "pred_conf": pred_conf, "true_label": true_label,
    }


def _tab_label() -> None:
    ctx = st.session_state.get("__current_context__")
    if not ctx:
        st.info("먼저 'Trend 보기' 탭에서 window를 선택하세요.")
        return
    st.markdown(
        f"현재: **{ctx['sensor']} / {ctx['equipment']}** | window_start = **{ctx['window_start']}** | "
        f"model = **{ctx['pred_class']}** ({ctx['pred_conf']:.2f}) | last-5 majority = **{ctx['true_label']}**"
    )

    with st.form("label_form", clear_on_submit=True):
        label = st.radio("Engineer 판정", options=list(CLASSES), horizontal=True, index=list(CLASSES).index(ctx["true_label"]) if ctx["true_label"] in CLASSES else 0)
        confidence = st.slider("자신감 (0~1)", 0.0, 1.0, 0.8, 0.05)
        comment = st.text_input("비고 (선택)", "")
        submitted = st.form_submit_button("Add to queue", type="primary")
        if submitted:
            entry = app_state.LabelEntry(
                series_key=(ctx["sensor"], ctx["equipment"]),
                window_start=int(ctx["window_start"]),
                label=str(label),
                confidence=float(confidence),
                comment=comment,
                added_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            app_state.add_label(entry)
            st.success(f"queue에 추가됨 — '{label}'")

    if st.session_state.labeled_queue:
        st.markdown("### Pending queue")
        qdf = pd.DataFrame(st.session_state.labeled_queue)
        qdf["series_key"] = qdf["series_key"].apply(lambda t: f"{t[0]} / {t[1]}")
        st.dataframe(qdf, width="stretch", hide_index=True)
        if st.button("queue 비우기", type="secondary"):
            app_state.clear_queue()
            st.rerun()
    else:
        st.caption("queue 비어 있음. window를 둘러보고 라벨을 추가해 보세요.")


def _tab_active_learning(initial_artifacts: dict) -> None:
    queue = st.session_state.labeled_queue
    st.markdown(f"**Pending labels:** {len(queue)} | replay sample size: 64 | fine-tune epochs: 6")
    can_train = len(queue) >= 3
    disabled_reason = "" if can_train else "queue에 최소 3개 라벨이 필요합니다."
    btn = st.button("Apply Active Learning (fine-tune)", type="primary", disabled=not can_train, help=disabled_reason)
    progress_box = st.empty()

    if btn:
        # build new training tensors
        new_X: list[np.ndarray] = []
        new_y: list[int] = []
        for entry in queue:
            sensor, equipment = entry["series_key"]
            start = int(entry["window_start"])
            new_X.append(features_for_window(sensor, equipment, start))
            new_y.append(CLASS_TO_IDX.get(entry["label"], CLASS_TO_IDX["normal"]))
        new_X_arr = np.stack(new_X, axis=0).astype(np.float32)
        new_y_arr = np.asarray(new_y, dtype=np.int64)

        prev_state = st.session_state.models[st.session_state.active_version]["state_dict"]
        model = load_snapshot(prev_state)
        replay: ReplayBuffer = initial_artifacts["replay"]

        prog = progress_box.progress(0.0, text="fine-tune 시작")
        history: dict[str, list[float]] = {"loss": [], "acc": []}

        def cb(epoch: int, total: int, loss: float, acc: float) -> None:
            history["loss"].append(loss)
            history["acc"].append(acc)
            prog.progress(epoch / total, text=f"epoch {epoch}/{total} | loss={loss:.4f} | acc={acc:.3f}")

        model, _ = fine_tune(model, new_X_arr, new_y_arr, replay=replay, epochs=6, lr=3e-4, replay_size=64, progress=cb)
        new_state = snapshot(model)

        labels_used = len(queue)
        new_version = app_state.append_model_version(new_state, labels_used, history)
        st.session_state.label_history.extend(queue)
        app_state.clear_queue()
        prog.progress(1.0, text=f"완료 — v{new_version} 등록")
        st.success(f"새 모델 버전 v{new_version} 추가. (이 버전을 sidebar에서 선택 가능)")
        # clear caches that depend on model version
        predict_window.clear()
        occlusion_for_window.clear()

    st.markdown("---")
    st.markdown("### Model version history")
    history_rows = [
        {
            "version": m["version"],
            "trained_at": m["trained_at"],
            "labels_used (cumulative new)": m["labels_used"],
            "last loss": (m["history"]["loss"][-1] if m["history"]["loss"] else None),
            "last train acc": (m["history"]["acc"][-1] if m["history"]["acc"] else None),
        }
        for m in st.session_state.models
    ]
    st.dataframe(pd.DataFrame(history_rows), hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("### Before / After 비교 (held-out series)")
    if len(st.session_state.models) < 2:
        st.info("Active learning 1회 이상 수행 후 비교가 가능합니다.")
        return

    versions = [m["version"] for m in st.session_state.models]
    col_a, col_b = st.columns(2)
    base_v = col_a.selectbox("Before", versions, index=0, key="cmp_base")
    new_v = col_b.selectbox("After", versions, index=len(versions) - 1, key="cmp_new")

    metadata = load_dataframes()[1]
    if not st.session_state.held_out_keys:
        keys = list_long_series_keys(metadata)
        rng = np.random.default_rng(7)
        chosen = rng.choice(len(keys), size=min(5, len(keys)), replace=False)
        st.session_state.held_out_keys = [keys[int(i)] for i in chosen]

    rows = []
    base_state = st.session_state.models[int(base_v)]["state_dict"]
    new_state = st.session_state.models[int(new_v)]["state_dict"]
    base_model = load_snapshot(base_state)
    new_model = load_snapshot(new_state)

    for sensor, equipment in st.session_state.held_out_keys:
        series = long_series_cached(sensor, equipment)
        max_start = max(0, len(series) - WINDOW)
        sample_starts = np.linspace(0, max_start, num=3, dtype=int)
        for s in sample_starts:
            feats = features_for_window(sensor, equipment, int(s))
            p0, _ = predict_with_attn(base_model, feats[None, ...])
            p1, _ = predict_with_attn(new_model, feats[None, ...])
            c0 = int(np.argmax(p0[0])); c1 = int(np.argmax(p1[0]))
            rows.append({
                "label": f"{sensor[:14]}/{equipment}@{s}",
                "class_v0": CLASSES[c0], "class_v1": CLASSES[c1],
                "prob_v0": float(p0[0, c0]), "prob_v1": float(p1[0, c1]),
                "changed": c0 != c1,
            })

    rows = sorted(rows, key=lambda r: (not r["changed"], -abs(r["prob_v1"] - r["prob_v0"])))
    cmp_df = pd.DataFrame(rows)
    st.dataframe(
        cmp_df.style.format({"prob_v0": "{:.2f}", "prob_v1": "{:.2f}"})
              .apply(lambda s: ["background-color: #fff2cc" if v else "" for v in cmp_df["changed"]], subset=["class_v1"]),
        hide_index=True, width="stretch",
    )
    st.plotly_chart(charts.comparison_chart(rows), width="stretch", config={"displayModeBar": False})


def main() -> None:
    artifacts = load_initial_model()
    app_state.init_state(artifacts["state_dict"])

    sensor, equipment, version = _sidebar()

    tabs = st.tabs(["📈 Trend & Window", "🏷️ Labeling", "🤖 Active Learning & Diff"])
    with tabs[0]:
        _tab_trend(sensor, equipment, version)
    with tabs[1]:
        _tab_label()
    with tabs[2]:
        _tab_active_learning(artifacts)


if __name__ == "__main__":
    main()
