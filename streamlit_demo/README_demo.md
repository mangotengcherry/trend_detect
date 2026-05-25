# Trend Detection Streamlit Demo

`../presentation_claude_review1.md`의 *multi-channel 1D-CNN + attention + active learning* 흐름을 클릭으로 체험하는 데모입니다. 데이터는 repo 루트의 `../data/` CSV를 자동으로 찾아 사용합니다.

## 실행 (repo 루트에서)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r streamlit_demo/requirements.txt
streamlit run streamlit_demo/app/streamlit_app.py
```

앱이 처음 기동될 때 `streamlit_demo/artifacts/model_v0.pt`가 없으면 자동으로 30초간 초기 학습 후 저장합니다. 미리 학습만 하려면:

```bash
PYTHONPATH=streamlit_demo python -m src.model.cnn --train
```

## 실행 (streamlit_demo 내부에서)

```bash
cd streamlit_demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.model.cnn --train
streamlit run app/streamlit_app.py
```

## 사용 흐름

1. **📈 Trend & Window** — Sidebar에서 sensor·equipment 선택 → 30일치 stitched trend가 표시됩니다. Window 슬라이더를 움직이면 30포인트 윈도우에 대한 예측·attention heatmap·12채널 feature·hard-rule trigger·channel occlusion 중요도가 갱신됩니다.
2. **🏷️ Labeling** — 현재 window를 보고 정상/이상 8종 중 하나로 라벨을 부여하여 queue에 적재합니다.
3. **🤖 Active Learning & Diff** — queue가 3건 이상이면 "Apply Active Learning" 버튼이 활성화됩니다. 클릭 시 이전 weights를 warm-start로 fine-tune하여 새 모델 버전을 만들고, held-out 5개 series에 대해 학습 전/후 prediction을 side-by-side로 비교합니다.

## 구성

```
streamlit_demo/
├── src/data_pipeline.py   # load, stitch(120 pts, 6h), windowing, 12-channel feature, hard rule
├── src/model/cnn.py       # MultiKernel 1D-CNN + AttentionPool, train/fine_tune/occlusion
├── src/model/training.py  # focal loss, replay buffer, seeding
├── app/streamlit_app.py   # 메인 UI (3 tab)
├── app/charts.py          # Plotly 차트 helpers
├── app/state.py           # session_state 스키마
├── artifacts/model_v0.pt  # 사전 학습된 초기 모델 (앱이 자동 갱신 가능)
└── requirements.txt       # streamlit, torch, pandas, plotly, scipy, numpy
```
