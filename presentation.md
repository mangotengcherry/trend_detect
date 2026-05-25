# 반도체 공정 Trend 이상 감지 고도화 제안서

> 목적: 기존 3σ 단일 기준 기반 모니터링을 **Action-worthy Trend Detection 체계**로 전환하여, 엔지니어 알람 피로도를 줄이고 공정 이상 대응의 일관성과 설명 가능성을 높인다.

---

## 1. Executive Summary

### 결론

현행 3σ 기반 알람은 단일 포인트 이탈에는 반응하지만, 실제 엔지니어 판정 기준인 **연속 이탈, 점진 drift, level shift, 상/하한 비대칭, 측정 오류 구분**을 충분히 반영하지 못한다.  
따라서 본 제안은 단순 딥러닝 도입이 아니라, 아래와 같은 **하이브리드 지능형 모니터링 체계**로 전환하는 것을 목표로 한다.

```mermaid
flowchart LR
    A["Raw 계측/센서 Trend"] --> B["Robust Normalization"]
    B --> C["SPC Rule Layer"]
    C --> D["Feature Channel 생성"]
    D --> E["Multi-kernel 1D-CNN"]
    E --> F["Attention Pooling"]
    F --> G["Anomaly Type / Score / Action Level"]
    G --> H["Explanation Module"]
    H --> I["Engineer Feedback"]
    I --> D
```

### 핵심 제안

| 구분 | 기존 방식 | 제안 방식 |
|---|---|---|
| 판단 기준 | target ± 3σ 중심 | SPC rule + 1D-CNN 기반 trend pattern 판단 |
| 감지 대상 | 단일 포인트 이탈 | spike, persistent high/low, drift, level shift, variance 증가, measurement error |
| 상/하한 차이 | 동일 기준 적용 | upper/lower score를 분리하여 비대칭 관리 |
| 설명 가능성 | 알람 발생 여부 중심 | 중요 시점, 중요 feature group, 이상 유형, 권장 확인 항목 제공 |
| 운영 방식 | 엔지니어 수동 확인 중심 | 알람 우선순위화 + feedback learning |

### 기대 효과

| 기대 효과 | PoC 검증 목표 |
|---|---:|
| 불필요 알람 감소 | 기존 3σ 대비 false positive 40~60% 감소 |
| Critical miss 방지 | 기존 rule 대비 critical miss 동등 이하 유지 |
| 대응 우선순위화 | action-worthy alert precision 2배 개선 목표 |
| 판정 기준 표준화 | 엔지니어별 눈높이 차이를 label/rule/threshold로 구조화 |
| 확장성 확보 | item별 개별 모델이 아닌 공통 모델 + item별 calibration 체계 |

---

## 2. 문제 정의: 왜 3σ 방식만으로는 부족한가

### 현재 Pain Point

- 관리 대상 item 수 증가로 엔지니어 육안 판정 부담 증가
- 3σ 단일 기준 적용 후에도 false positive가 많아 알람 피로도 지속
- 실제 엔지니어 판정 기준은 단일 포인트 이탈보다 복합적임
  - 한 포인트 spike는 생산성 관점에서 무시하는 경우 존재
  - 2~3포인트 이상 연속 이탈 시에만 조치하는 경우 존재
  - 점진적 상승/하락 drift가 중요할 수 있음
  - 상한 방향과 하한 방향의 위험도가 다른 item 존재
  - 측정 오류와 실제 공정 이상을 구분해야 함

### 핵심 진단

> 현재 문제의 본질은 “3σ threshold가 낮다/높다”가 아니라, **엔지니어의 암묵적 trend 판정 기준이 시스템에 구조화되어 있지 않다는 점**이다.

---

## 3. 학술적·기술적 근거

### 3.1 SPC 관점: Shewhart 3σ는 큰 이탈에는 강하지만 작은 drift에는 약하다

- NIST/Sematech Engineering Statistics Handbook은 Shewhart 방식이 최신 측정값과 control limit 이탈 여부에 크게 의존한다고 설명한다.
- 같은 문헌에서 EWMA는 과거 데이터를 지수 가중 평균하여 **small or gradual drift**에 민감하게 설계할 수 있다고 설명한다.
- CUSUM chart는 평균의 작은 shift, 특히 2σ 이하 shift를 감지할 때 Shewhart 방식보다 효율적이라고 설명된다.

**시사점:** 3σ rule을 폐기하기보다는, hard rule로 유지하되 EWMA/CUSUM/연속 이탈 rule을 feature 또는 rule layer로 반영해야 한다.

### 3.2 Time-series Deep Learning 관점: 1D Convolution은 trend shape 감지에 적합하다

- InceptionTime 연구는 time-series classification에서 convolution 기반 deep architecture가 높은 정확도와 확장성을 동시에 달성할 수 있음을 보였다.
- MiniROCKET 연구는 random convolutional kernels 기반 time-series transform이 높은 정확도 대비 계산 비용을 크게 낮출 수 있음을 보였다.

**시사점:** 최근 30포인트 trend처럼 짧은 sequence에서는 RNN보다 1D-CNN 또는 convolutional time-series classifier가 더 단순하고 운영 친화적일 가능성이 높다.

### 3.3 Explainability 관점: Attention은 설명 단서이지 원인 자체는 아니다

- Attention mechanism은 입력의 어느 구간을 많이 참고했는지 보여줄 수 있다.
- 하지만 “Attention is not Explanation” 계열 연구는 attention weight만으로 모델 판단의 완전한 원인 설명이라고 보기 어렵다고 지적한다.

**시사점:** Attention heatmap은 엔지니어 설명 UI에 유용하지만, channel-wise occlusion, time-wise occlusion, rule trigger를 함께 제공해야 한다.

---

## 4. 제안 Architecture

### 4.1 전체 구조

```mermaid
flowchart TB
    A["Trend Data\n계측/센서 value, target, spec, context"] --> B["Context-aware Normalization\nitem/product/equipment/recipe 기준 보정"]
    B --> C["Hard Rule Filter\n5σ, spec hard violation, invalid reading"]
    B --> D["Feature Channel Generator\nvalue, delta, slope, EWMA, upper/lower score, recency"]
    D --> E["Multi-kernel 1D-CNN\nK=3 spike, K=5 short drift, K=9 level shift"]
    E --> F["Attention Pooling\n중요 time range 산출"]
    F --> G["Prediction Heads\nanomaly type, probability, action level"]
    C --> H["Alarm Policy Layer\nrule + ML score + item threshold"]
    G --> H
    H --> I["Engineer UI\ntrend, score, reason, recommended check"]
    I --> J["Feedback Store\naccept/reject/defer/measurement issue"]
    J --> K["Active Learning\nlabel refinement & threshold calibration"]
    K --> D
```

### 4.2 Feature Channel 설계

1D-CNN 입력은 단순 raw sequence가 아니라, 엔지니어가 trend를 볼 때 사용하는 관점을 channel로 구조화한 형태이다.

```text
Input shape = [time=30, channels=8~10]
```

| Channel | 의미 | 기대 역할 |
|---|---|---|
| `value_norm` | target 대비 robust normalized value | 기본 level 판단 |
| `delta` | 직전 point 대비 변화량 | spike, jump 감지 |
| `rolling_slope` | 최근 k개 point 기울기 | gradual up/down drift 감지 |
| `ewma` | 지수 가중 이동 평균 | 완만한 평균 이동 감지 |
| `upper_score` | 상한 방향 위험도 | 상한 비대칭 관리 |
| `lower_score` | 하한 방향 위험도 | 하한 비대칭 관리 |
| `position` | window 내 절대 위치 | 오래된 이상과 최근 이상 구분 |
| `recency` | 최근성 가중 | 현재 action 필요성 반영 |
| `missing_flag` | 결측/invalid 여부 | 측정 오류 분리 |
| `time_gap` | sampling gap | irregular sampling 보정 |

### 4.3 Multi-kernel 1D-CNN의 역할

```mermaid
flowchart LR
    A["30-point Multi-channel Trend"] --> B1["Conv1D K=3\nspike / 2~3 point 이탈"]
    A --> B2["Conv1D K=5\nshort drift"]
    A --> B3["Conv1D K=9\nlevel shift / 완만한 변화"]
    B1 --> C["Feature Fusion"]
    B2 --> C
    B3 --> C
    C --> D["Attention Pooling"]
    D --> E["Anomaly Type + Action Level"]
```

**왜 RNN보다 우선 검토하는가?**

- window length가 30으로 짧아 long-term memory보다 local pattern 감지가 중요함
- convolution은 병렬 계산이 가능해 item 수가 많을 때 운영 비용이 낮음
- kernel size별로 spike, drift, level shift를 분리 학습할 수 있음
- feature channel과 결합하면 엔지니어 설명 가능성이 RNN보다 직관적임

---

## 5. 이상 유형 Taxonomy

단순 정상/이상 binary label은 현장 판단을 과도하게 단순화한다. 아래와 같이 anomaly type을 분리해야 false positive를 줄이고 action 우선순위를 정할 수 있다.

| Type | 설명 | Action Policy |
|---|---|---|
| `normal` | target 주변 정상 노이즈 | no alarm |
| `single_spike_ignore` | 단발성 spike 후 정상 복귀 | suppress 또는 low priority |
| `critical_spike` | 5σ 또는 hard spec 초과 | immediate alarm |
| `persistent_high` | 상한 방향 연속 이탈 | alarm |
| `persistent_low` | 하한 방향 연속 이탈 | alarm |
| `gradual_up_drift` | 점진 상승 drift | warning 또는 monitoring 강화 |
| `gradual_down_drift` | 점진 하락 drift | warning 또는 monitoring 강화 |
| `level_shift` | baseline 급격 이동 | alarm |
| `variance_increase` | 산포 증가 | warning |
| `measurement_error` | sensor dropout/out-of-range | 공정 이상과 분리, measurement check |

현재 repository의 synthetic dataset은 위 10개 유형을 기준으로 구성되어 있으며, 각 유형별 40개 series, 총 400개 series/12,000개 point를 제공한다.

![Synthetic SPC trend scatter chart](assets/spc_trend_scatter.svg)

---

## 6. 설명 가능성: 엔지니어가 납득할 수 있는 알람으로 전환

### 6.1 모델 설명의 목표

모델의 목표는 “AI가 이상이라고 했다”가 아니라, 엔지니어가 다음 질문에 답할 수 있게 만드는 것이다.

1. 어느 시점 때문에 알람이 발생했는가?
2. 어떤 trend 관점이 중요했는가?
3. 단발 spike인지, drift인지, persistent 이탈인지?
4. 지금 action이 필요한가, monitoring만 하면 되는가?
5. 측정 오류 가능성은 없는가?

### 6.2 Explanation Payload

```mermaid
flowchart LR
    A["Prediction"] --> B["Attention Heatmap\n중요 time range"]
    A --> C["Channel Attribution\n중요 feature group"]
    A --> D["Rule Trigger\n5σ, CUSUM, EWMA 등"]
    B --> E["Explanation Template"]
    C --> E
    D --> E
    E --> F["Engineer UI"]
```

### 6.3 설명 예시

> 판정: `gradual_up_drift` / Action Level: Warning  
> 주요 근거: 최근 5포인트에서 `rolling_slope`, `EWMA`, `upper_score`가 동시에 증가했습니다.  
> 중요 구간: t26~t30  
> 해석: 단발 spike보다는 상한 방향 drift가 진행 중인 패턴입니다. 동일 설비/recipe의 최근 PM 및 neighbor wafer trend 확인을 권장합니다.

---

## 7. 운영 설계: 공통 모델 + item별 calibration

관리 item마다 단위, target, sigma, 상/하한 중요도가 다르기 때문에 item별 개별 모델을 만드는 방식은 장기 운영에 부적합하다. 대신 아래 구조를 제안한다.

```mermaid
flowchart LR
    A["다수 Item / Sensor"] --> B["공통 정규화 체계"]
    B --> C["공통 1D-CNN 모델"]
    C --> D["공통 anomaly type score"]
    D --> E["Item별 threshold calibration"]
    E --> F["Alarm policy"]
```

### 장점

- item이 늘어나도 모델 수가 선형적으로 증가하지 않음
- 신규 item은 target/sigma/spec만 있으면 cold-start 가능
- item별 민감도는 threshold와 calibration table로 관리 가능
- 공통 anomaly taxonomy를 통해 엔지니어 간 판정 기준 표준화 가능

---

## 8. PoC 설계

### 8.1 비교 대상

| 실험군 | 방식 | 목적 |
|---|---|---|
| A | 현행 3σ rule | baseline |
| B | SPC enhanced rule | 연속 이탈, EWMA, CUSUM 추가 효과 확인 |
| C | Tabular model | window summary feature + CatBoost/LightGBM |
| D | Raw 1D-CNN | sequence shape 학습력 확인 |
| E | Multi-channel 1D-CNN | feature channel 효과 확인 |
| F | Multi-channel 1D-CNN + Attention | 설명 가능성과 최근성 반영 효과 확인 |

### 8.2 핵심 KPI

| KPI | 정의 | 목표 |
|---|---|---:|
| False positive / item / week | 엔지니어 reject 알람 수 | 기존 대비 40~60% 감소 |
| Action-worthy precision | 엔지니어 accept 알람 비율 | 기존 대비 2배 개선 |
| Critical miss count | critical event 미검출 수 | 기존 rule 이하 유지 |
| Detection delay | 이상 시작 후 알람까지 point 수 | drift 유형에서 단축 |
| Duplicate suppression rate | 동일 root-cause 중복 알람 묶기 | 운영 정책 수립 |
| Explanation acceptance | 설명이 판단에 도움된 비율 | pilot engineer feedback으로 측정 |

### 8.3 PoC Roadmap

```mermaid
gantt
    title 8주 PoC 추진 일정
    dateFormat  YYYY-MM-DD
    section Phase 0: 기준 정리
    알람 유형 taxonomy 확정           :a1, 2026-06-01, 7d
    label 정책 및 feedback schema 정의 :a2, after a1, 7d
    section Phase 1: Baseline
    3σ / SPC enhanced rule 구현        :b1, after a2, 7d
    tabular baseline 구현              :b2, after b1, 7d
    section Phase 2: Model
    multi-channel feature generator 구현 :c1, after b2, 7d
    1D-CNN + attention 학습/평가        :c2, after c1, 14d
    section Phase 3: 운영 검증
    explanation UI prototype           :d1, after c2, 7d
    KPI review 및 pilot decision        :d2, after d1, 7d
```

---

## 9. 리스크와 대응 방안

| 리스크 | 내용 | 대응 방안 |
|---|---|---|
| Label inconsistency | 엔지니어별 판정 기준 차이 | taxonomy 기반 label guide, disagreement review |
| False negative | critical event 미검출 | hard rule layer 유지, critical class 가중치 부여 |
| False positive 재발 | 모델 score만으로 알람 시 피로도 지속 | item별 threshold, suppression, action policy layer 적용 |
| Attention 오해 | attention을 원인으로 과해석 | occlusion/gradient/rule trigger와 함께 설명 |
| Data leakage | 동일 series/window가 train-test에 중복 | series/time/product/equipment 기준 split 엄격 적용 |
| Drift over time | 공정 조건 변경으로 모델 성능 저하 | feedback loop, monthly calibration, champion-challenger 운영 |
| 운영 복잡도 | feature/version/model 관리 필요 | feature store schema와 model card 관리 |

---

## 10. 의사결정 요청 사항

### 요청 1. PoC 추진 승인

- 범위: synthetic data + 제한된 실제 비식별 trend sample 기반 검증
- 기간: 8주
- 산출물: baseline 비교 리포트, 모델 prototype, explanation UI mock, KPI review

### 요청 2. 현업 label review 리소스 확보

- 최소 2~3명의 experienced engineer가 주 1회 label disagreement review 참여
- accept/reject/defer/measurement issue feedback을 표준화

### 요청 3. Pilot 적용 기준 합의

Pilot 전환 기준은 모델 정확도가 아니라 아래 운영 KPI로 판단한다.

```text
1. 기존 3σ 대비 false positive 40% 이상 감소
2. critical miss count 기존 rule 이하 유지
3. engineer accept rate 유의미 개선
4. alarm explanation이 현업 판단에 도움된다는 feedback 확보
```

---

## 11. 최종 메시지

본 과제는 단순히 RNN 또는 CNN을 도입하는 모델링 과제가 아니다.  
핵심은 엔지니어가 눈으로 보던 trend 판정 기준을 **feature channel, SPC rule, anomaly taxonomy, feedback learning**으로 구조화하는 것이다.

> 제안 방향은 “AI가 알람을 대신 내는 시스템”이 아니라,  
> **불필요 알람은 줄이고, 조치 가능한 이상은 설명과 함께 우선순위화하는 공정 모니터링 의사결정 시스템**이다.

---

## References

1. NIST/Sematech Engineering Statistics Handbook, [EWMA Control Charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc314.htm): EWMA는 과거 관측값을 지수 가중 평균하여 small/gradual drift에 민감하게 설계 가능.
2. NIST/Sematech Engineering Statistics Handbook, [CUSUM Control Charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc313.htm): CUSUM은 process mean의 작은 shift, 특히 2σ 이하 shift 감지에 효율적.
3. Ismail Fawaz et al. (2019), [InceptionTime: Finding AlexNet for Time Series Classification](https://arxiv.org/abs/1909.04939): convolution 기반 time-series classification architecture의 정확도와 확장성 근거.
4. Dempster et al. (2020), [MINIROCKET: A Very Fast Almost Deterministic Transform for Time Series Classification](https://arxiv.org/abs/2012.08791): convolutional kernel 기반 time-series transform의 계산 효율성 근거.
5. Jain & Wallace (2019), [Attention is not Explanation](https://arxiv.org/abs/1902.10186): attention weight를 단독 설명 근거로 과신하면 안 된다는 해석 리스크 근거.
