# LoL i.i.d. 상대가치 분석

Bo3·Bo5의 각 세트가 독립이고 A의 세트 승률이 항상 같은 `p`라는 조건에서 매치 승리,
총 세트 수, 세트 핸디캡 가격을 비교하는 오프라인 연구 도구입니다. 실제 주문·시세 수집·프록시
기능은 없습니다.

## 판단의 의미

- **모형상 상대가치 기회**: 기준 시장에서 역산한 `p` 구간 전체에서 대상 상품의 기대 지급액이
  실제 매수비용과 안전마진보다 큽니다. 이는 i.i.d. 가정과 기준 시장 가격이 유효하다는 조건부 판단입니다.
- **지급구조 차익거래**: 확률 가정 없이 모든 정상 최종 스코어에서 순손익이 양수인 매수 조합입니다.

두 판정은 별도로 출력합니다. `YES`와 `NO` 매수만 사용하며 공매도는 가정하지 않습니다.

## 설치와 실행

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
iid-rv analyze examples/analysis.json
iid-rv analyze examples/analysis.json --format json
```

입력 가격과 확률은 `0~1`, `depth`는 주, `bankroll`은 달러입니다. 기준 시장은 기본적으로
단조 증가해 역산해가 유일한 `match`를 사용합니다. 총 세트 수처럼 복수해가 생기는 시장을
기준으로 쓰면 후보를 자동 선택하지 않고 `indeterminate`를 반환합니다.

`fees.rate`의 연구용 주당 비용은 `rate * price * (1-price)`이고 `flat_per_share`를 더할 수
있습니다. 실제 거래소 비용은 실행 시점의 공식 규칙을 입력해야 합니다.

## Python API

```python
from iid_relative_value import Market, SeriesState, invert_anchor

state = SeriesState(3)
anchor = invert_anchor(.64, .66, Market("match"), state)
print(anchor.p, anchor.p_intervals)
```

주요 API는 `invert_anchor`, `terminal_distribution`, `price_markets`,
`analyze_relative_value`, `find_payoff_arbitrage`, `optimize_portfolio_kelly`입니다.
Kelly는 후보 상품의 최종 스코어별 결합 손익을 한 번에 최적화하고 기본적으로 전체 Kelly의
25%만 적용합니다. 확률 추정 오류와 미체결 위험까지 제거해 주는 값은 아닙니다.
