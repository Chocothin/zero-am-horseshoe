# PF-figures: 논문 1 그림 1~3

- `make_figures.py`: `paper/p1/figures/` 에 `fig1-orbits.pdf`, `fig2-graph.pdf`, `fig3-margins.pdf`, `figdata.json` 을 쓴다(`--out` 으로 다른 폴더 지정 가능). 새 CAP 계산은 없다.
- `orbits.py`: 단면 점 → 배위 공간 곡선. 저장된 고정밀 점(hex)에서 float DOP853 으로 다음 syzygy 까지만 적분하고, 비행마다 반사 S 와 회전으로 이어 붙인다. 설명용 수치 적분이다.
- 읽는 자료: 그림 1 `results/2026-10-15-T6-horseshoe/orbit_refit_mp.json`, 그림 2 `results/2026-10-15-T6-horseshoe/graph_design.json`·`results/2026-10-17-T6-horseshoe/coverings_summary.json`(변의 구간 이름), 그림 3 `coverings_summary.json`·`results/2026-10-17-T7-hyperbolic/cones.json`·`theorem_Aprime.json`(μ_s, μ_u).
- `verify.sh`: 임시 폴더에 다시 만들어 PDF 생성과 `figdata.json` 일치를 확인한다(exit 0 = 통과).
- 실행: `.venv/bin/python -I -B tracks/PF-figures/make_figures.py`. 눈으로 점검하려면 `FIG_PREVIEW_PNG=1` 로 PNG 도 만든다.
