"""exp04 평가 재현 — baseline·gSDE·휴리스틱을 한 격자에서 평가해 eval_results.pkl 생성.

세 그룹을 동일 (거리 × 각도 × 지형 seed) 격자에서 돌린다:
  - baseline : exp02 seed1~5 (3M)
  - gSDE     : exp04 seed1~5 (3M)
  - heuristic: BC의 무학습 expert(goal 방향 조향+전진) — 상한 참조

기본값으로 실행하면 커밋된 eval_results.pkl과 동일한 결과를 재생성한다(전체 ~45분).

  python analysis/exp04_gsde/run_eval.py
  python analysis/exp04_gsde/run_eval.py --dists 3 --angle-step 90 --seeds 1 --out /tmp/smoke.pkl  # 빠른 확인
"""
import argparse
import sys
from glob import glob
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from policy_eval import PolicyEvaluator, save_results   # noqa: E402
from bc.collect_demos import heuristic                  # noqa: E402

# 그룹별 3M 정책 zip (vecnormalize .pkl은 _steps.zip 패턴에 안 걸림). exp04-00은 2M라 제외.
BASELINE = sorted(glob(str(ROOT / "experiments/exp02-seed-variance/*/ckpt/*_3000000_steps.zip")))
GSDE     = sorted(glob(str(ROOT / "experiments/exp04-explore/exp04-0[1-5]-*/ckpt/*_3000000_steps.zip")))


def run_name(zip_path):
    return Path(zip_path).parent.parent.name          # .../<run>/ckpt/<file>.zip → <run>


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dists", nargs="+", type=float, default=[2, 3, 4, 5])
    ap.add_argument("--angle-step", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=100)
    ap.add_argument("--out", type=str, default=str(HERE / "eval_results.pkl"))
    args = ap.parse_args()

    # 이름(그룹#run)→정책. baseline/gSDE는 zip 경로, heuristic은 action_fn 직접.
    policies, group_names = {}, {"baseline": [], "gSDE": [], "heuristic": []}
    for label, paths in [("baseline", BASELINE), ("gSDE", GSDE)]:
        for p in paths:
            name = f"{label}#{run_name(p)}"
            policies[name] = p
            group_names[label].append(name)
    policies["heuristic#heuristic"] = heuristic        # 무학습 규칙(obs만으로 action)
    group_names["heuristic"] = ["heuristic#heuristic"]

    print(f"baseline {len(BASELINE)} · gSDE {len(GSDE)} · heuristic 1  "
          f"| dists={args.dists} step={args.angle_step}° seeds={args.seeds}@{args.seed_start}")
    ev = PolicyEvaluator(dists=args.dists, angle_step=args.angle_step)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    results = ev.evaluate(policies, seeds)
    ev.close()

    save_results(args.out, results, group_names, ev.dists, ev.angles, seeds)
    print(f"→ 저장: {args.out}")


if __name__ == "__main__":
    main()
