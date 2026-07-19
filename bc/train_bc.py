"""시연(npz)으로 BC 정책 학습 (지도학습: obs → action MSE 회귀).

  python bc/train_bc.py --demos bc/demos_heuristic.npz --out bc/bc_policy.pt

여러 시연 파일을 합칠 수 있다 (휴리스틱 + 사람 3점 회전):
  python bc/train_bc.py --demos bc/demos_heuristic.npz bc/demos_3pt.npz --out bc/bc_policy.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bc.bc_model import BCPolicy, save_bc


def load_demos(paths):
    """여러 npz를 합치되, 에피소드 ID를 파일 간 겹치지 않게 이어붙인다(누수 없는 분할용)."""
    obs, act, ep_id = [], [], []
    offset = 0
    for p in paths:
        d = np.load(p)
        obs.append(d["obs"]); act.append(d["act"])
        eid = d["ep_id"] + offset            # 파일마다 ID 오프셋 → 전역 고유
        ep_id.append(eid)
        offset = int(eid.max()) + 1
        print(f"  {p}: {len(d['obs'])} transition")
    return np.concatenate(obs), np.concatenate(act), np.concatenate(ep_id)


def main():
    p = argparse.ArgumentParser(description="BC 학습")
    p.add_argument("--demos",  type=str, nargs="+", default=["bc/demos_heuristic.npz"])
    p.add_argument("--out",    type=str, default="bc/bc_policy.pt")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch",  type=int, default=256)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed",   type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("시연 로드:")
    obs, act, ep_id = load_demos(args.demos)
    print(f"  합계 {len(obs)} transition, {len(np.unique(ep_id))} 에피소드")

    # train/val 분할 — 에피소드 단위 (한 궤적을 통째로 한쪽에 → 인접 프레임 누수 방지)
    eps = np.unique(ep_id); np.random.shuffle(eps)
    n_val_ep = max(1, int(len(eps) * args.val_frac))
    val_eps = set(eps[:n_val_ep].tolist())
    is_val = np.array([e in val_eps for e in ep_id])
    ti, vi = ~is_val, is_val

    # obs 표준화 통계는 train만으로 계산 (val 통계가 새면 그것도 누수) → 모델과 함께 저장
    obs_mean = obs[ti].mean(0); obs_std = obs[ti].std(0)
    denom = np.where(obs_std < 1e-6, 1.0, obs_std)
    obs_n = (obs - obs_mean) / denom

    Xtr = torch.tensor(obs_n[ti], dtype=torch.float32)
    Ytr = torch.tensor(act[ti],   dtype=torch.float32)
    Xva = torch.tensor(obs_n[vi], dtype=torch.float32).to(device)
    Yva = torch.tensor(act[vi],   dtype=torch.float32).to(device)
    print(f"  train {ti.sum()} / val {vi.sum()} transition (에피소드 {len(eps)-n_val_ep}/{n_val_ep})")

    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=args.batch, shuffle=True)
    model = BCPolicy(obs_dim=obs.shape[1], act_dim=act.shape[1], hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for ep in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva), Yva).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  val MSE {vloss:.5f}")

    model.load_state_dict(best_state)                    # 최저 val 모델 저장
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_bc(args.out, model, obs_mean, obs_std, args.hidden)
    print(f"\n최저 val MSE {best_val:.5f}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
