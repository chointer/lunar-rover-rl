"""BC 정책 모델 정의 + 저장/로드 (train_bc·eval_bc가 공유).

obs(65) → action(2: throttle, steer) 회귀 MLP.
학습 때 obs를 표준화(mean/std)하므로, 그 통계를 모델과 함께 저장한다.
없으면 평가 때 다른 스케일의 obs가 들어가 엉뚱하게 움직인다 (RL의 VecNormalize와 같은 이유).

action은 표준화하지 않는다: 휴리스틱 throttle이 항상 +1이라 std≈0 → 0으로 나눔 방지.
(사람 3점 회전 시연을 합치면 throttle에 후진(-1)이 섞여 더 이상 상수가 아니게 됨.)
"""
import numpy as np
import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=65, act_dim=2, hidden=(256, 256)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, act_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def save_bc(path, model, obs_mean, obs_std, hidden):
    torch.save({
        "state_dict": model.state_dict(),
        "obs_mean": np.asarray(obs_mean, np.float32),
        "obs_std":  np.asarray(obs_std,  np.float32),
        "hidden":   tuple(hidden),
    }, path)


def load_bc(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    obs_dim = len(ckpt["obs_mean"])
    model = BCPolicy(obs_dim=obs_dim, hidden=ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["obs_mean"], ckpt["obs_std"]


def make_bc_action_fn(path, device="cpu"):
    """sim_viewer/eval에서 쓰는 action_fn(obs)->action 반환 (obs 표준화 포함)."""
    model, mean, std = load_bc(path, device)
    std = np.where(std < 1e-6, 1.0, std)     # 분산 0 차원 보호

    def action_fn(obs):
        x = ((np.asarray(obs, np.float32) - mean) / std)
        with torch.no_grad():
            a = model(torch.from_numpy(x).float().to(device)).cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)
    return action_fn
