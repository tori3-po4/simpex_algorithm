import torch
import matplotlib.pyplot as plt
from itertools import combinations


def _get_device():
    """利用可能な最適デバイスを返す (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Simplex():
    def __init__(self, objective, constraints, maximize=True, device=None):
        """
        Args:
            objective: 目的関数の係数ベクトル (n,)
            constraints: タプル (A, b)。A @ x <= b の形式。A は (m, n)、b は (m,)
            maximize: True なら最大化、False なら最小化
            device: 使用デバイス。None なら自動検出
        """
        self.device = device or _get_device()
        self.c = objective.float().to(self.device)
        self.A_orig = constraints[0].float().to(self.device)
        self.b_orig = constraints[1].float().to(self.device)
        self.maximize = maximize
        self.n = self.c.shape[0]
        self.m_orig = self.A_orig.shape[0]

        self._convert_to_standard_form()

    def _convert_to_standard_form(self):
        """非負制約 x_i >= 0 を -x_i <= 0 として追加し、不等式標準形に変換する"""
        n = self.n
        self.A = torch.cat([self.A_orig, -torch.eye(n, device=self.device)], dim=0)
        self.b = torch.cat([self.b_orig, torch.zeros(n, device=self.device)], dim=0)
        self.m = self.A.shape[0]

    def _find_all_vertices(self):
        """可視化用: 制約境界の全交点を逐次的に求め、実行可能頂点を返す"""
        m, n = self.m, self.n
        vertices = []

        for combo in combinations(range(m), n):
            A_sub = self.A[list(combo)]
            b_sub = self.b[list(combo)]

            det = torch.linalg.det(A_sub)
            if det.abs() < 1e-10:
                continue

            x = torch.linalg.solve(A_sub, b_sub)

            residuals = self.A @ x - self.b
            if (residuals <= 1e-6).all():
                is_dup = any((v - x).norm() < 1e-6 for v in vertices)
                if not is_dup:
                    vertices.append(x)

        if not vertices:
            return torch.empty(0, n, device=self.device)
        return torch.stack(vertices)

    def _var_name(self, j):
        """変数インデックスから名前を返す (x1, x2, ..., s1, s2, ...)"""
        if j < self.n:
            return f"x{j + 1}"
        return f"s{j - self.n + 1}"

    def _simplex_method(self):
        """
        被約費用を用いた単体法。原点から開始し、逐次的に隣接頂点へピボットする。

        拡大形: [A_orig | I] [x; s] = b_orig, x >= 0, s >= 0
        初期基底: スラック変数 (s1, ..., sm) → 原点 x = 0
        """
        n = self.n
        m_orig = self.m_orig

        if (self.b_orig < -1e-10).any():
            raise ValueError("b に負の要素があります。原点が実行可能でないため、2段階法が必要です。")

        # 拡大係数行列: [A_orig | I]
        A_aug = torch.cat([self.A_orig, torch.eye(m_orig, device=self.device)], dim=1)
        c_aug = torch.zeros(n + m_orig, device=self.device)
        c_aug[:n] = self.c

        sign = 1.0 if self.maximize else -1.0

        # 初期基底: スラック変数 → 原点 x = 0 から開始
        basis = list(range(n, n + m_orig))

        path = []         # 訪問した頂点のリスト (元の変数空間)
        pivot_log = []    # 各反復のピボット情報

        for _ in range(100):
            # 基底行列とその逆行列
            B = A_aug[:, basis]
            B_inv = torch.linalg.inv(B)
            x_B = B_inv @ self.b_orig

            # 現在の頂点を元の変数空間で取得
            x_full = torch.zeros(n + m_orig, device=self.device)
            for i, bi in enumerate(basis):
                x_full[bi] = x_B[i]
            path.append(x_full[:n].clone())

            # 被約費用の計算
            c_B = c_aug[basis]
            y = c_B @ B_inv   # 双対変数（シンプレックス乗数）

            non_basis = sorted(set(range(n + m_orig)) - set(basis))
            reduced_costs = {}
            best_j = None
            best_rc_val = 0.0

            for j in non_basis:
                rc_j = (c_aug[j] - y @ A_aug[:, j]).item()
                reduced_costs[j] = rc_j
                if sign * rc_j > best_rc_val + 1e-10:
                    best_rc_val = sign * rc_j
                    best_j = j

            # 最適性判定: 改善可能な非基底変数がなければ終了
            if best_j is None:
                pivot_log.append({
                    'basis': list(basis),
                    'reduced_costs': reduced_costs,
                    'entering': None,
                    'leaving': None,
                })
                break

            # ピボット列の計算 (基底逆行列 × 入基底変数の列)
            d = B_inv @ A_aug[:, best_j]

            # 最小比テスト: 出基底変数の決定
            min_ratio = float('inf')
            leave_idx = -1
            for i in range(m_orig):
                if d[i] > 1e-10:
                    ratio = (x_B[i] / d[i]).item()
                    if ratio < min_ratio:
                        min_ratio = ratio
                        leave_idx = i

            if leave_idx == -1:
                raise ValueError("問題は非有界です（最小比テスト失敗）")

            leaving_var = basis[leave_idx]

            pivot_log.append({
                'basis': list(basis),
                'reduced_costs': reduced_costs,
                'entering': best_j,
                'leaving': leaving_var,
            })

            # ピボット: 基底の更新
            basis[leave_idx] = best_j

        opt_x = path[-1]
        opt_val = (self.c @ opt_x).item()

        return opt_x, opt_val, path, pivot_log

    def solve(self):
        """LPを解き、最適解・最適値・探索パス・ピボットログを返す"""
        opt_x, opt_val, path, pivot_log = self._simplex_method()
        return opt_x, opt_val, path, pivot_log

    def solve_with_visualization(self):
        """2D問題を解き、実行可能領域・頂点・探索パスを可視化する"""
        if self.n != 2:
            raise ValueError("可視化は2次元問題のみ対応しています")

        opt_x, opt_val, path, pivot_log = self._simplex_method()
        vertices = self._find_all_vertices()

        if vertices.shape[0] == 0:
            raise ValueError("実行可能解が存在しません")

        verts = vertices.cpu().numpy()
        path_np = torch.stack(path).cpu().numpy()
        obj_vals = (vertices @ self.c).cpu().numpy()

        _, ax = plt.subplots(figsize=(10, 8))

        # --- 実行可能領域の描画 ---
        import numpy as np
        centroid = verts.mean(axis=0)
        angles = np.arctan2(verts[:, 1] - centroid[1], verts[:, 0] - centroid[0])
        order = np.argsort(angles)
        polygon = plt.Polygon(verts[order], alpha=0.2, color="cyan", label="Feasible Region")
        ax.add_patch(polygon)

        # --- 制約直線の描画 ---
        x_max = verts[:, 0].max() * 1.3 + 1
        y_max = verts[:, 1].max() * 1.3 + 1
        x_line = np.linspace(-0.5, x_max, 300)

        for i in range(self.A_orig.shape[0]):
            a1, a2 = self.A_orig[i].tolist()
            bi = self.b_orig[i].item()
            if abs(a2) > 1e-10:
                y_line = (bi - a1 * x_line) / a2
                mask = (y_line >= -0.5) & (y_line <= y_max)
                ax.plot(x_line[mask], y_line[mask], "--", alpha=0.5,
                        label=f"{a1:.0f}x1 + {a2:.0f}x2 <= {bi:.0f}")
            else:
                x_val = bi / a1
                ax.axvline(x_val, linestyle="--", alpha=0.5,
                           label=f"{a1:.0f}x1 <= {bi:.0f}")

        # 非負制約の軸
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)

        # --- 全実行可能頂点 ---
        ax.scatter(verts[:, 0], verts[:, 1], s=80, c="blue", zorder=5, label="Feasible Vertices")
        for i, (x, y) in enumerate(verts):
            ax.annotate(f"  ({x:.1f}, {y:.1f})\n  z={obj_vals[i]:.1f}",
                        (x, y), fontsize=8, zorder=6)

        # --- 単体法の探索パス ---
        for i in range(len(path_np) - 1):
            ax.annotate(
                "", xy=path_np[i + 1], xytext=path_np[i],
                arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
                zorder=7,
            )
            mid_x = (path_np[i, 0] + path_np[i + 1, 0]) / 2
            mid_y = (path_np[i, 1] + path_np[i + 1, 1]) / 2

            # ピボット情報をステップに表示
            info = pivot_log[i]
            entering = self._var_name(info['entering'])
            leaving = self._var_name(info['leaving'])
            label = f"step {i + 1}\n{entering} <-> {leaving}"

            ax.text(mid_x, mid_y, label, fontsize=8, color="red",
                    ha="center", va="bottom", fontweight="bold", zorder=8)

        # 始点（原点）と最適解を強調
        ax.scatter(*path_np[0], s=200, c="orange", marker="s",
                   zorder=9, label="Start (Origin)")
        ax.scatter(*path_np[-1], s=200, c="red", marker="*",
                   zorder=9, label=f"Optimal (z={opt_val:.1f})")

        # --- 目的関数の等高線 ---
        c1, c2 = self.c.tolist()
        for frac in [0.25, 0.5, 0.75, 1.0]:
            z = opt_val * frac
            if abs(c2) > 1e-10:
                y_obj = (z - c1 * x_line) / c2
                mask = (y_obj >= -0.5) & (y_obj <= y_max)
                ax.plot(x_line[mask], y_obj[mask], ":", color="green", alpha=0.3)

        ax.set_xlim(-0.5, x_max)
        ax.set_ylim(-0.5, y_max)
        ax.set_xlabel("x1", fontsize=12)
        ax.set_ylabel("x2", fontsize=12)
        mode = "Maximize" if self.maximize else "Minimize"
        ax.set_title(f"Simplex Method (Reduced Cost) — {mode} z = {c1:.0f}x1 + {c2:.0f}x2",
                     fontsize=14)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

        return opt_x, opt_val


if __name__ == "__main__":
    # 例題: maximize 3x₁ + 5x₂
    # subject to:
    #   x₁ + 2x₂ ≤ 12
    #   2x₁ + x₂ ≤ 12
    #   x₁ + x₂  ≤ 8
    #   x₁, x₂   ≥ 0
    c = torch.tensor([3.0, 5.0])
    A = torch.tensor([
        [1.0, 2.0],
        [2.0, 1.0],
        [1.0, 1.0],
    ])
    b = torch.tensor([12.0, 12.0, 8.0])

    solver = Simplex(c, (A, b), maximize=True)
    print(f"Device: {solver.device}")

    opt_x, opt_val, path, pivot_log = solver.solve()

    print("=== 単体法（被約費用） ===")
    print()

    for i, log in enumerate(pivot_log):
        basis_names = [solver._var_name(j) for j in log['basis']]
        obj_val = (solver.c @ path[i]).item()

        print(f"--- 反復 {i + 1} ---")
        print(f"  基底: {basis_names}")
        print(f"  現在の頂点: x = ({path[i][0].item():.2f}, {path[i][1].item():.2f})")
        print(f"  目的関数値: z = {obj_val:.2f}")
        print(f"  被約費用:")
        for j, rc in log['reduced_costs'].items():
            print(f"    {solver._var_name(j)}: {rc:+.4f}")

        if log['entering'] is not None:
            print(f"  → {solver._var_name(log['entering'])} が基底に入り、"
                  f"{solver._var_name(log['leaving'])} が基底から出る")
        else:
            print(f"  → 全ての被約費用が非正 → 最適解に到達")
        print()

    print(f"最適解:   x = ({opt_x[0].item():.2f}, {opt_x[1].item():.2f})")
    print(f"最適値:   z = {opt_val}")
    print(f"反復回数: {len(pivot_log) - 1}")
    print(f"探索パス: {[f'({p[0].item():.1f}, {p[1].item():.1f})' for p in path]}")
    print()

    solver.solve_with_visualization()
