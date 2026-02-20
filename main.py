import torch
import matplotlib.pyplot as plt
from itertools import combinations


class Simplex():
    def __init__(self, objective, constraints, maximize=True):
        """
        Args:
            objective: 目的関数の係数ベクトル (n,)
            constraints: タプル (A, b)。A @ x <= b の形式。A は (m, n)、b は (m,)
            maximize: True なら最大化、False なら最小化
        """
        self.c = objective.float()
        self.A_orig = constraints[0].float()
        self.b_orig = constraints[1].float()
        self.maximize = maximize
        self.n = self.c.shape[0]

        self._convert_to_standard_form()

    def _convert_to_standard_form(self):
        """非負制約 x_i >= 0 を -x_i <= 0 として追加し、標準形に変換する"""
        n = self.n
        self.A = torch.cat([self.A_orig, -torch.eye(n)], dim=0)
        self.b = torch.cat([self.b_orig, torch.zeros(n)], dim=0)
        self.m = self.A.shape[0]

    def _find_all_vertices(self):
        """制約境界の全交点をバッチ処理で一括計算し、実行可能頂点を返す"""
        m, n = self.m, self.n

        # m本の制約からn本を選ぶ全組み合わせを列挙
        combos = list(combinations(range(m), n))
        idx = torch.tensor(combos, dtype=torch.long)  # (C, n)

        # バッチで連立方程式を構築: A_batch @ x = b_batch
        A_batch = self.A[idx]   # (C, n, n)
        b_batch = self.b[idx]   # (C, n)

        # 正則な系のみを抽出（行列式が0でないもの）
        dets = torch.linalg.det(A_batch)  # (C,)
        nonsingular = dets.abs() > 1e-10

        A_valid = A_batch[nonsingular]
        b_valid = b_batch[nonsingular]

        if A_valid.shape[0] == 0:
            return torch.empty(0, n), []

        # 正則な系を一括で解く
        vertices = torch.linalg.solve(A_valid, b_valid)  # (V, n)
        # 実行可能性の判定: 全制約 A @ x <= b + eps を満たすか
        residuals = self.A @ vertices.T - self.b.unsqueeze(1)  # (m, V)
        feasible = (residuals <= 1e-6).all(dim=0)               # (V,)

        feas_vertices = vertices[feasible]

        # 重複頂点を除去（距離が十分近いものを統合）
        if feas_vertices.shape[0] > 1:
            feas_vertices = self._deduplicate(feas_vertices)

        # 各実行可能頂点の有効制約（等式で成立する制約）を特定
        if feas_vertices.shape[0] > 0:
            res = (self.A @ feas_vertices.T - self.b.unsqueeze(1)).abs()  # (m, F)
            active = [
                set(torch.where(res[:, j] < 1e-6)[0].tolist())
                for j in range(feas_vertices.shape[0])
            ]
        else:
            active = []

        return feas_vertices, active

    def _deduplicate(self, vertices, tol=1e-6):
        """距離がtol以内の重複頂点を除去する"""
        unique = [vertices[0]]
        for i in range(1, vertices.shape[0]):
            dists = torch.stack(unique) - vertices[i].unsqueeze(0)  # (U, n)
            if (dists.norm(dim=1) > tol).all():
                unique.append(vertices[i])
        return torch.stack(unique)

    def _build_adjacency(self, active_sets):
        """共有する有効制約がn-1個以上の頂点ペアを隣接とみなす"""
        num_v = len(active_sets)
        adj = [[] for _ in range(num_v)]
        for i in range(num_v):
            for j in range(i + 1, num_v):
                shared = len(active_sets[i] & active_sets[j])
                if shared >= self.n - 1:
                    adj[i].append(j)
                    adj[j].append(i)
        return adj

    def _greedy_search(self, vertices, adj):
        """単体法の貪欲探索: 隣接頂点を辿って目的関数を改善していく"""
        obj_vals = vertices @ self.c  # (F,)
        sign = 1.0 if self.maximize else -1.0

        # 最悪の頂点から開始して、探索パスを長く見せる
        current = torch.argmin(sign * obj_vals).item()
        path = [current]
        visited = {current}

        while True:
            neighbors = adj[current]
            best_neighbor = None
            best_val = sign * obj_vals[current].item()

            for nb in neighbors:
                if nb in visited:
                    continue
                val = sign * obj_vals[nb].item()
                if val > best_val + 1e-10:
                    best_val = val
                    best_neighbor = nb

            if best_neighbor is None:
                break

            current = best_neighbor
            path.append(current)
            visited.add(current)

        return current, path, obj_vals

    def solve(self):
        """LPを解き、最適頂点・最適値・探索パスを返す"""
        vertices, active = self._find_all_vertices()

        if vertices.shape[0] == 0:
            raise ValueError("実行可能解が存在しません")

        adj = self._build_adjacency(active)
        opt_idx, path, obj_vals = self._greedy_search(vertices, adj)

        return vertices[opt_idx], obj_vals[opt_idx].item(), vertices, path

    def solve_with_visualization(self):
        """2D問題を解き、実行可能領域・頂点・探索パスを可視化する"""
        if self.n != 2:
            raise ValueError("可視化は2次元問題のみ対応しています")

        vertices, active = self._find_all_vertices()

        if vertices.shape[0] == 0:
            raise ValueError("実行可能解が存在しません")

        adj = self._build_adjacency(active)
        opt_idx, path, obj_vals = self._greedy_search(vertices, adj)
        verts = vertices.numpy()

        _, ax = plt.subplots(figsize=(10, 8))

        # --- 実行可能領域の描画 ---
        centroid = verts.mean(axis=0)
        import numpy as np
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
            ax.annotate(f"  ({x:.1f}, {y:.1f})\n  z={obj_vals[i].item():.1f}",
                        (x, y), fontsize=8, zorder=6)

        # --- 貪欲探索パス ---
        path_verts = verts[path]
        for i in range(len(path) - 1):
            ax.annotate(
                "", xy=path_verts[i + 1], xytext=path_verts[i],
                arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
                zorder=7,
            )
            # ステップ番号
            mid_x = (path_verts[i, 0] + path_verts[i + 1, 0]) / 2
            mid_y = (path_verts[i, 1] + path_verts[i + 1, 1]) / 2
            ax.text(mid_x, mid_y, f"step {i + 1}", fontsize=9, color="red",
                    ha="center", va="bottom", fontweight="bold", zorder=8)

        # 始点と終点を強調
        ax.scatter(*path_verts[0], s=200, c="orange", marker="s",
                   zorder=9, label="Start Point")
        ax.scatter(*path_verts[-1], s=200, c="red", marker="*",
                   zorder=9, label=f"Optimal (z={obj_vals[opt_idx].item():.1f})")

        # --- 目的関数の等高線 ---
        c1, c2 = self.c.tolist()
        opt_val = obj_vals[opt_idx].item()
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
        ax.set_title(f"Simplex Method (Greedy Search) - {mode} z = {c1:.0f}x1 + {c2:.0f}x2", fontsize=14)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

        return vertices[opt_idx], obj_vals[opt_idx].item()


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

    opt_x, opt_val, all_vertices, search_path = solver.solve()
    print("=== 単体法（PyTorch バッチ処理 + 貪欲探索） ===")
    print(f"最適解:   x = {opt_x.tolist()}")
    print(f"最適値:   z = {opt_val}")
    print(f"全実行可能頂点:\n{all_vertices}")
    print(f"探索パス: {search_path}")
    print()

    solver.solve_with_visualization()
