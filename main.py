import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


class Simplex:
    def __init__(self, objective, constraints, maximize=True):
        """
        Args:
            objective: 目的関数の係数ベクトル (n,)
            constraints: タプル (A, b)。A @ x <= b の形式。A は (m, n)、b は (m,)
            maximize: True なら最大化、False なら最小化
        """
        self.c = np.asarray(objective, dtype=float)
        self.A = np.asarray(constraints[0], dtype=float)
        self.b = np.asarray(constraints[1], dtype=float)
        self.maximize = maximize
        self.n = self.c.shape[0]
        self.m = self.A.shape[0]

    def solve(self):
        """
        タブロー形式の単体法。原点から開始し、被約費用に基づきピボットする。

        タブロー構造 (m+1) x (n+m+1):
            [ -sign*c | 0 | 0 ]   <- 目的関数行 (被約費用)
            [    A    | I | b ]   <- 制約行
        """
        n, m = self.n, self.m
        sign = 1.0 if self.maximize else -1.0

        tab = np.zeros((m + 1, n + m + 1))
        tab[0, :n] = -sign * self.c
        tab[1:, :n] = self.A
        tab[1:, n:n + m] = np.eye(m)
        tab[1:, -1] = self.b

        basis = list(range(n, n + m))
        path = [np.zeros(n)]

        for _ in range(100):
            # 被約費用が最も負の列を選択（入基底変数）
            rc = tab[0, :-1]
            pivot_col = rc.argmin()
            if rc[pivot_col] >= -1e-10:
                break  # 全被約費用が非負 → 最適

            # 最小比テスト（出基底変数）
            col = tab[1:, pivot_col]
            rhs = tab[1:, -1]
            ratios = np.where(col > 1e-10, rhs / col, np.inf)
            pivot_row = ratios.argmin()

            if ratios[pivot_row] == np.inf:
                raise ValueError("問題は非有界です")

            # ピボット操作（掃き出し）
            pr = pivot_row + 1
            tab[pr] /= tab[pr, pivot_col]
            for i in range(m + 1):
                if i != pr:
                    tab[i] -= tab[i, pivot_col] * tab[pr]

            basis[pivot_row] = pivot_col

            # 現在の頂点を記録
            x = np.zeros(n)
            for i, bi in enumerate(basis):
                if bi < n:
                    x[bi] = tab[i + 1, -1]
            path.append(x.copy())

        opt_x = path[-1]
        opt_val = sign * tab[0, -1]
        return opt_x, opt_val, path

    def solve_with_visualization(self):
        """2D問題を解き、実行可能領域・頂点・探索パスを可視化する"""
        if self.n != 2:
            raise ValueError("可視化は2次元問題のみ対応しています")

        opt_x, opt_val, path = self.solve()
        vertices = self._find_all_vertices()

        if len(vertices) == 0:
            raise ValueError("実行可能解が存在しません")

        verts = np.array(vertices)
        path_np = np.array(path)
        obj_vals = verts @ self.c

        _, ax = plt.subplots(figsize=(10, 8))

        # --- Feasible Region ---
        centroid = verts.mean(axis=0)
        angles = np.arctan2(verts[:, 1] - centroid[1], verts[:, 0] - centroid[0])
        order = np.argsort(angles)
        polygon = plt.Polygon(verts[order], alpha=0.2, color="cyan", label="Feasible Region")
        ax.add_patch(polygon)

        # --- Constraint Lines ---
        x_max = verts[:, 0].max() * 1.3 + 1
        y_max = verts[:, 1].max() * 1.3 + 1
        x_line = np.linspace(-0.5, x_max, 300)

        for i in range(self.m):
            a1, a2 = self.A[i]
            bi = self.b[i]
            if abs(a2) > 1e-10:
                y_line = (bi - a1 * x_line) / a2
                mask = (y_line >= -0.5) & (y_line <= y_max)
                ax.plot(x_line[mask], y_line[mask], "--", alpha=0.5,
                        label=f"{a1:.0f}x1 + {a2:.0f}x2 <= {bi:.0f}")
            else:
                ax.axvline(bi / a1, linestyle="--", alpha=0.5,
                           label=f"{a1:.0f}x1 <= {bi:.0f}")

        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)

        # --- Feasible Vertices ---
        ax.scatter(verts[:, 0], verts[:, 1], s=80, c="blue", zorder=5, label="Feasible Vertices")
        for i, (x, y) in enumerate(verts):
            ax.annotate(f"  ({x:.1f}, {y:.1f})\n  z={obj_vals[i]:.1f}",
                        (x, y), fontsize=8, zorder=6)

        # --- Simplex Path ---
        for i in range(len(path_np) - 1):
            ax.annotate("", xy=path_np[i + 1], xytext=path_np[i],
                        arrowprops=dict(arrowstyle="->", color="red", lw=2.5), zorder=7)
            mid = (path_np[i] + path_np[i + 1]) / 2
            ax.text(mid[0], mid[1], f"step {i + 1}", fontsize=9, color="red",
                    ha="center", va="bottom", fontweight="bold", zorder=8)

        ax.scatter(*path_np[0], s=200, c="orange", marker="s",
                   zorder=9, label="Start (Origin)")
        ax.scatter(*path_np[-1], s=200, c="red", marker="*",
                   zorder=9, label=f"Optimal (z={opt_val:.1f})")

        # --- Objective Contours ---
        c1, c2 = self.c
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
        ax.set_title(f"Simplex Method (Reduced Cost) -- {mode} z = {c1:.0f}x1 + {c2:.0f}x2",
                     fontsize=14)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

        return opt_x, opt_val

    def _find_all_vertices(self):
        """可視化用: 全実行可能頂点を逐次的に求める"""
        n = self.n
        A_full = np.vstack([self.A, -np.eye(n)])
        b_full = np.concatenate([self.b, np.zeros(n)])
        m_full = A_full.shape[0]

        vertices = []
        for combo in combinations(range(m_full), n):
            A_sub = A_full[list(combo)]
            b_sub = b_full[list(combo)]
            if abs(np.linalg.det(A_sub)) < 1e-10:
                continue
            x = np.linalg.solve(A_sub, b_sub)
            if (A_full @ x - b_full <= 1e-6).all():
                if not any(np.linalg.norm(v - x) < 1e-6 for v in vertices):
                    vertices.append(x)

        return vertices


if __name__ == "__main__":
    # maximize 3x1 + 5x2
    # subject to:
    #   x1 + 2x2 <= 12
    #   2x1 + x2 <= 12
    #   x1 + x2  <= 8
    #   x1, x2   >= 0
    c = np.array([3.0, 5.0])
    A = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
    b = np.array([12.0, 12.0, 8.0])

    solver = Simplex(c, (A, b), maximize=True)

    opt_x, opt_val, path = solver.solve()
    print(f"最適解: x = {opt_x.tolist()}")
    print(f"最適値: z = {opt_val}")
    print(f"探索パス: {[p.tolist() for p in path]}")
    print()

    solver.solve_with_visualization()
