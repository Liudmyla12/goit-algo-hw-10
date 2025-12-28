from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


def f(x: np.ndarray) -> np.ndarray:
    return x ** 2


@dataclass
class Result:
    estimate: float
    reference: float
    abs_error: float
    method: str


def monte_carlo_integral_mean(a: float, b: float, n: int, seed: int | None = None) -> float:
    """
    Монте-Карло оцінка інтеграла через математичне сподівання:
    ∫ f(x) dx ≈ (b-a) * mean(f(U)), де U ~ Uniform(a,b)
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n)
    return (b - a) * float(np.mean(f(x)))


def reference_integral(a: float, b: float) -> tuple[float, str]:
    """
    Референс: або SciPy quad (якщо доступний), або аналітично (для x^2).
    ∫ x^2 dx = (b^3 - a^3)/3
    """
    try:
        import scipy.integrate as spi  # type: ignore
        val, _err = spi.quad(lambda t: t**2, a, b)
        return float(val), "scipy.integrate.quad"
    except Exception:
        val = (b**3 - a**3) / 3.0
        return float(val), "analytic"


def save_plot(a: float, b: float, out_path: str = "integral_plot.png") -> None:
    x = np.linspace(a - 0.5, b + 0.5, 400)
    y = f(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2)

    ix = np.linspace(a, b, 300)
    iy = f(ix)
    ax.fill_between(ix, iy, alpha=0.3)

    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0, float(np.max(y)) + 0.2])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.axvline(x=a, linestyle="--")
    ax.axvline(x=b, linestyle="--")
    ax.set_title(f"Графік інтегрування f(x) = x^2 від {a} до {b}")
    ax.grid(True)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo integration for f(x)=x^2")
    parser.add_argument("--a", type=float, default=0.0, help="lower bound")
    parser.add_argument("--b", type=float, default=2.0, help="upper bound")
    parser.add_argument("--samples", type=int, default=200_000, help="number of random samples")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--plot", action="store_true", help="save plot to integral_plot.png")
    args = parser.parse_args()

    a, b = args.a, args.b
    n, seed = args.samples, args.seed

    mc = monte_carlo_integral_mean(a, b, n, seed=seed)
    ref, ref_method = reference_integral(a, b)
    abs_err = abs(mc - ref)

    print("=== Task 2: Monte Carlo integral ===")
    print(f"Function: f(x)=x^2, interval [{a}, {b}]")
    print(f"Samples: {n}, seed={seed}")
    print(f"Monte Carlo estimate : {mc:.10f}")
    print(f"Reference ({ref_method}): {ref:.10f}")
    print(f"Absolute error       : {abs_err:.10f}")

    if args.plot:
        save_plot(a, b, out_path="integral_plot.png")
        print("✅ Plot saved: integral_plot.png")


if __name__ == "__main__":
    main()
