"""
Self-check script for Deus project
---------------------------------
Run this from the project root (where Deus_Orchestrator.py is):

    python self_check.py

What it does:
1) Verifies Python version and required packages are installed.
2) Verifies file/folder structure and key modules/classes exist.
3) If data files are missing, builds a synthetic dataset with the required columns
   (Close, Return, MA50, ATR, NormalizedVolume) to exercise the pipeline.
4) Instantiates all agents (Analyst, Architect, Learning, PM, QA), builds a strategy,
   runs optimization on synthetic data, runs a backtest, and prints a summary.

Exit code 0 = OK, non-zero = a problem was detected.
"""

import os
import sys
import importlib
import traceback
from datetime import datetime

REQUIRED_PACKAGES = [
    ("pandas", None),
    ("numpy", None),
    ("matplotlib", None),
    ("sklearn", None),
    ("yaml", "pyyaml"),  # yaml module provided by pyyaml
]

REQUIRED_FILES = [
    ("Deus_Orchestrator.py", "root file"),
    (os.path.join("agents", "Deus_Analyst.py"), "agent"),
    (os.path.join("agents", "Deus_Architect.py"), "agent"),
    (os.path.join("agents", "Deus_Learning.py"), "agent"),
    (os.path.join("agents", "Deus_PM.py"), "agent"),
    (os.path.join("agents", "Deus_QA.py"), "agent"),
]

DATA_FILES = [
    os.path.join("data", "sp500.csv"),
    os.path.join("data", "nasdaq100.csv"),
]


def ok(msg: str):
    print(f"✅ {msg}")

def warn(msg: str):
    print(f"⚠️  {msg}")

def fail(msg: str):
    print(f"❌ {msg}")


def check_python():
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 9):
        warn(f"Python {major}.{minor} detected. Recommended >= 3.9")
    else:
        ok(f"Python {major}.{minor} detected")


def check_packages():
    problems = False
    for mod_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(mod_name)
            ok(f"Package '{mod_name}' available")
        except Exception:
            problems = True
            name_for_install = pip_name or mod_name
            fail(f"Missing package '{mod_name}'. Install with: pip install {name_for_install}")
    return not problems


def check_files():
    problems = False
    for path, label in REQUIRED_FILES:
        if os.path.exists(path):
            ok(f"Found {label} file: {path}")
        else:
            problems = True
            fail(f"Missing {label} file: {path}")
    return not problems


def quick_imports_and_classes():
    """Ensure we can import agents and classes exist with expected names."""
    problems = False
    try:
        from agents.Deus_Analyst import Analyst
        from agents.Deus_Architect import Architect
        from agents.Deus_Learning import Learning
        from agents.Deus_PM import PM
        from agents.Deus_QA import QA
        ok("Imported agents successfully (Analyst, Architect, Learning, PM, QA)")
    except Exception as e:
        problems = True
        fail(f"Import error: {e}\n{traceback.format_exc()}")
    return not problems


def load_or_build_sample_features():
    """Load data if available, else build synthetic features DataFrame that matches the strategy needs."""
    import numpy as np
    import pandas as pd

    has_any_data = any(os.path.exists(p) for p in DATA_FILES)
    if has_any_data:
        # We won't fully parse real CSVs here—Analyst will do that.
        ok("Data folder present (at least one CSV found)")
        return None

    # Build synthetic dataset (252 trading days ~ 1Y)
    n = 300
    rng = pd.date_range(end=datetime.today(), periods=n, freq="B")
    close = 1000 + np.cumsum(np.random.normal(0, 5, size=n))
    ret = np.zeros(n)
    ret[1:] = (close[1:] - close[:-1]) / close[:-1]
    ma50 = pd.Series(close).rolling(50).mean().fillna(method="bfill").values
    atr = np.abs(np.random.normal(5, 1.5, size=n))
    vol_norm = 1 + np.abs(np.random.normal(0, 0.5, size=n))

    features = __import__("pandas").DataFrame({
        "Close": close,
        "Return": ret,
        "MA50": ma50,
        "ATR": atr,
        "NormalizedVolume": vol_norm,
    }, index=rng)
    ok("Built synthetic features DataFrame (no CSVs detected)")
    return features


def exercise_pipeline(features_override=None):
    """Instantiate agents and run a minimal end-to-end to ensure APIs line up."""
    from agents.Deus_Analyst import Analyst
    from agents.Deus_Architect import Architect
    from agents.Deus_Learning import Learning
    from agents.Deus_PM import PM
    from agents.Deus_QA import QA

    # Logger
    def _log(msg):
        print(msg)

    analyst = Analyst(logger=_log)
    architect = Architect(logger=_log)
    qa = QA(logger=_log)

    # If features provided, use them; else try real pipeline through Analyst
    if features_override is not None:
        features = features_override
    else:
        data = analyst.load_data()
        features = analyst.compute_features()
        anomalies = analyst.detect_anomalies()
        if anomalies:
            warn(f"Analyst anomalies: {anomalies}")

    # Design strategy
    strategy = architect.design_strategy(features)
    architect.detect_overfitting(strategy, len(features))

    # Learning optimize
    learning = Learning(strategy, logger=_log)
    learning.use_previous_runs(experiments_dir="experiments")
    best_params, train_metrics, val_metrics = learning.optimize_strategy(features)

    if not best_params:
        warn("Learning returned no best_params (possibly too little data). Using a safe default if available.")
        # Build a safe default from the grid if possible
        if hasattr(strategy, "param_grid") and strategy.param_grid:
            first = {k: v[0] for k, v in strategy.param_grid.items()}
            best_params = first
        else:
            fail("Strategy has no param_grid; cannot proceed.")
            return False

    ok(f"Learning best params: {best_params}")

    # Backtest via PM
    pm = PM(initial_capital=10000, logger=_log)
    trade_log, equity = pm.run_strategy(strategy, best_params, features)
    perf = pm.compute_performance(equity)

    ok("PM backtest completed")
    print("Summary:")
    print({
        "trades": len(trade_log),
        "total_return": perf.get("total_return"),
        "sharpe": perf.get("sharpe"),
        "max_drawdown": perf.get("max_drawdown"),
    })

    # QA checks
    qa_issues = qa.validate_performance({
        "sharpe": perf.get("sharpe"),
        "drawdown": perf.get("max_drawdown"),
        "trades": len(trade_log)
    }, train_metrics)

    stab_issues = qa.stability_analysis(equity)

    if qa_issues:
        warn(f"QA issues: {qa_issues}")
    if stab_issues:
        warn(f"Stability issues: {stab_issues}")

    ok("Pipeline exercised successfully")
    return True


if __name__ == "__main__":
    overall_ok = True

    print("\n=== Deus Self-Check ===")
    check_python()

    if not check_packages():
        overall_ok = False

    if not check_files():
        overall_ok = False

    if not quick_imports_and_classes():
        overall_ok = False

    # If structure OK, try to run a tiny pipeline on synthetic data if needed
    try:
        features_override = load_or_build_sample_features()
        if not exercise_pipeline(features_override):
            overall_ok = False
    except Exception as e:
        overall_ok = False
        fail(f"Pipeline exercise failed: {e}\n{traceback.format_exc()}")

    print("\n=== Result ===")
    if overall_ok:
        ok("All checks passed. Project is wired correctly.")
        sys.exit(0)
    else:
        fail("Some checks failed. Please review messages above.")
        sys.exit(1)
