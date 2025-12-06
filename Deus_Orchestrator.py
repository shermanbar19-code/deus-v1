# Deus_Orchestrator.py
# ---------------------------------------------------------
# Pipeline: Analyst -> Architect -> Learning -> PM -> QA
# פלטים נשמרים בתיקייה experiments/session_yyyymmdd_hhmmss/
#
# הקוד מניח את הקבצים:
# agents/Deus_Analyst.py  (Analyst)
# agents/Deus_Architect.py (Architect)
# agents/Deus_Learning.py  (Learning)
# agents/Deus_PM.py        (PM)
# agents/Deus_QA.py        (QA)
#
# להרצה:  python Deus_Orchestrator.py
# ---------------------------------------------------------

import os
import json
import datetime
import traceback

import pandas as pd
import matplotlib.pyplot as plt

# --- יבוא הסוכנים בשמות החדשים ---
from agents.Deus_Analyst import Analyst
from agents.Deus_Architect import Architect
from agents.Deus_Learning import Learning
from agents.Deus_PM import PM
from agents.Deus_QA import QA


class Orchestrator:
    def __init__(self):
        # תיקית ניסויים ראשית
        self.experiments_dir = "experiments"
        os.makedirs(self.experiments_dir, exist_ok=True)

        # תיקיה לריצה הנוכחית עם חותמת זמן
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.experiments_dir, f"session_{ts}")
        os.makedirs(self.session_dir, exist_ok=True)

        # קובץ לוגים
        self.log_fp = open(os.path.join(self.session_dir, "log.txt"), "w", encoding="utf-8")

        def _log(msg: str):
            print(msg)
            self.log_fp.write(msg + "\n")
            self.log_fp.flush()

        self.log = _log

    # --------------------------
    # הפעלה מלאה
    # --------------------------
    def run(self):
        self.log("Orchestrator: Starting new trading bot session.")
        issues_found = []

        try:
            # 1) טעינת נתונים וחישוב פיצ'רים
            analyst = Analyst(logger=self.log)
            data = analyst.load_data()                   # מצפה לקרוא data/sp500.csv ו/או data/nasdaq100.csv
            features = analyst.compute_features()        # מוסיף ATR/MA/חוזק יחסי וכו'
            anomalies = analyst.detect_anomalies()
            if anomalies:
                issues_found.extend([f"Data: {issue}" for issue in anomalies])

            # 2) עיצוב אסטרטגיה וגריד פרמטרים
            architect = Architect(logger=self.log)
            strategy = architect.design_strategy(features)
            architect.detect_overfitting(strategy, len(features))

            # 3) אופטימיזציה/למידה + שימוש בריצות קודמות
            learning = Learning(strategy, logger=self.log)
            learning.use_previous_runs(experiments_dir=self.experiments_dir)
            best_params, train_metrics, test_metrics = learning.optimize_strategy(features)
            self.log(f"Orchestrator: Best parameters found: {best_params}")

            # שמירת הפרמטרים הטובים
            with open(os.path.join(self.session_dir, "best_params.json"), "w", encoding="utf-8") as f:
                json.dump(best_params, f, ensure_ascii=False, indent=2)

            # 4) הרצת האסטרטגיה וקבלת עקומת Equity
            pm = PM(initial_capital=10000, logger=self.log)
            trade_log, equity_curve = pm.run_strategy(strategy, best_params, features)
            perf = pm.compute_performance(equity_curve)
            self.log(
                "Orchestrator: Combined performance -> "
                f"Total Return: {perf.get('total_return', 0)*100:.2f}%, "
                f"Sharpe: {perf.get('sharpe', 0):.2f}, "
                f"Max Drawdown: {perf.get('max_drawdown', 0):.2f}"
            )

            # 5) בדיקות QA (אוברפיט/יציבות)
            train_size = int(0.7 * len(features))
            qa = QA(logger=self.log)
            perf_issues = qa.validate_performance(test_metrics, train_metrics)
            stab_issues = qa.stability_analysis(equity_curve, start_index_of_test=train_size)
            if perf_issues:
                issues_found.extend([f"Performance: {issue}" for issue in perf_issues])
            if stab_issues:
                issues_found.extend([f"Stability: {issue}" for issue in stab_issues])

            # 6) יצוא דו״ח מסכם
            report_path = os.path.join(self.session_dir, "report.txt")
            with open(report_path, "w", encoding="utf-8") as report:
                report.write("Trading Bot Performance Report\n")
                report.write("================================\n")
                report.write(f"Session: {os.path.basename(self.session_dir)}\n")
                report.write(f"Best Parameters: {best_params}\n")
                report.write(
                    f"Train Sharpe: {train_metrics.get('sharpe'):.2f}, "
                    f"Drawdown: {train_metrics.get('drawdown'):.2f}, "
                    f"Trades: {train_metrics.get('trades')}\n"
                )
                report.write(
                    f"Test  Sharpe: {test_metrics.get('sharpe'):.2f}, "
                    f"Drawdown: {test_metrics.get('drawdown'):.2f}, "
                    f"Trades: {test_metrics.get('trades')}\n"
                )

                # חישוב תשואת תקופת המבחן בלבד
                try:
                    from agents.Deus_PM import PM as PM_NoLog
                    pm_test = PM_NoLog(initial_capital=10000)
                    val_data = features.iloc[train_size:]
                    _, equity_test = pm_test.run_strategy(strategy, best_params, val_data)
                    if equity_test is not None and len(equity_test) > 0:
                        test_return_pct = (equity_test.iloc[-1] / equity_test.iloc[0] - 1) * 100
                        report.write(f"Test Total Return: {test_return_pct:.2f}%\n")
                except Exception as e:
                    self.log(f"Orchestrator: Could not compute isolated test return: {e}")

                report.write(
                    f"Combined Sharpe (full period): {perf.get('sharpe', 0):.2f}, "
                    f"Max Drawdown: {perf.get('max_drawdown', 0):.2f}\n"
                )

                if issues_found:
                    report.write("Issues Detected:\n")
                    for issue in issues_found:
                        report.write(f" - {str(issue)}\n")
                else:
                    report.write("No critical issues detected in validation.\n")

                report.write("See log.txt for detailed logs and warnings.\n")
                report.write("Equity curve graph saved as equity_curve.png.\n")

            # 7) גרף Equity
            try:
                # הבטחת אינדקס datetime (אם לא—המרה)
                try:
                    equity_curve.index = pd.to_datetime(equity_curve.index)
                except Exception:
                    pass

                plt.style.use("dark_background")
                plt.figure(figsize=(11, 6))
                equity_curve.plot(label='Equity', linewidth=1.6)
                if 0 < train_size < len(equity_curve):
                    split_date = equity_curve.index[train_size]
                    plt.axvline(x=split_date, linestyle='--', label='Train/Test Split')
                plt.title('Equity Curve')
                plt.xlabel('Date')
                plt.ylabel('Equity')
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                out_path = os.path.join(self.session_dir, "equity_curve.png")
                plt.savefig(out_path)
                plt.close()
                self.log(f"Orchestrator: Saved equity curve plot to {out_path}")
            except Exception as e:
                self.log(f"Orchestrator: Plotting failed: {e}")

        except Exception as e:
            self.log("Orchestrator: Exception during run - " + str(e))
            self.log(traceback.format_exc())
        finally:
            self.log("Orchestrator: Session complete.")
            try:
                self.log_fp.close()
            except Exception:
                pass


if __name__ == '__main__':
    orch = Orchestrator()
    orch.run()
