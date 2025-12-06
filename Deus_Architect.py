import os
import yaml
import pandas as pd
from datetime import datetime
from agents.Deus_StrategyConfig import StrategyConfig

class Architect:
    """
    Deus Architect — גרסה מתוקנת לחלוטין עם טיפול בשדה 'Date' ללא זריקת KeyError.
    """

    def __init__(self, logger=None):
        self.log = logger if logger is not None else print
        self.config_dir = os.path.join(os.getcwd(), "configs")
        os.makedirs(self.config_dir, exist_ok=True)
        self.log(f"[DEBUG] Architect initialized. Config path: {self.config_dir}")

    def design_strategy(self, features: pd.DataFrame) -> StrategyConfig:
        try:
            if features is None or features.empty:
                raise ValueError("Features DataFrame is empty or None")

            # טיפול ב-'Date' באופן בטוח
            if not isinstance(features.index, pd.DatetimeIndex):
                self.log("[DEBUG] Converting index to DatetimeIndex...")
                if 'Date' in features.columns:
                    features = features.rename(columns={'Date': 'date'})  # שינוי שם אם יש בעיות קייס
                    features['date'] = pd.to_datetime(features['date'], errors='coerce')
                    features = features.dropna(subset=['date'])
                    features = features.set_index('date')
                else:
                    features.index = pd.to_datetime(features.index, errors='coerce')
                    features = features[features.index.notna()]

            long_k, short_k = 0.6, 0.6
            self.log(f"[DEBUG] Building StrategyConfig with {features.shape[1]} features.")

            strategy = StrategyConfig(features=features, long_k=long_k, short_k=short_k, logger=self.log)
            if not hasattr(strategy, 'backtest'):
                raise AttributeError("StrategyConfig does not contain backtest method — check Deus_StrategyConfig.py")

            config_path = os.path.join(self.config_dir, f"strategy_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
            self._save_config_yaml({
                "feature_count": features.shape[1],
                "long_k": long_k,
                "short_k": short_k,
                "created_at": datetime.now().isoformat(),
                "columns": list(features.columns)
            }, config_path)

            self.log(f"Architect: Saved YAML to {config_path}")
            return strategy

        except Exception as e:
            self.log(f"[ERROR] Architect.design_strategy failed — {e}")
            raise

    def detect_overfitting(self, strategy: StrategyConfig, data_length: int):
        try:
            dof = 26
            ratio = data_length / max(dof, 1)
            if ratio < 50:
                self.log(f"[WARNING] Overfitting risk — ratio={ratio:.2f}")
            else:
                self.log(f"[OK] Data ratio={ratio:.2f}")
        except Exception as e:
            self.log(f"[ERROR] detect_overfitting failed — {e}")

    def _save_config_yaml(self, cfg: dict, path: str):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(cfg, f, allow_unicode=True)
        except Exception as e:
            self.log(f"[ERROR] YAML save failed — {e}")