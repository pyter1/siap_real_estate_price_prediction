from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


@dataclass
class ImportanceResult:
    feature_names: List[str]
    importances: np.ndarray


def _get_feature_names(pipeline: Pipeline) -> List[str]:
    preprocess = pipeline.named_steps["preprocess"]
    try:
        names = preprocess.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return []


def get_model_importance(pipeline: Pipeline, top_n: int = 30) -> ImportanceResult:
    model = pipeline.named_steps["model"]
    names = _get_feature_names(pipeline)

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    else:
        imp = np.zeros(len(names), dtype=float)

    if len(names) != len(imp):
        names = [f"f{i}" for i in range(len(imp))]

    idx = np.argsort(-imp)[:top_n]
    return ImportanceResult(feature_names=[names[i] for i in idx], importances=imp[idx])


def plot_importance(result: ImportanceResult, out_path: str, title: str) -> None:
    plt.figure()
    y = np.arange(len(result.feature_names))
    plt.barh(y, result.importances[::-1])
    plt.yticks(y, result.feature_names[::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def shap_summary_plot(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    out_path: str,
    max_display: int = 30,
) -> None:
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    X_trans = preprocess.transform(X_sample)
    names = _get_feature_names(pipeline)
    if len(names) == 0:
        names = [f"f{i}" for i in range(X_trans.shape[1])]

    # TreeExplainer supports XGBoost and sklearn tree models.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    shap.summary_plot(
        shap_values,
        features=X_trans,
        feature_names=names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
