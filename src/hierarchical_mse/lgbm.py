"""LightGBM adapters: a custom objective and a matching eval metric.

``lightgbm`` is an optional dependency and is imported lazily, so this module can
be imported (and the rest of the package used) without it installed.

Both factories return callables that work with **either** LightGBM API:

* native ``lgb.train`` -- objective ``(preds, dataset)``, feval ``(preds, dataset)``
* sklearn wrapper -- objective ``(y_true, y_pred)``, eval_metric ``(y_true, y_pred)``

They dispatch on whether the second argument is an ``lgb.Dataset``.

Prefer the native API when you have validation sets: the sklearn ``eval_metric``
callable receives only ``(y_true, y_pred)`` with no dataset identity, so with more
than one eval set there is no reliable way to select the right
:class:`~hierarchical_mse.groups.GroupIndex`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .groups import GroupIndex
from .loss import HessianMode, grad_hess, power_loss

__all__ = ["lgbm_eval", "lgbm_objective"]

_INSTALL_HINT = (
    "The LightGBM helpers in hierarchical-mse require lightgbm, which is an "
    "optional dependency. Install it with:  pip install 'hierarchical-mse[lightgbm]'"
)


def _require_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc
    return lgb


def _unpack(first: Any, second: Any, lgb: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, preds)`` from either calling convention."""
    if isinstance(second, lgb.Dataset):
        # native API: (preds, dataset)
        label = second.get_label()
        if label is None:
            raise ValueError("the LightGBM Dataset has no label; cannot evaluate the loss")
        return np.asarray(label, dtype=np.float64), np.asarray(first, dtype=np.float64)
    # sklearn API: (y_true, y_pred)
    return np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)


def lgbm_objective(
    idx: GroupIndex, hessian: HessianMode = "bound"
) -> Callable[[Any, Any], tuple[np.ndarray, np.ndarray]]:
    """Build a LightGBM custom objective minimizing the hierarchical MSE.

    Pass the result as ``params["objective"]`` (native) or
    ``LGBMRegressor(objective=...)`` (sklearn).

    Two things are easy to forget and both are silent failures:

    * LightGBM does not apply ``boost_from_average`` to custom objectives, so pass
      ``init_score=init_raw_score(y, idx)``.
    * Early stopping must watch :func:`lgbm_eval`, not ``l2`` -- otherwise the
      stopping rule optimizes the objective this loss exists to replace.

    ``idx.codes`` is positional, so the rows LightGBM trains on must be in the
    same order as the ``groups`` array used to build ``idx``.
    """
    lgb = _require_lightgbm()

    def objective(first: Any, second: Any) -> tuple[np.ndarray, np.ndarray]:
        y_true, preds = _unpack(first, second, lgb)
        if y_true.size != idx.n:
            raise ValueError(
                f"objective received {y_true.size} rows but the GroupIndex describes "
                f"{idx.n}. The index must be built from the training rows, in order."
            )
        return grad_hess(y_true, preds, idx, hessian=hessian)

    objective.__name__ = "hierarchical_mse_objective"
    return objective


def lgbm_eval(
    index: GroupIndex | Mapping[Any, GroupIndex],
    name: str = "hierarchical_mse",
) -> Callable[[Any, Any], tuple[str, float, bool]]:
    """Build a LightGBM eval metric reporting the hierarchical MSE.

    Parameters
    ----------
    index
        A single :class:`~hierarchical_mse.groups.GroupIndex`, or a mapping from
        ``lgb.Dataset`` to ``GroupIndex`` when you evaluate on more than one set.
        With a single index the row count is checked on every call, so a
        mismatched validation set raises instead of silently scoring nonsense.
    name
        Metric name reported to LightGBM.

    Use as ``feval=`` (native) or ``eval_metric=`` (sklearn). Lower is better.
    """
    lgb = _require_lightgbm()

    def feval(first: Any, second: Any) -> tuple[str, float, bool]:
        if isinstance(index, Mapping):
            if not isinstance(second, lgb.Dataset):
                raise TypeError(
                    "a mapping of Datasets to GroupIndex requires the native LightGBM "
                    "API, where the eval function receives the Dataset. The sklearn "
                    "eval_metric callable cannot identify which eval set it is scoring."
                )
            idx = index.get(second) or _lookup_by_identity(index, second)
            if idx is None:
                raise KeyError(
                    "no GroupIndex registered for this eval Dataset. Every Dataset "
                    "passed to valid_sets must be a key in the mapping."
                )
        else:
            idx = index

        y_true, preds = _unpack(first, second, lgb)
        if y_true.size != idx.n:
            raise ValueError(
                f"eval received {y_true.size} rows but the GroupIndex describes {idx.n}. "
                "Build one GroupIndex per dataset and pass them as a mapping."
            )
        return name, power_loss(y_true, preds, idx), False

    return feval


def _lookup_by_identity(mapping: Mapping[Any, GroupIndex], key: Any) -> GroupIndex | None:
    for candidate, value in mapping.items():
        if candidate is key:
            return value
    return None
