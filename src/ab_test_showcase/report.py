from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def decide(
    conversion: dict[str, Any],
    checkout_error: dict[str, Any],
    mde_absolute: float,
    max_checkout_error_increase: float,
) -> tuple[str, str]:
    if conversion["ci_high"] < 0:
        return "Не внедрять", "Интервал эффекта указывает на снижение конверсии."
    if checkout_error["ci_high"] > max_checkout_error_increase:
        primary_status = (
            "Интервал основной метрики подтверждает эффект не меньше MDE."
            if conversion["ci_low"] >= mde_absolute
            else "Основная метрика пока не подтверждает эффект не меньше MDE."
        )
        return (
            "Не масштабировать; проверить guardrail",
            f"{primary_status} Верхняя граница CI ошибок checkout "
            f"({checkout_error['ci_high'] * 100:.2f} п.п.) выше допустимого предела "
            f"({max_checkout_error_increase * 100:.2f} п.п.).",
        )
    if conversion["ci_low"] >= mde_absolute:
        return (
            "Внедрять поэтапно",
            "Нижняя граница интервала превышает MDE, guardrail ошибок не превышает предел.",
        )
    if conversion["ci_high"] < mde_absolute:
        return (
            "Не внедрять широкий rollout",
            "Верхняя граница интервала ниже минимально полезного эффекта (MDE).",
        )
    return (
        "Результат неопределён; продолжить/повторить тест",
        "Интервал совместим с практически значимым эффектом, но не подтверждает его с заданной точностью.",
    )


def save_effect_plot(
    results: dict[str, Any],
    path: str | Path,
    mde: float,
    max_checkout_error_increase: float,
) -> None:
    conversion = results["conversion"]
    error = results["checkout_error"]
    labels = ["Конверсия (B − A)", "Ошибка checkout (B − A)"]
    estimates = [100 * conversion["absolute_lift"], 100 * error["absolute_lift"]]
    lower = [100 * conversion["ci_low"], 100 * error["ci_low"]]
    upper = [100 * conversion["ci_high"], 100 * error["ci_high"]]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(labels))
    axis.errorbar(
        estimates,
        y,
        xerr=[
            np.asarray(estimates) - np.asarray(lower),
            np.asarray(upper) - np.asarray(estimates),
        ],
        fmt="o",
        capsize=5,
        color="#1769aa",
    )
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    axis.vlines(
        100 * mde,
        -0.3,
        0.3,
        color="#e67e22",
        linewidth=1,
        linestyle=":",
        label="MDE (конверсия)",
    )
    axis.vlines(
        100 * max_checkout_error_increase,
        0.7,
        1.3,
        color="#c0392b",
        linewidth=1,
        linestyle=":",
        label="Предел guardrail",
    )
    axis.set_yticks(y, labels)
    axis.set_ylim(-0.5, 1.5)
    axis.set_xlabel("Абсолютная разница, п.п.")
    axis.set_title("Эффект варианта B и 95% доверительные интервалы")
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def write_memo(
    path: str | Path,
    results: dict[str, Any],
    design: dict[str, Any],
    config: dict[str, Any],
    source_description: str,
) -> None:
    conversion = results["conversion"]
    checkout_error = results["checkout_error"]
    checkout_started = results["checkout_started"]
    revenue = results["revenue_per_user"]
    recommendation, reasoning = decide(
        conversion,
        checkout_error,
        config["mde_absolute"],
        config["maximum_checkout_error_increase"],
    )
    relative_lift = (
        f"{conversion['relative_lift']:+.1%}"
        if conversion["relative_lift"] is not None
        else "н/д (нет конверсий в контроле)"
    )
    memo = f"""# Бизнес-меморандум: упрощение checkout

## Гипотеза

Упрощённый checkout повышает конверсию в заказ на пользователя; доля ошибок оформления не растёт более чем на {config["maximum_checkout_error_increase"] * 100:.1f} п.п.

## Дизайн

- Единица рандомизации: пользователь; распределение 1:1.
- Основная метрика: конверсия в заказ.
- H0: конверсия B равна конверсии A. H1: конверсии различаются (двусторонний тест).
- MDE: {config["mde_absolute"] * 100:.1f} п.п. абсолютного прироста; α = {config["alpha"]:.2f}; мощность = {config["power"]:.0%}.
- Заложенный в симуляцию эффект: +{config["simulated_treatment_effect_absolute"] * 100:.2f} п.п. абсолютного прироста (сегментные эффекты отличаются).
- Требуемая выборка: {design["sample_size_per_arm"]:,} пользователей на группу; ожидаемая длительность — {design["estimated_duration_days"]} дней.
- Источник baseline: {source_description}.
- Проверка SRM: p = {results["srm"]["p_value"]:.4g}; {"пройдена" if results["srm"]["passed"] else "не пройдена"}.

## Результаты симуляции

- Конверсия: контроль {conversion["control_rate"]:.2%} ({conversion["control_n"]:,}), вариант {conversion["treatment_rate"]:.2%} ({conversion["treatment_n"]:,}).
- Эффект B−A: {conversion["absolute_lift"] * 100:+.2f} п.п. абсолютный, {relative_lift} относительный; 95% CI [{conversion["ci_low"] * 100:+.2f} п.п., {conversion["ci_high"] * 100:+.2f} п.п.], p = {conversion["p_value"]:.4g} ({conversion["method"]}).
- Начало оформления (proxy): эффект B−A {checkout_started["absolute_lift"] * 100:+.2f} п.п.; 95% CI [{checkout_started["ci_low"] * 100:+.2f} п.п., {checkout_started["ci_high"] * 100:+.2f} п.п.], p = {checkout_started["p_value"]:.4g}.
- Ошибки checkout: эффект B−A {checkout_error["absolute_lift"] * 100:+.2f} п.п.; 95% CI [{checkout_error["ci_low"] * 100:+.2f} п.п., {checkout_error["ci_high"] * 100:+.2f} п.п.].
- Выручка на пользователя: B−A {revenue["absolute_lift"]:+.2f} условных ден. ед.; Welch p = {revenue["p_value"]:.4g}; bootstrap 95% CI [{revenue["bootstrap_ci_low"]:+.2f}, {revenue["bootstrap_ci_high"]:+.2f}]. Доля нулевой выручки: контроль {revenue["control_zero_fraction"]:.1%}, вариант {revenue["treatment_zero_fraction"]:.1%}; асимметрия проверяется описательно, Welch дополняется bootstrap.

## Рекомендация

**{recommendation}.** {reasoning}

Не следует принимать решение только по p-value: сравнивайте интервал эффекта с MDE, учитывайте guardrails и стоимость внедрения.

## Ограничения

Это синтетический эксперимент с заданным в конфигурации эффектом, а не оценка эффекта реального checkout. Исторические Kaggle/UCI-данные (если переданы) используются только для калибровки baseline и долей типов посетителей: в них нет случайного назначения. Параметры baseline, потока пользователей, treatment effect и допустимого ухудшения guardrail демонстрационные и должны быть заменены данными продукта. Анализ сегментов исследовательский; оценки зависят от выбранного сценария и множественных сравнений. Для продуктового решения нужны реальные логи назначения, достаточный горизонт с учётом сезонности, проверка качества эксперимента и заранее согласованные бизнес-пороги.
"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(memo, encoding="utf-8")


def save_json(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
