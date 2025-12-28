from __future__ import annotations

from pulp import LpMaximize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, value


def solve_production():
    # Змінні: кількість одиниць продуктів
    lemonade = LpVariable("Lemonade", lowBound=0, cat="Integer")
    juice = LpVariable("FruitJuice", lowBound=0, cat="Integer")

    # Модель: максимізуємо загальну кількість продуктів
    model = LpProblem("Production_Optimization", LpMaximize)
    model += lemonade + juice, "Total_Products"

    # Обмеження ресурсів
    # Вода: 2*лимонад + 1*сік <= 100
    model += 2 * lemonade + 1 * juice <= 100, "Water"

    # Цукор: 1*лимонад <= 50
    model += 1 * lemonade <= 50, "Sugar"

    # Лимонний сік: 1*лимонад <= 30
    model += 1 * lemonade <= 30, "LemonJuice"

    # Фруктове пюре: 2*сік <= 40
    model += 2 * juice <= 40, "FruitPuree"

    # Вимикаємо детальний лог CBC
    model.solve(PULP_CBC_CMD(msg=False))

    status = LpStatus[model.status]
    lemonade_val = int(value(lemonade))
    juice_val = int(value(juice))
    total = int(value(model.objective))

    return status, lemonade_val, juice_val, total


def main():
    status, lemonade_val, juice_val, total = solve_production()

    print("=== Task 1: Production optimization (PuLP) ===")
    print(f"Status: {status}")
    print(f"Lemonade     : {lemonade_val}")
    print(f"Fruit Juice  : {juice_val}")
    print(f"TOTAL        : {total}")


if __name__ == "__main__":
    main()

