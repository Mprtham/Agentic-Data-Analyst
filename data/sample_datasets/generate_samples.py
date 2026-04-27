"""
Generate synthetic sample datasets for demo purposes.
Run once: python generate_samples.py
"""

import math
import random
import csv
from pathlib import Path

rng = random.Random(42)
out = Path(__file__).parent


# ---------------------------------------------------------------------------
# 1. E-commerce orders (time series + regression)
# ---------------------------------------------------------------------------

def generate_ecommerce():
    rows = []
    base_date = "2022-01-01"
    from datetime import date, timedelta

    start = date(2022, 1, 1)
    for i in range(730):  # 2 years
        d = start + timedelta(days=i)
        month = d.month
        seasonal = 1.0 + 0.4 * math.sin(2 * math.pi * (month - 3) / 12)
        trend = 1.0 + i / 730 * 0.5
        revenue = round(5000 * seasonal * trend + rng.gauss(0, 300), 2)
        orders = int(revenue / rng.uniform(45, 55))
        rows.append({
            "date": d.isoformat(),
            "revenue": max(0, revenue),
            "orders": max(0, orders),
            "avg_order_value": round(revenue / max(orders, 1), 2),
            "marketing_spend": round(rng.uniform(200, 800) * seasonal, 2),
            "channel": rng.choice(["organic", "paid_search", "email", "social"]),
        })

    with open(out / "ecommerce_daily.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated ecommerce_daily.csv ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# 2. Customer churn (classification / comparison)
# ---------------------------------------------------------------------------

def generate_churn():
    rows = []
    for i in range(2000):
        tenure_months = rng.randint(1, 72)
        plan = rng.choice(["basic", "standard", "premium"])
        monthly_spend = {"basic": 9.99, "standard": 19.99, "premium": 39.99}[plan]
        support_tickets = max(0, int(rng.gauss(2, 1.5)))

        # Churn probability: higher for short tenure, more tickets, basic plan
        churn_prob = (
            0.3 * (1 - min(tenure_months, 24) / 24)
            + 0.2 * min(support_tickets / 5, 1.0)
            + (0.3 if plan == "basic" else 0.1)
        )
        churned = int(rng.random() < churn_prob)

        rows.append({
            "customer_id": f"CUST_{i:05d}",
            "tenure_months": tenure_months,
            "plan": plan,
            "monthly_spend": monthly_spend,
            "support_tickets_6m": support_tickets,
            "login_days_last_30": max(0, int(rng.gauss(15, 8))),
            "nps_score": rng.randint(1, 10) if rng.random() > 0.2 else None,
            "churned": churned,
        })

    with open(out / "customer_churn.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated customer_churn.csv ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# 3. Employee salary (regression + clustering)
# ---------------------------------------------------------------------------

def generate_salary():
    departments = ["Engineering", "Sales", "Marketing", "Operations", "Finance"]
    levels = ["Junior", "Mid", "Senior", "Lead", "Director"]
    rows = []

    for i in range(500):
        dept = rng.choice(departments)
        level = rng.choice(levels)
        years_exp = {"Junior": rng.randint(0, 2), "Mid": rng.randint(2, 5),
                     "Senior": rng.randint(5, 10), "Lead": rng.randint(8, 15),
                     "Director": rng.randint(12, 25)}[level]

        base = {"Junior": 35000, "Mid": 55000, "Senior": 75000,
                "Lead": 95000, "Director": 130000}[level]
        dept_mult = {"Engineering": 1.25, "Sales": 1.1, "Marketing": 1.0,
                     "Operations": 0.9, "Finance": 1.15}[dept]

        salary = int(base * dept_mult + years_exp * 1500 + rng.gauss(0, 3000))

        rows.append({
            "employee_id": f"EMP_{i:04d}",
            "department": dept,
            "level": level,
            "years_experience": years_exp,
            "salary": max(25000, salary),
            "performance_score": round(rng.gauss(3.5, 0.6), 1),
            "remote": rng.choice([0, 1]),
            "city": rng.choice(["London", "Manchester", "Edinburgh", "Bristol", "Leeds"]),
        })

    with open(out / "employee_salary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated employee_salary.csv ({len(rows)} rows)")


if __name__ == "__main__":
    generate_ecommerce()
    generate_churn()
    generate_salary()
    print("Done. Sample datasets ready in data/sample_datasets/")
