import pandas as pd
import numpy as np


def generate_customer_churn_data(n=5000, random_state=42):
    np.random.seed(random_state)

    customer_id = np.arange(1, n + 1)
    age = np.random.randint(18, 75, n)
    tenure_months = np.random.randint(1, 121, n)
    monthly_spend = np.random.uniform(50, 2500, n).round(2)
    support_calls = np.random.poisson(2, n)
    complaints = np.random.binomial(4, 0.15, n)
    digital_engagement_score = np.random.uniform(0, 100, n).round(2)
    payment_failed_flag = np.random.binomial(1, 0.12, n)
    contract_type = np.random.choice(
        ["Monthly", "Quarterly", "Annual"],
        size=n,
        p=[0.55, 0.20, 0.25]
    )
    product_count = np.random.randint(1, 6, n)
    region = np.random.choice(
        ["Western Cape", "Gauteng", "KwaZulu-Natal", "Eastern Cape", "Limpopo"],
        size=n
    )

    contract_risk = {
        "Monthly": 0.9,
        "Quarterly": 0.4,
        "Annual": -0.6,
    }

    region_risk = {
        "Western Cape": 0.0,
        "Gauteng": 0.1,
        "KwaZulu-Natal": 0.15,
        "Eastern Cape": 0.2,
        "Limpopo": 0.1,
    }

    logit = (
        -1.8
        - 0.012 * tenure_months
        + 0.0005 * monthly_spend
        + 0.30 * support_calls
        + 0.55 * complaints
        - 0.018 * digital_engagement_score
        + 0.95 * payment_failed_flag
        - 0.22 * product_count
        + np.vectorize(contract_risk.get)(contract_type)
        + np.vectorize(region_risk.get)(region)
        + np.random.normal(0, 0.5, n)
    )

    churn_probability = 1 / (1 + np.exp(-logit))
    churn = np.random.binomial(1, churn_probability)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "tenure_months": tenure_months,
        "monthly_spend": monthly_spend,
        "support_calls": support_calls,
        "complaints": complaints,
        "digital_engagement_score": digital_engagement_score,
        "payment_failed_flag": payment_failed_flag,
        "contract_type": contract_type,
        "product_count": product_count,
        "region": region,
        "churn_probability_true": churn_probability.round(4),
        "churn": churn,
    })

    return df


if __name__ == "__main__":
    df = generate_customer_churn_data()
    df.to_csv("customer_churn_data.csv", index=False)

    print("Dataset created: customer_churn_data.csv")
    print(df.head())
    print("\nChurn rate:")
    print(df["churn"].mean().round(4))