import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N = 5000

def main():
    rng = np.random.default_rng(RANDOM_STATE)

    amount = np.round(rng.lognormal(mean=5.0, sigma=1.0, size=N), 2)
    hour = rng.integers(0, 24, N)
    distance = np.round(rng.gamma(shape=2.0, scale=35.0, size=N), 2)
    account_age = rng.integers(1, 3000, N)
    transactions_24h = rng.poisson(lam=3, size=N)
    failed_attempts = rng.poisson(lam=0.4, size=N)
    is_international = rng.binomial(1, 0.18, N)
    is_new_device = rng.binomial(1, 0.15, N)

    score = (
        -4.0
        + 0.0009 * amount
        + 0.75 * is_international
        + 0.90 * is_new_device
        + 0.55 * (failed_attempts >= 2)
        + 0.18 * transactions_24h
        + 0.012 * distance
        + 0.55 * ((hour <= 4) | (hour >= 23))
        - 0.00015 * account_age
    )

    probability = 1 / (1 + np.exp(-score))
    probability = np.clip(
        probability + rng.normal(0, 0.06, N),
        0.001,
        0.999
    )

    high_risk = rng.binomial(1, probability)

    df = pd.DataFrame({
        "amount": amount,
        "hour": hour,
        "distance_from_usual_km": distance,
        "account_age_days": account_age,
        "transactions_24h": transactions_24h,
        "failed_attempts": failed_attempts,
        "is_international": is_international,
        "is_new_device": is_new_device,
        "high_risk": high_risk,
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)

    print("Dataset created successfully!")

if __name__ == "__main__":
    main()
