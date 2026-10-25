"""Plain EMI arithmetic. Scheme numbers (subsidy, caps) are never hardcoded: the user enters what the lender/portal confirms."""


def emi(principal: float, annual_rate_pct: float, years: int) -> float:
    n = int(years * 12)
    r = annual_rate_pct / 1200
    if principal <= 0 or n <= 0:
        return 0.0
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def burden(emi_amount: float, existing_emi: float, monthly_income: float) -> float:
    """Share of monthly income going to loan repayments."""
    return (emi_amount + existing_emi) / monthly_income if monthly_income > 0 else 0.0
