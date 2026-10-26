"""Intake profile and how it becomes retrieval queries. No eligibility rules live here."""
from dataclasses import dataclass


@dataclass
class Profile:
    occupation: str          # e.g. "Factory / industrial worker"
    monthly_income: int      # INR
    city: str
    state: str
    gender: str
    is_migrant: bool         # moved from another city/state for work or study
    owns_pucca_house: bool   # self or family, anywhere in India
    need: str                # "Short-term rental" / "Long-term rental" / "Own a house" / "Not sure"
    employer_housing: bool
    income_proof: str = "Not specified"   # Salary slips / Bank statements only / Mostly cash, no proof
    existing_emi: int = 0                # INR per month already going to other loans

    def summary(self) -> str:
        return (
            f"Occupation: {self.occupation}; monthly income: Rs {self.monthly_income:,} "
            f"(annual Rs {self.monthly_income * 12:,}); location: {self.city}, {self.state}; "
            f"gender: {self.gender}; migrant: {'yes' if self.is_migrant else 'no'}; "
            f"family owns pucca house in India: {'yes' if self.owns_pucca_house else 'no'}; "
            f"housing need: {self.need}; employer provides housing: {'yes' if self.employer_housing else 'no'}; "
            f"income proof: {self.income_proof}; existing monthly EMIs: Rs {self.existing_emi:,}"
        )

    def wants_loan(self) -> bool:
        return self.need in ("Own a house", "Not sure")

    def loan_queries(self) -> list[str]:
        return [
            "interest subsidy scheme eligibility income limit loan amount property value",
            "interest subsidy rate tenure maximum subsidy how it is credited",
            "credit risk guarantee fund collateral free housing loan EWS LIG",
            "primary lending institutions banks housing finance companies eligible",
            "documents required for housing loan application income proof",
            "beneficiary led construction or affordable housing in partnership beneficiary contribution loan",
        ]

    def queries(self) -> list[str]:
        q = []
        if "rental" in self.need.lower():  # first, so the rental passages are never crowded out
            q += ["Affordable Rental Housing ARH vertical who is eligible rental housing for EWS LIG beneficiaries urban migrants workers",
                  "ARH rent rules concessionaire rental housing complexes how rent is paid"]
        if "own" in self.need.lower():
            q += ["interest subsidy scheme eligibility home loan subsidy",
                  "beneficiary led construction affordable housing in partnership eligibility"]
        q += [
            f"eligibility for {self.occupation} housing scheme",
            "annual household income limit EWS LIG MIG eligibility",
            "beneficiary must not own a pucca house",
            "documents required to apply and how to apply",
        ]
        if self.is_migrant:
            q.append("urban migrants rental housing eligibility")
        if self.gender.lower() == "female":
            q.append("women working women hostel rental housing benefits")
        if self.employer_housing:
            q.append("employer provided housing industrial workers rental")
        q.append(f"{self.state} state implementation housing scheme")
        return q
