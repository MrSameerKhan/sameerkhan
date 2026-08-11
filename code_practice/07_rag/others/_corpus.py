# Shared financial policy corpus — imported by all Phase 07 RAG sessions.
# 8 documents covering mortgage, personal finance, accounts, and credit cards.

DOCUMENTS = [
    {
        "id": "mortgage_ltv",
        "title": "Residential Mortgage Policy — LTV Requirements",
        "content": (
            "The maximum loan-to-value (LTV) ratio for standard residential mortgage applications "
            "is 90% of the property's assessed value for standard borrowers who meet income verification "
            "requirements. First-time buyers are eligible for an enhanced LTV of up to 95% under the "
            "government-backed Help to Buy scheme. To qualify, applicants must have no prior property "
            "ownership and the property must be a new-build valued under £600,000. "
            "For buy-to-let properties, the maximum LTV is 75%. Rental income must demonstrate 125% "
            "coverage of the monthly mortgage payment at a stressed interest rate of 5.5%. "
            "Investment properties and holiday homes are capped at 70% LTV and require a minimum "
            "6-month rental history or evidence of rental demand."
        ),
    },
    {
        "id": "mortgage_income",
        "title": "Residential Mortgage Policy — Income and Affordability",
        "content": (
            "All mortgage applications require proof of income for the last 3 months. Acceptable "
            "documents include payslips for employed applicants, SA302 tax returns for self-employed "
            "applicants (minimum 2 years), and pension statements for retired applicants. "
            "Affordability is assessed at a stressed interest rate 3% above the product rate. Total "
            "monthly debt obligations must not exceed 45% of gross monthly income. "
            "For joint applications, income from both applicants is considered. If one applicant has "
            "adverse credit, the combined application may be assessed at a lower LTV ceiling of 85%. "
            "Bonus income, commission, and overtime are accepted at 50% of the 12-month average."
        ),
    },
    {
        "id": "mortgage_rates",
        "title": "Residential Mortgage Policy — Interest Rates and Terms",
        "content": (
            "Fixed-rate mortgage products are available for initial periods of 2, 3, 5, and 10 years. "
            "After the fixed period, the rate reverts to the lender's Standard Variable Rate (SVR), "
            "currently 7.24%. Early repayment charges (ERC) apply during the fixed-rate period: "
            "5% in year 1, 4% in year 2, 3% in year 3, 2% in year 4, 1% in year 5. No ERC applies "
            "after the fixed period ends. Tracker mortgage products follow the Bank of England Base "
            "Rate plus a fixed margin. Customers on tracker products may overpay by up to 10% of the "
            "outstanding balance per year without incurring charges. "
            "Interest-only mortgages are available for LTV up to 75%, subject to an acceptable "
            "repayment vehicle (ISA, endowment policy, or investment portfolio)."
        ),
    },
    {
        "id": "personal_finance_eligibility",
        "title": "Personal Finance Policy — Eligibility and Limits",
        "content": (
            "Personal finance products are available to customers aged 21-65 with a minimum monthly "
            "salary of SAR 4,000. Applicants must have held a salary transfer account with the bank "
            "for a minimum of 3 months. The maximum personal finance amount is 36× the monthly salary "
            "for Saudi nationals and 24× for expatriate residents. The maximum tenor is 60 months "
            "(5 years) for Saudi nationals and 36 months for expatriates. "
            "Customers with existing personal finance facilities must demonstrate that total monthly "
            "obligations do not exceed 33% of gross salary (the Debt Burden Ratio mandated by the "
            "Saudi Central Bank). Refinancing is permitted after 6 months of satisfactory repayment, "
            "subject to a processing fee of 1% of the new finance amount."
        ),
    },
    {
        "id": "personal_finance_rates",
        "title": "Personal Finance Policy — Profit Rates and Sharia Compliance",
        "content": (
            "All personal finance products are structured as Murabaha contracts, complying with "
            "Islamic finance principles by avoiding Riba (interest). The bank purchases the goods "
            "or commodity on behalf of the customer and resells at a fixed profit margin. "
            "The annual percentage rate (APR) equivalent ranges from 5.5% to 9.5%, depending on "
            "the customer's salary band, credit score, and employment sector. Government employees "
            "typically qualify for the lowest rates. A flat rate of 3.5% approximately equates to "
            "an APR of 6.7% over a 60-month tenor. Early settlement is permitted at any time. "
            "Customers are entitled to a rebate on unearned profit, calculated on a pro-rata basis "
            "per the Sharia board-approved methodology."
        ),
    },
    {
        "id": "accounts_current_savings",
        "title": "Account Services Policy — Current and Savings Accounts",
        "content": (
            "Current accounts are available to individuals aged 18 and above with valid national "
            "identification. The daily ATM cash withdrawal limit is SAR 5,000 for standard accounts "
            "and SAR 20,000 for premium accounts. International ATM withdrawals are subject to a "
            "transaction fee of SAR 20 plus 1.75% of the withdrawn amount. "
            "Savings accounts accrue returns at an indicative rate of 2.8% per annum, paid monthly. "
            "No minimum balance is required to earn returns, but accounts below SAR 500 are subject "
            "to a monthly maintenance fee of SAR 10. Joint accounts require consent from all holders "
            "for transactions above SAR 10,000."
        ),
    },
    {
        "id": "accounts_transfers",
        "title": "Account Services Policy — Transfers and Payments",
        "content": (
            "Domestic SARIE transfers are processed immediately 24/7. International SWIFT transfers "
            "are processed on the next business day if submitted before 2:00 PM. The maximum single "
            "international transfer is SAR 100,000 per day for personal accounts. "
            "Foreign currency exchange is available for 25 currencies. The bank's exchange rate "
            "includes a spread of 1.5% over the mid-market rate. Currency orders above USD 10,000 "
            "equivalent require prior arrangement with the treasury desk. "
            "Suspicious transaction alerts are generated for: transactions above 3× the customer's "
            "average monthly value, transactions to new beneficiaries above SAR 50,000, and multiple "
            "rapid transactions. Failed payment fees are SAR 30 per returned item."
        ),
    },
    {
        "id": "credit_card",
        "title": "Credit Card Policy — Eligibility and Limits",
        "content": (
            "Credit cards are available to customers with a minimum monthly salary of SAR 5,000. "
            "The credit limit is set at 3× the monthly salary for entry-level cards and up to 12× "
            "for premium cards, subject to credit bureau assessment. Applications require: valid "
            "national ID, proof of salary, and a minimum 6-month banking relationship. Expatriate "
            "customers require a valid Iqama with minimum 12 months validity. "
            "Supplementary cards can be issued for immediate family members above age 18. The primary "
            "cardholder is liable for all spending on supplementary cards, capped at 50% of the "
            "primary card limit. Credit limit increases can be requested after 6 months of satisfactory "
            "repayment. Customers must have no missed payments in the preceding 12 months to qualify."
        ),
    },
]
