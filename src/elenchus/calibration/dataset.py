"""Calibration dataset — math problems with known answers for prompt optimization."""

from __future__ import annotations

# Hand-curated problems covering algebra domain types.
# Each has an exact numeric answer for evaluation.
# Categories: compound_interest, kinematics, exponential, rate, equation, system

_PROBLEMS: list[dict] = [
    # === Compound Interest ===
    {
        "question": "$10,000 invested at 5% annual interest compounded monthly. What is the value after 3 years?",
        "expected_answer": 11614.72,
        "category": "compound_interest",
    },
    {
        "question": "$5,000 invested at 8% annual interest compounded quarterly. What is the value after 5 years?",
        "expected_answer": 7429.74,
        "category": "compound_interest",
    },
    {
        "question": "$20,000 invested at 3.5% annual interest compounded monthly. What is the value after 10 years?",
        "expected_answer": 28366.90,
        "category": "compound_interest",
    },
    {
        "question": "$1,000 invested at 12% annual interest compounded monthly. What is the value after 2 years?",
        "expected_answer": 1269.73,
        "category": "compound_interest",
    },
    # === Kinematics ===
    {
        "question": "A ball is thrown upward from 20 meters with velocity 15 m/s. What is the maximum height? (g=9.8 m/s^2)",
        "expected_answer": 31.48,
        "category": "kinematics",
    },
    {
        "question": "A car accelerates from rest at 3 m/s^2 for 8 seconds. How far does it travel?",
        "expected_answer": 96.0,
        "category": "kinematics",
    },
    {
        "question": "A projectile is launched at 25 m/s at 45 degrees. What is its range? (g=9.8 m/s^2)",
        "expected_answer": 63.78,
        "category": "kinematics",
    },
    {
        "question": "An object falls from 100 meters. How long until it hits the ground? (g=9.8 m/s^2)",
        "expected_answer": 4.52,
        "category": "kinematics",
    },
    # === Exponential Decay ===
    {
        "question": "A substance has a half-life of 5 years. Starting with 200g, how much remains after 12 years?",
        "expected_answer": 37.15,
        "category": "exponential",
    },
    {
        "question": "A radioactive sample decays at 10% per year. Starting with 500g, how much remains after 8 years?",
        "expected_answer": 214.99,
        "category": "exponential",
    },
    {
        "question": "A population of bacteria doubles every 3 hours. Starting with 100, how many after 15 hours?",
        "expected_answer": 3200.0,
        "category": "exponential",
    },
    {
        "question": "A car depreciates by 15% per year. If it cost $30,000, what is it worth after 6 years?",
        "expected_answer": 11314.68,
        "category": "exponential",
    },
    # === Rate Problems ===
    {
        "question": "A pool fills at 50 gallons/hour and drains at 20 gallons/hour. The pool holds 600 gallons. How many hours to fill it?",
        "expected_answer": 20.0,
        "category": "rate",
    },
    {
        "question": "Worker A completes a job in 6 hours. Worker B completes it in 4 hours. How long working together?",
        "expected_answer": 2.4,
        "category": "rate",
    },
    {
        "question": "A train travels 240 km. If it went 20 km/h faster, the trip would take 1 hour less. What is its speed?",
        "expected_answer": 60.0,
        "category": "rate",
    },
    {
        "question": "Two pipes fill a tank. Pipe A fills it in 5 hours, pipe B in 8 hours. How long to fill it together?",
        "expected_answer": 3.08,
        "category": "rate",
    },
    # === Simple Equations ===
    {
        "question": "Solve for x: 3x + 7 = 22",
        "expected_answer": 5.0,
        "category": "equation",
    },
    {
        "question": "Solve for x: 2x^2 - 8 = 0 (positive root)",
        "expected_answer": 2.0,
        "category": "equation",
    },
    {
        "question": "Solve for x: 5(x - 3) = 2(x + 6)",
        "expected_answer": 9.0,
        "category": "equation",
    },
    {
        "question": "What is the sum of the roots of x^2 - 7x + 12 = 0?",
        "expected_answer": 7.0,
        "category": "equation",
    },
    # === Systems ===
    {
        "question": "Solve: 2x + y = 10 and x - y = 2. What is x?",
        "expected_answer": 4.0,
        "category": "system",
    },
    {
        "question": "Solve: 3x + 2y = 16 and x + y = 6. What is y?",
        "expected_answer": 2.0,
        "category": "system",
    },
    {
        "question": "A rectangle has perimeter 30 and area 50. What is the length of the longer side?",
        "expected_answer": 10.0,
        "category": "system",
    },
    {
        "question": "The sum of two numbers is 15 and their product is 56. What is the larger number?",
        "expected_answer": 8.0,
        "category": "system",
    },
]


def load_calibration_problems() -> list[dict]:
    """Return the calibration problem corpus.

    Each dict has keys: question, expected_answer, category.
    """
    return list(_PROBLEMS)
