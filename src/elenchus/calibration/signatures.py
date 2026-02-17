"""DSPy signatures for councilor prompt optimization."""

import dspy


class MathSolver(dspy.Signature):
    """Solve a math problem and return the numeric answer with reasoning."""

    problem: str = dspy.InputField(desc="A math problem to solve")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning to solve the problem")
    answer: float = dspy.OutputField(desc="The final numeric answer")


class NumericalMathSolver(dspy.Signature):
    """Solve a math problem using numerical estimation and verification."""

    problem: str = dspy.InputField(desc="A math problem to solve")
    estimate: str = dspy.OutputField(desc="Initial estimate with rough mental math")
    reasoning: str = dspy.OutputField(desc="Detailed numerical computation steps")
    answer: float = dspy.OutputField(desc="The final numeric answer")


class AlgebraicMathSolver(dspy.Signature):
    """Solve a math problem using algebraic manipulation."""

    problem: str = dspy.InputField(desc="A math problem to solve")
    variables: str = dspy.OutputField(desc="Variables identified and their meanings")
    reasoning: str = dspy.OutputField(desc="Step-by-step algebraic derivation")
    answer: float = dspy.OutputField(desc="The final numeric answer")
