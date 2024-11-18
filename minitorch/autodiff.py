from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol  # ,list


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # return (f(vals[0] + epsilon, *vals[1:]) - f(vals[0] - epsilon, *vals[1:])) / (2 * epsilon)
    vals_plus = list(vals)
    vals_minus = list(vals)
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon
    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate a derivative."""

    @property
    def unique_id(self) -> int:
        """Return the unique identifier for this variable."""
        ...

    def is_constant(self) -> bool:
        """Check if the variable is constant."""
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf in the computation graph."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return an iterable of the variable's parents."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule with the given output gradient."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    sorted_nodes = []

    def visit(v: Variable) -> None:
        if v in visited or v.is_constant():
            return
        else:
            # visited.add(v)
            for parent in v.parents:
                visit(parent)
            sorted_nodes.append(v)
            visited.add(v)

    visit(variable)
    return reversed(sorted_nodes)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The variable to start backpropagation from.
        deriv: The derivative of the output with respect to this variable.

    """
    ordered_variables = [*topological_sort(variable)]
    gradients = {variable: deriv}

    for var in ordered_variables:
        grad = gradients.get(var, 0)
        if var.is_leaf():
            var.accumulate_derivative(grad)
        else:
            for parent, parent_grad in var.chain_rule(grad):
                if parent_grad is None:
                    parent_grad = 0
                if parent not in gradients:
                    gradients[parent] = parent_grad
                else:
                    gradients[parent] += parent_grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors."""
        return self.saved_values
