from __future__ import annotations


def forward(inputs, state):
    output = []
    for value in inputs:
        state[0] = 0.75 * state[0] + value
        output.append(state[0])
    return output
