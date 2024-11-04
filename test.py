#!/usr/bin/env python

import os
import sys
import traceback
from pathlib import Path

import numpy as np


def backprop(seed: int):
    from engine import Tensor

    ret = True
    try:
        np.random.seed(seed)
        a = Tensor(np.random.rand(2), req_grad=True)
        np.random.seed(seed + 42)
        b = Tensor(np.random.rand(3, 2), req_grad=True)

        c = a + b
    except Exception as e:
        print(f"FAILED! {e}\n{traceback.format_exc()}")
        ret = False
    else:
        try:  # Backward function
            print("Backward:")
            c.backward()
        except Exception as e:
            print(f"FAILED! {e}\n{traceback.format_exc()}")
            ret = False
        else:
            print("PASSED!")

        try:  # Zero grad function
            print("Zero grad:")
            c.zero_grad()
        except Exception as e:
            print(f"FAILED! {e}\n{traceback.format_exc()}")
            ret = False
        else:
            if np.allclose(a.grad, np.zeros((2,))) and np.allclose(
                b.grad, np.zeros((3, 2))
            ):
                print("PASSED!")
            else:
                print("FAILED! Gradient is left non-zero.")
                ret = False

        try:  # Step function
            print("Step:")
            np.random.seed(seed - 42)
            a.grad = np.random.rand(2) / 10
            np.random.seed(seed + 13)
            b.grad = np.random.rand(3, 2) / 10
            c.grad = b.grad * 0.987

            c.step(learning_rate=1)
        except Exception as e:
            print(f"FAILED! {e}\n{traceback.format_exc()}")
            ret = False
        else:
            a_ref = np.array([0.31965877, 0.87919537])
            b_ref = np.array(
                [
                    [0.03673016, 0.27307672],
                    [0.19393227, 0.60422773],
                    [0.93699768, 0.10203363],
                ]
            )
            if np.allclose(a.data, a_ref) and np.allclose(b.data, b_ref):
                print("PASSED!")
            else:
                print("FAILED! Results do not match reference.")
                ret = False
    return ret


def basic_operations(seed: int):
    from engine import Tensor

    np.random.seed(seed)
    a = np.random.rand(2, 2)
    a_t = Tensor(a)
    np.random.seed(seed + 1)
    b = np.random.rand(2, 2)
    b_t = Tensor(b)

    ret = True
    try:
        print("Basic operations:")
        if (a_t + b_t).data.all() != (a + b).all():
            print("Addition incorrect")
        if (a_t - b_t).data.all() != (a - b).all():
            print("Substraction incorrect")
        if (a_t * b_t).data.all() != (a * b).all():
            print("Multiplication incorrect")
        if (a_t / b_t).data.all() != (a / b).all():
            print("Division incorrect")
        if (a_t**3).data.all() != (a**3).all():
            print("Power incorrect")
    except Exception as e:
        print(f"FAILED! {e}\n{traceback.format_exc()}")
        ret = False
    else:
        print("PASSED!")

    return ret


def basic_functions(seed: int):
    from engine import Tensor

    np.random.seed(seed)
    a = np.random.rand(2, 2)
    a = -a if a.all() < 0 else a
    a_t = Tensor(a)

    ret = True
    try:
        print("Basic functions:")
        if (a_t.sin()).data.all() != np.sin(a).all():
            print("Sine function incorrect")
        if (a_t.cos()).data.all() != np.cos(a).all():
            print("Cosine function incorrect")
        if (a_t.exp()).data.all() != np.exp(a).all():
            print("Exponential function incorrect")
        if (a_t.log()).data.all() != np.log(a).all():
            print("Logarithm function incorrect")
    except Exception as e:
        print(f"FAILED! {e}\n{traceback.format_exc()}")
        ret = False
    else:
        print("PASSED!")

    return ret


def activation_functions(seed: int):
    from engine import Tensor

    np.random.seed(seed)
    a = np.random.rand(1)
    a = -a if a.all() < 0 else a
    a_t = Tensor(a)

    ret = True
    try:
        print("Activation functions:")
        if (a_t.relu()).data != max(a, 0):
            print("ReLU function incorrect")
        if (a_t.sigmoid()).data != (1 / (1 + np.exp(-a))):
            print("Sigmoid function incorrect")
        if (a_t.tanh()).data != np.tanh(a):
            print("Hyperbolic tangens function incorrect")
    except Exception as e:
        print(f"FAILED! {e}\n{traceback.format_exc()}")
        ret = False
    else:
        print("PASSED!")

    return ret


def backward(seed: int):
    from engine import Tensor

    np.random.seed(seed)
    a = Tensor(np.random.rand(1))
    np.random.seed(seed + 1)
    b = Tensor(np.random.rand(1))

    ret = True
    try:
        print("Backward pass:")
        c = a + b
        d = a * (b - 3)
        e = c.cos() / a.sin()
        f = d.exp().log() ** -2
        g = e.relu() + f.sigmoid()

        g.backward()
        res = np.array([a.grad, b.grad, c.grad, d.grad, e.grad, f.grad, g.grad])
    except Exception as e:
        print(f"FAILED! {e}\n{traceback.format_exc()}")
        ret = False
    else:
        expected = np.array(
            [
                [-8.378998],
                [-1.16119983],
                [-1.28543176],
                [0.33169193],
                [1.0],
                [0.2092246],
                [1.0],
            ]
        )
        if np.allclose(res, expected):
            print("PASSED!")
        else:
            print("FAILED! Result does not match expected results.")

    return ret


if __name__ == "__main__":
    module_path = Path("./engine.py")
    if not os.path.exists(module_path):
        print("Module file engine could not be found")
        sys.exit(1)

    try:
        from engine import Tensor
    except Exception as e:
        print(f"Error importing module: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    seed = 42  # DO NOT CHANGE! or certain test will not work

    if not backprop(seed):
        sys.exit(1)
    basic_operations(seed)
    basic_functions(seed)
    activation_functions(seed)
    backward(seed)
