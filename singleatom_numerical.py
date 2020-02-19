import pickle as rick 
import numpy as np
import matplotlib as plt
from scipy.integrate import odeint


def dU_dx(U, x):
    return