import sympy as sp
sp.init_printing(use_unicode=True)

t, Ω, λp, λm, k1, k2 = sp.symbols('t Ω λp λm k1, k2')

c1 = k1*Ω/sp.sqrt(Ω**2 + 4*λp**2)*sp.exp(sp.I * λp * t) + \
    k2*λm/sp.sqrt(Ω**2 + 4*λm**2)*sp.exp(sp.I * λm * t)

cr = k1*λp/sp.sqrt(Ω**2 + 4*λp**2)*sp.exp(sp.I * λp * t) + \
    k2*λm/sp.sqrt(Ω**2 + 4*λm**2)*sp.exp(sp.I * λm * t)

norm = sp.Abs(c1)**2 + sp.Abs(cr)**2

print(sp.simplify(norm))
