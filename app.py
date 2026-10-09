import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, convert_xor, implicit_multiplication_application,
)

st.set_page_config(page_title="Derivative & Tangent Line Visualizer", layout="wide")

# ============================================================
# 1) Math parser: แปลงข้อความสมการเป็นฟังก์ชันที่ใช้กับ numpy ได้
# ============================================================
X, A, B, C = sp.symbols("x A B C")
PARAM_SYMBOLS = [A, B, C]
TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)


def _log(v, base=None):
    """log(x) = ฐาน 10,  log(x, b) = ฐาน b"""
    return sp.log(v) / sp.log(10 if base is None else base)


LOCALS = {
    "x": X, "A": A, "B": B, "C": C,
    "ln": sp.log,        # ln(x)   = ลอการิทึมธรรมชาติ (ฐาน e)
    "log": _log,         # log(x)  = ฐาน 10
    "sqrt": sp.sqrt,     # sqrt(x) = รากที่สอง
    "abs": sp.Abs, "pi": sp.pi, "e": sp.E,
}


def make_f(text, params):
    """คืนค่า (f, expr, df): f = ฟังก์ชันตัวเลข, expr = นิพจน์ sympy, df = อนุพันธ์จริง (ไว้เทียบ error)"""
    try:
        expr = parse_expr(text, local_dict=LOCALS, transformations=TRANSFORMS)
    except Exception:
        raise ValueError("รูปแบบสมการไม่ถูกต้อง (ตรวจวงเล็บและเครื่องหมายอีกครั้ง)")
    expr = expr.subs(params)
    unknown = expr.free_symbols - {X}
    if unknown:
        raise ValueError(f"มีตัวแปรที่ไม่รู้จัก: {', '.join(map(str, unknown))}")

    fn = sp.lambdify(X, expr, "numpy")
    dfn = sp.lambdify(X, sp.diff(expr, X), "numpy")

    def wrap(g):
        def inner(x):
            x = np.asarray(x, dtype=float)
            with np.errstate(all="ignore"):          # ln/sqrt ของค่าลบ -> nan (กราฟจะเว้นช่วงนั้น)
                y = np.asarray(g(x), dtype=float) * np.ones_like(x)
            return y
        return inner

    return wrap(fn), expr, wrap(dfn)


# ============================================================
# 2) Numerical methods
# ============================================================
def central_diff(f, x, h):
    """f'(x) ≈ (f(x+h) - f(x-h)) / 2h"""
    return (f(x + h) - f(x - h)) / (2 * h)


def newton(g, x0, tol=1e-10, max_iter=50):
    """Newton-Raphson: x_{n+1} = x_n - g(x_n)/g'(x_n)  (g' หาด้วย central difference)"""
    x = float(x0)
    steps = [(0, x, float(g(x)))]
    for n in range(1, max_iter + 1):
        d = float(central_diff(g, x, 1e-6))
        if not np.isfinite(d) or abs(d) < 1e-12:
            break
        xn = x - float(g(x)) / d
        if not np.isfinite(xn):
            break
        steps.append((n, xn, float(g(xn))))
        done = abs(xn - x) < tol
        x = xn
        if done:
            break
    return x, steps, abs(float(g(x))) < 1e-6


def fmt(v, d=4):
    if not np.isfinite(v):
        return "—"
    return f"{v:.2e}" if (v != 0 and abs(v) < 1e-4) else f"{v:.{d}f}"


# ============================================================
# 3) ตัวอย่างประยุกต์ใช้กับปัญหาจริง
#    params = [(ชื่อที่แสดง, ค่าเริ่มต้น)] แทนตัวแปร A, B, C ตามลำดับ
# ============================================================
PRESETS = {
    "custom": dict(name="กำหนดเอง (พิมพ์ฟังก์ชันเอง)", expr="A*sin(B*x + C)",
                   params=[("A", 1.0), ("B", 1.0), ("C", 0.0)],
                   xr=(-6.0, 6.0), xl="x", yl="f(x)", sl="f'(a)"),
    "quad": dict(name="ทั่วไป: ax² + bx + c", expr="A*x^2 + B*x + C",
                 params=[("a", 1.0), ("b", 0.0), ("c", 0.0)],
                 xr=(-5.0, 5.0), xl="x", yl="f(x)", sl="f'(a)"),
    "exp": dict(name="ทั่วไป: A·e^(Bx)", expr="A*exp(B*x)",
                params=[("A", 1.0), ("B", 0.3)],
                xr=(-3.0, 5.0), xl="x", yl="f(x)", sl="f'(a)"),
    "position": dict(name="ฟิสิกส์: ตำแหน่ง → ความเร็ว", expr="A*x + 0.5*B*x^2",
                     params=[("v0 ความเร็วต้น (m/s)", 5.0), ("a ความเร่ง (m/s²)", 2.0)],
                     xr=(0.0, 10.0), xl="time t (s)", yl="position s (m)", sl="v (m/s)"),
    "projectile": dict(name="ฟิสิกส์: วิถีโพรเจกไทล์",
                       expr="x*tan(A*pi/180) - 9.8*x^2/(2*B^2*cos(A*pi/180)^2)",
                       params=[("มุมขว้าง (องศา)", 45.0), ("ความเร็วต้น (m/s)", 20.0)],
                       xr=(0.0, 45.0), xl="horizontal distance x (m)", yl="height y (m)", sl="dy/dx"),
    "logistic": dict(name="ชีววิทยา: การเติบโตของประชากร (logistic)",
                     expr="C/(1 + ((C-A)/A)*exp(-B*x))",
                     params=[("P0 ประชากรเริ่มต้น", 10.0), ("r อัตราการเติบโต", 0.6), ("K ความจุสิ่งแวดล้อม", 500.0)],
                     xr=(0.0, 20.0), xl="time t (day)", yl="population P(t)", sl="dP/dt"),
}

st.title("Interactive Derivative & Tangent Line Visualizer")
st.caption("ฟังก์ชันที่ใช้ได้: sin cos tan exp sqrt(x) ln(x) [ฐาน e] log(x) [ฐาน 10] log(x, b) abs และ pi, e  |  ใช้ ^ แทนยกกำลัง")

tab1, tab2 = st.tabs(["Mode 1: Tangent Line", "Mode 2: Intersection Finder"])

# ============================================================
# 4) Mode 1: เส้นสัมผัส
#    ทุก widget ใส่ key ที่มีชื่อ preset เพื่อให้ค่าเริ่มต้นรีเซ็ตเมื่อเปลี่ยนตัวอย่าง
# ============================================================
with tab1:
    key = st.selectbox("ตัวอย่าง / โหมดฟังก์ชัน", list(PRESETS), format_func=lambda k: PRESETS[k]["name"])
    P = PRESETS[key]

    expr_text = st.text_input("ฟังก์ชัน f(x)  (ตัวแปร x, A, B, C)", P["expr"],
                              disabled=(key != "custom"), key=f"{key}_expr")

    c1, c2, c3, c4 = st.columns(4)
    a = c1.number_input("จุดสัมผัส a", value=(P["xr"][0] + P["xr"][1]) / 2, format="%g", key=f"{key}_a")
    h = c2.number_input("ค่า h", value=0.001, format="%g", key=f"{key}_h")
    x0 = c3.number_input("x ต่ำสุด", value=P["xr"][0], format="%g", key=f"{key}_x0")
    x1 = c4.number_input("x สูงสุด", value=P["xr"][1], format="%g", key=f"{key}_x1")

    pcols = st.columns(len(P["params"]))
    pvals = {}
    for i, (label, default) in enumerate(P["params"]):
        pvals[PARAM_SYMBOLS[i]] = pcols[i].number_input(
            f"{label}  ({PARAM_SYMBOLS[i]})", value=default, format="%g", key=f"{key}_p{i}")

    k1, k2 = st.columns(2)
    show_secant = k1.checkbox("แสดงเส้นเซแคนต์", True)
    show_deriv = k2.checkbox("แสดงกราฟ f'(x)", True)

    if h <= 0:
        st.error("ค่า h ต้องมากกว่า 0")
    elif x1 <= x0:
        st.error("x สูงสุดต้องมากกว่า x ต่ำสุด")
    else:
        try:
            f, expr, df = make_f(expr_text, pvals)
        except Exception as ex:
            st.error(f"สมการไม่ถูกต้อง: {ex}")
        else:
            xs = np.linspace(x0, x1, 500)
            ys = f(xs)
            fin = ys[np.isfinite(ys)]
            if fin.size == 0:
                st.error("คำนวณค่าฟังก์ชันในช่วงนี้ไม่ได้ (ตรวจโดเมน เช่น ln และ sqrt ต้องการ x > 0)")
            else:
                fa = float(f(a))
                m = float(central_diff(f, a, h))          # ความชันที่ h ที่เลือก
                ref = float(central_diff(f, a, 1e-6))     # ค่าอ้างอิง (h เล็กมาก)
                exact = float(df(a))                      # อนุพันธ์จริงจาก sympy
                ok = np.isfinite(m) and np.isfinite(fa)

                ncols = 2 if show_deriv else 1
                fig, axs = plt.subplots(1, ncols, figsize=(6.5 * ncols, 4.5))
                ax = axs[0] if ncols == 2 else axs
                ax.plot(xs, ys, color="#2563eb", lw=2.5, label="f(x)")
                if ok:
                    span = (x1 - x0) * 0.3
                    tx = np.array([a - span, a + span])
                    ax.plot(tx, fa + m * (tx - a), "r--", label="tangent line")
                    ms = (float(f(a + h)) - fa) / h
                    if show_secant and np.isfinite(ms):
                        ax.plot(tx, fa + ms * (tx - a), "g:", label=f"secant (h={h:.1e})")
                    ax.plot(a, fa, "ko")
                pad = 0.2 * ((fin.max() - fin.min()) or 1)
                ax.set_ylim(fin.min() - pad, fin.max() + pad)
                ax.set_xlabel(P["xl"]); ax.set_ylabel(P["yl"]); ax.grid(alpha=.3); ax.legend()
                ax.set_title("f(x) and tangent line")

                if show_deriv:
                    ax2 = axs[1]
                    ax2.plot(xs, central_diff(f, xs, 1e-5), color="purple", lw=2.5, label="f'(x)")
                    ax2.axhline(0, color="gray", lw=.8)
                    if ok:
                        ax2.plot(a, m, "ro")
                    ax2.set_xlabel(P["xl"]); ax2.set_ylabel("f'(x)"); ax2.grid(alpha=.3); ax2.legend()
                    ax2.set_title("derivative f'(x)")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("ขั้นตอนการคำนวณ")
                if not ok:
                    st.warning(f"คำนวณอนุพันธ์ที่ a = {a:g} ไม่ได้ เพราะฟังก์ชันไม่มีค่าที่ a−h, a หรือ a+h "
                               "(อาจอยู่นอกโดเมน เช่น ln หรือ sqrt ของค่าลบ) ลองเลือก a ให้อยู่ในโดเมน หรือลดค่า h")
                else:
                    st.code(
                        f"f(x) = {sp.sstr(expr)}\n"
                        f"จุดสัมผัส: a = {a:.3f},  f(a) = {fmt(fa)}\n"
                        f"{P['sl']} ≈ [f(a+h) - f(a-h)] / 2h = {fmt(m, 6)}   (h = {h:.1e})\n"
                        f"ค่าอ้างอิง (h = 1e-6)   = {fmt(ref, 6)}\n"
                        f"อนุพันธ์จริง (sympy)    = {fmt(exact, 6)}\n"
                        f"คลาดเคลื่อนของ h ที่เลือก = {abs(m - exact):.2e}\n"
                        f"สมการเส้นสัมผัส: y - {fmt(fa, 3)} = {fmt(m, 3)}(x - {a:.3f})\n"
                        f"               y = {fmt(m, 3)}x + {fmt(fa - m * a, 3)}",
                        language=None)

# ============================================================
# 5) Mode 2: หาจุดตัดด้วย Newton-Raphson
# ============================================================
with tab2:
    expr2 = st.text_input("ฟังก์ชัน f(x)", "x^3 - 2*x", key="m2_expr")
    d1, d2, d3, d4 = st.columns(4)
    m_ = d1.number_input("ค่า m (y = mx + c)", value=1.0, format="%g", key="m2_m")
    c_ = d2.number_input("ค่า c", value=0.0, format="%g", key="m2_c")
    xmin = d3.number_input("x ต่ำสุด", value=-4.0, format="%g", key="m2_xmin")
    xmax = d4.number_input("x สูงสุด", value=4.0, format="%g", key="m2_xmax")

    if xmax <= xmin:
        st.error("x สูงสุดต้องมากกว่า x ต่ำสุด")
    else:
        try:
            f2, _, _ = make_f(expr2, {})
        except Exception as ex:
            st.error(f"สมการไม่ถูกต้อง: {ex}")
        else:
            g = lambda t: float(f2(t)) - (m_ * t + c_)     # จุดตัดคือรากของ g(x) = 0

            # สแกนหาจุดเริ่มต้น: ช่วงที่เครื่องหมายเปลี่ยน + จุดที่ |g| ต่ำสุดเฉพาะที่
            N = 800
            xs = np.linspace(xmin, xmax, N + 1)
            gs = f2(xs) - (m_ * xs + c_)
            seeds = []
            for i in range(N):
                if np.isfinite(gs[i]) and np.isfinite(gs[i + 1]) and gs[i] * gs[i + 1] <= 0:
                    seeds.append((xs[i] + xs[i + 1]) / 2)
                if i > 0 and np.isfinite(gs[i]) and abs(gs[i]) <= abs(gs[i - 1]) and abs(gs[i]) <= abs(gs[i + 1]):
                    seeds.append(xs[i])

            roots = []                                      # (x, steps)
            for s in seeds:
                r, steps, ok = newton(g, s)
                if ok and xmin - 1e-9 <= r <= xmax + 1e-9 and all(abs(r - q[0]) > 1e-6 for q in roots):
                    roots.append((r, steps))
            roots.sort(key=lambda q: q[0])

            ys2 = f2(xs)
            fin2 = ys2[np.isfinite(ys2)]
            if fin2.size == 0:
                st.error("คำนวณค่าฟังก์ชันในช่วงนี้ไม่ได้ (ตรวจโดเมนของฟังก์ชัน)")
            else:
                lx = np.array([xmin, xmax])
                allv = np.concatenate([fin2, m_ * lx + c_])
                pad = 0.2 * ((allv.max() - allv.min()) or 1)
                fig, ax = plt.subplots(figsize=(9, 4.5))
                ax.plot(xs, ys2, color="#2563eb", lw=2.5, label="f(x)")
                ax.plot(lx, m_ * lx + c_, "r--", label="y = mx + c")
                for r, _ in roots:
                    ax.plot(r, float(f2(r)), "ko")
                    ax.annotate(f"({r:.3f}, {float(f2(r)):.3f})", (r, float(f2(r))),
                                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
                ax.set_ylim(allv.min() - pad, allv.max() + pad)
                ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=.3); ax.legend()
                ax.set_title("intersection of f(x) and y = mx + c")
                st.pyplot(fig)
                plt.close(fig)

                st.subheader(f"พบจุดตัด {len(roots)} จุด")
                if roots:
                    st.dataframe(pd.DataFrame(
                        [(i + 1, r, float(f2(r)), len(s) - 1) for i, (r, s) in enumerate(roots)],
                        columns=["#", "x", "y", "จำนวนรอบ Newton"]), hide_index=True)
                    st.markdown("**ขั้นตอน Newton-Raphson ของจุดตัดที่ 1**")
                    st.dataframe(pd.DataFrame(roots[0][1], columns=["รอบ n", "x_n", "g(x_n)"]), hide_index=True)