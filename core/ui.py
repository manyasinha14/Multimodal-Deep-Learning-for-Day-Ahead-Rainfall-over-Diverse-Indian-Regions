"""Small Streamlit helpers shared by all pages."""
import streamlit as st

try:
    _VER = tuple(int(p) for p in st.__version__.split(".")[:2])
except Exception:  # pragma: no cover
    _VER = (1, 40)


def show_chart(fig, key=None):
    """Full-width plotly chart on both old (use_container_width) and new (width='stretch') Streamlit."""
    if _VER >= (1, 50):
        st.plotly_chart(fig, width="stretch", theme="streamlit", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, theme="streamlit", key=key)


def show_table(df, **kw):
    kw.setdefault("hide_index", True)
    if _VER >= (1, 50):
        st.dataframe(df, width="stretch", **kw)
    else:
        st.dataframe(df, use_container_width=True, **kw)


def rain_badge(rain: bool) -> str:
    return "🌧️ **RAIN** expected" if rain else "☀️ **NO RAIN** expected"


def pct(x):
    return "—" if x != x else f"{100 * x:.0f}%"


def fmt(x, d=2):
    return "—" if x != x else f"{x:.{d}f}"
