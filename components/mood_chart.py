import streamlit as st, pandas as pd

def draw_chart(log):
    if not log:
        st.info("Chat to start mood tracking.")
        return
    df = pd.DataFrame(log, columns=["ts","val","ar"]).set_index("ts")
    st.line_chart(df)
