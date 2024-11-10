import streamlit as st
import pandas as pd
import numpy as np
import time



dashboard = st.Page(
    "reports/dashboard.py", title="Dashboard", icon=":material/dashboard:"
)
bugs = st.Page("reports/bugs.py", title="Bug reports", icon=":material/bug_report:")
alerts = st.Page(
    "reports/alerts.py", title="System alerts", icon=":material/notification_important:"
)

search = st.Page("tools/search.py", title="Search", icon=":material/search:")
history = st.Page("tools/history.py", title="History", icon=":material/history:")
project=st.Page("projects/main.py", title="Project", icon=":material/history:")
intro_page=st.Page("home/home.py", title="Intro", icon=":material/home:", default=True)

pg = st.navigation(
    {
        "Home": [intro_page], 
        "Project":[project],
        "Reports": [dashboard, bugs, alerts],
        "Tools": [search, history],
        
    }
)

pg.run()
