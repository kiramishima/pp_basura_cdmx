import streamlit as st
import pandas as pd
import polars as pl
import geopandas as gpd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from datetime import datetime as dt
import folium
import altair as alt
from streamlit_elements import elements, mui, html, dashboard
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Dashboard - Monitor Basura CDMX",
    page_icon="📊",
    layout="wide",
)

# Layouts
layout = [
    # Editor item is positioned in coordinates x=0 and y=0, and takes 6/12 columns and has a height of 3.
    dashboard.Item("editor", 0, 0, 6, 3),
    # Chart item is positioned in coordinates x=6 and y=0, and takes 6/12 columns and has a height of 3.
    dashboard.Item("chart", 6, 0, 6, 3),
    # Media item is positioned in coordinates x=0 and y=3, and takes 6/12 columns and has a height of 4.
    dashboard.Item("media", 0, 2, 12, 4),
]

with elements('home'):
    with dashboard.Grid(layout):
        st.title('Monitor Basura CDMX')

        st.divider()

        # Creamos las columnas
        col1, col2, col3 = st.columns(3)
        with mui.Card(key="editor", sx={"display": "flex", "flexDirection": "column"}):

            mui.CardHeader(title="Particulas", className="draggable")
            with mui.CardContent(sx={"flex": 1, "minHeight": 0}):
                with mui.Typography:
                    html.h2("PST", component='div')
                    html.div("2ed")
                    html.p("PM10")
                    html.p("2ed")
                    html.p("PM2.5")
                    html.p("2ed")