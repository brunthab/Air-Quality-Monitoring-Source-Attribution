"""AeroTrace dashboard.   Run:  streamlit run app.py"""
import pandas as pd
import plotly.express as px
import streamlit as st

from aerotrace.aqi import category, pollution_score
from aerotrace.model import forecast
from aerotrace.pipeline import attribute_cell, build_bundle

st.set_page_config(page_title="AeroTrace", layout="wide")


@st.cache_resource(show_spinner="Generating data and training model...")
def get_bundle():
    return build_bundle()


b = get_bundle()
df = b.df
times = sorted(df["time"].unique())

st.title("AeroTrace")
st.caption("Don't just measure pollution. Discover its source.  "
           "Demo data is SIMULATED; swap `simulate.py` for real IoT / satellite / weather loaders.")

with st.sidebar:
    t_idx = st.slider("Time", 0, len(times) - 1, len(times) - 1)
    t = pd.Timestamp(times[t_idx])
    st.write(f"**{t:%d %b %Y %H:%M}**")
    alert_threshold = st.number_input("Alert threshold (score)", 100, 400, 200, step=50)

snap = df[df["time"] == t].copy()
snap["score"] = snap["pm25_best"].apply(lambda v: pollution_score({"pm25": v})[0])
snap["category"] = snap["score"].apply(category)

tab_over, tab_map, tab_attr, tab_pred, tab_wx, tab_alert, tab_hist = st.tabs(
    ["AQI overview", "Pollution map", "Source attribution", "Prediction",
     "Weather", "Alerts", "History"])

with tab_over:
    sens = snap[snap["is_sensor"]]
    score, cat, driver = pollution_score({"pm25": sens["pm25"].mean(), "pm10": sens["pm10"].mean(),
                                          "co": sens["co"].mean()})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated pollution score", f"{score:.0f}", cat)
    c2.metric("Mean PM2.5 (sensors)", f"{sens['pm25'].mean():.0f} µg/m³")
    c3.metric("Mean PM10 (sensors)", f"{sens['pm10'].mean():.0f} µg/m³")
    c4.metric("Mean CO (sensors)", f"{sens['co'].mean():.2f} mg/m³")
    st.caption("Score uses PM2.5, PM10 and CO only, so it is an estimated pollution score, "
               "not an official CPCB AQI.")
    st.subheader("Model validation (PM2.5, µg/m³)")
    st.dataframe(pd.DataFrame(b.metrics).T.round(3))
    st.write("Spatial = leave-one-sensor-out (unmonitored location). Temporal = last 20% of time held out.")

with tab_map:
    fig = px.scatter(snap, x="lon", y="lat", color="pm25_best", size_max=14,
                     color_continuous_scale="YlOrRd", symbol="is_sensor",
                     hover_data={"cell_id": True, "pm25_best": ":.0f", "lon": False, "lat": False},
                     labels={"pm25_best": "PM2.5 µg/m³"})
    fig.update_traces(marker=dict(size=18))
    fig.update_yaxes(scaleanchor="x")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Diamonds/crosses = IoT sensor cells (measured). Others = model estimate from satellite + weather.")

with tab_attr:
    top = snap.sort_values("pm25_best", ascending=False)
    options = top["cell_id"].head(15).tolist()
    cell = st.selectbox("Location (cell id, worst first)", options)
    row, ranked, note, mode = attribute_cell(b, t, int(cell))
    st.write(f"Evidence: **{mode}** · wind {row['wind_dir']:.0f}° at {row['wind_speed']:.1f} m/s · "
             f"PM2.5 {row['pm25_best']:.0f} µg/m³")
    if not ranked:
        st.info(note)
    for a in ranked:
        st.markdown(f"**{a.share * 100:.0f}% — {a.source}** · confidence: {a.confidence}")
        st.progress(float(a.share))
        for r in a.reasons:
            st.write(f"• {r}")
    if ranked:
        st.caption(note)

with tab_pred:
    fc = forecast(df)
    hist = df[df["is_sensor"]].groupby("time")["pm25"].mean().tail(48).reset_index()
    fig = px.line(hist, x="time", y="pm25", labels={"pm25": "PM2.5 µg/m³"})
    fig.add_scatter(x=fc["time"], y=fc["pm25_forecast"], mode="lines+markers", name="forecast")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Why the model estimates what it does (feature importance)")
    imp = b.estimator.importances().head(10).reset_index()
    imp.columns = ["feature", "importance"]
    st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h"), use_container_width=True)

with tab_wx:
    w = df[df["cell_id"] == 0].tail(72)
    st.plotly_chart(px.line(w, x="time", y="wind_speed", title="Wind speed (m/s)"), use_container_width=True)
    st.plotly_chart(px.scatter(w, x="time", y="wind_dir", title="Wind direction (° from)"), use_container_width=True)
    st.plotly_chart(px.line(w, x="time", y=["temp", "rh"], title="Temperature (°C) and humidity (%)"),
                    use_container_width=True)

with tab_alert:
    hot = snap[snap["score"] >= alert_threshold].sort_values("score", ascending=False)
    if hot.empty:
        st.success(f"No cell above score {alert_threshold} at this time.")
    else:
        st.warning(f"{len(hot)} cell(s) at or above score {alert_threshold}")
        st.dataframe(hot[["cell_id", "pm25_best", "score", "category", "is_sensor"]].round(1))

with tab_hist:
    h = df[df["is_sensor"]].copy()
    h["day"] = h["time"].dt.date
    daily = h.groupby("day")[["pm25", "pm10", "no2", "so2", "co"]].mean().reset_index()
    st.plotly_chart(px.line(daily, x="day", y=["pm25", "pm10"], title="Daily mean (sensors)"),
                    use_container_width=True)
    hourly = h.groupby(h["time"].dt.hour)["pm25"].mean().reset_index()
    st.plotly_chart(px.bar(hourly, x="time", y="pm25", title="Average PM2.5 by hour of day",
                           labels={"time": "hour"}), use_container_width=True)
