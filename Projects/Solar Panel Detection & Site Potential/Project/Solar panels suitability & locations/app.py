
import streamlit as st
import ee
import folium
import pandas as pd
import html
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# 1. Configuration
st.set_page_config(page_title="Bahrain Solar Analysis", layout="wide")
st.title("Bahrain Solar Analysis")
st.caption("Solar site suitability and existing solar installations")

# 2. Earth Engine
PROJECT_ID = "primeval-stack-514517-u2"
ASSET_ID = f"projects/{PROJECT_ID}/assets/Bahrain_Solar_Results"

@st.cache_resource
def initialize_ee():
    ee.Initialize(project=PROJECT_ID)
    return ee.Image(ASSET_ID)

try:
    solar = initialize_ee()
except Exception as e:
    st.error("Google Earth Engine initialization failed.")
    st.code(str(e))
    st.stop()

# 3. Existing solar projects
try:
    df = pd.read_csv("bahrain_solar_existing_sites_map.csv")
except FileNotFoundError:
    st.error("Could not find bahrain_solar_existing_sites_map.csv.")
    st.stop()

for col in ["latitude", "longitude", "panel_count"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["latitude", "longitude"]).copy()
df["site_id"] = df["site_id"].astype(str)

top3 = df.dropna(subset=["panel_count"]).nlargest(3, "panel_count")
top3_ids = set(top3["site_id"])

# 4. Analysis overview
st.header("1. Solar Site Suitability")
st.subheader("Analysis Overview")

for col, label, value in zip(
    st.columns(3),
    ["Satellite Images", "Spatial Resolution", "Spectral Indices"],
    ["139", "20 m", "4"]
):
    col.metric(label, value)

# 5. Scenario A
st.subheader("Suitability Scenario")
st.markdown("### Scenario A")

c1, c2 = st.columns(2)
c1.metric("Mean Score", "0.554")
c2.metric("Good Suitability Area", "610.98 km²")

# 6. Existing projects
st.divider()
st.header("2. Existing Solar Projects")

for col, label, value in zip(
    st.columns(3),
    ["Mapped Projects", "Projects with Panel Data", "Top Projects Highlighted"],
    [len(df), int(df["panel_count"].notna().sum()), len(top3)]
):
    col.metric(label, value)

st.subheader("Top 3 Projects by Number of Panels")

if not top3.empty:
    st.bar_chart(top3.set_index("project_name")["panel_count"])
    st.dataframe(
        top3[["project_name", "panel_count", "capacity_value", "capacity_unit"]],
        hide_index=True,
        use_container_width=True
    )

# 7. Map layers
layers = {
    "Satellite RGB": (["B4", "B3", "B2"], {"min": 0, "max": 3000}),
    "Scenario A": (["Scenario_A"], {
        "min": 0, "max": 1, "palette": ["red", "yellow", "green"]
    }),
    "High Suitability Areas": (["High_Suitability"], {
        "min": 0.75, "max": 1, "palette": ["00FF00"]
    }),
    "NDVI": (["NDVI"], {
        "min": -0.2, "max": 0.8, "palette": ["brown", "yellow", "green"]
    }),
    "NDBI": (["NDBI"], {
        "min": -0.3, "max": 0.3, "palette": ["blue", "white", "red"]
    }),
    "NDWI": (["NDWI"], {
        "min": -0.5, "max": 0.5, "palette": ["brown", "white", "blue"]
    }),
    "BSI": (["BSI"], {
        "min": -0.3, "max": 0.3, "palette": ["blue", "white", "brown"]
    })
}

# 8. Interactive map
st.divider()
st.subheader("Interactive Bahrain Solar Map")
show_heatmap = st.checkbox("Show Solar Projects Heatmap", value=False)

m = folium.Map(
    location=[26.12, 50.55],
    zoom_start=10,
    tiles="OpenStreetMap"
)

# Earth Engine layers
for name, (bands, vis) in layers.items():
    image = solar.select(bands)

    if name == "High Suitability Areas":
        image = image.updateMask(image.gte(0.75))

    url = image.getMapId(vis)["tile_fetcher"].url_format

    folium.TileLayer(
        tiles=url,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        show=(name == "Scenario A"),
        opacity=0.75
    ).add_to(m)

# 9. Project markers
projects = folium.FeatureGroup(name="Existing Solar Projects", show=True)

for _, site in df.iterrows():
    name = html.escape(str(site.get("project_name", "Solar Project")))
    area = html.escape(str(site.get("location_area", "Not specified")))
    note = html.escape(str(site.get("map_note", "No additional information")))

    panels = site["panel_count"]
    panels_text = f"{panels:,.0f}" if pd.notna(panels) else "Not available"

    popup = (
        f"<b>{name}</b><br>"
        f"Location: {area}<br>"
        f"Number of Panels: {panels_text}<br>"
        f"Note: {note}"
    )

    folium.CircleMarker(
        location=[site["latitude"], site["longitude"]],
        radius=11 if site["site_id"] in top3_ids else 7,
        color="green",
        fill=True,
        fill_color="green",
        fill_opacity=0.9,
        weight=2,
        popup=folium.Popup(popup, max_width=300),
        tooltip=name
    ).add_to(projects)

projects.add_to(m)

# 10. Optional heatmap
if show_heatmap:
    heat = folium.FeatureGroup(name="Solar Projects Heatmap", show=True)
    HeatMap(
        df[["latitude", "longitude"]].values.tolist(),
        radius=25,
        blur=18
    ).add_to(heat)
    heat.add_to(m)

# 11. Map controls and display
folium.LayerControl(collapsed=False).add_to(m)

st.markdown(
    "🟢 Existing solar projects | "
    "Larger circles represent the top 3 projects by panel count."
)

st_folium(m, width=None, height=700, use_container_width=True)