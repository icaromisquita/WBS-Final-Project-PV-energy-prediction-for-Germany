import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="PV_APP",
    page_icon="🌞",
)

st.write("# Welcome to Solar Predict APP!🔆")

st.sidebar.success("🌞 Select the desired function 🌞")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects. \n
    **👈 Select the desired function from the sidebar** to see what this APP can do!
    
    ### API dataset sources used in this APP.\n
    🔹 PVLIB [documentation](https://pvlib-python.readthedocs.io/en/stable/index.html)\n
    🔹 PVGIS [documentation](https://joint-research-centre.ec.europa.eu/pvgis-photovoltaic-geographical-information-system_en)\n
    🔹 Link with the the project code [GitHub](https://github.com/icaromisquita/WBS-Final-Project-)\n
    """
)

Pv_mount_image = Image.open('./img/solar_mount-removebg-preview.png')
st.image(Pv_mount_image, caption='**Elevated mounting systems for pitched roofs.')