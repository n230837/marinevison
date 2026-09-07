

import folium
import numpy as np

# Create demo offshore locations
np.random.seed(42)

m = folium.Map(
    location=[13.08, 80.36],
    zoom_start=10
)

for i in range(100):
    lat = np.random.uniform(12.98, 13.18)
    lon = np.random.uniform(80.30, 80.42)

    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        popup=f"Marine Anomaly #{i+1}",
        fill=True
    ).add_to(m)

# Save map
map_path = "/content/marine_debris_offshore_map.html"
m.save(map_path)

print("✅ Map recreated!")
print("Map path:", map_path)
