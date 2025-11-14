from flask import Flask, request, send_file
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Point, Polygon
import io

# -----------------------------
# Model Architecture
# -----------------------------
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


app = Flask(__name__)

# Load model
model = GATModel(4, 32, 1, heads=4, dropout=0.3)
state_dict = torch.load("gat_k20_drop0.3_unc4_best.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

edge_index = torch.zeros((2, 0), dtype=torch.long)


@app.route("/predict-image", methods=["POST"])
def predict_image():
    data = request.json
    rows = data["rows"]

    features = []
    lon_list = []
    lat_list = []

    for r in rows:
        features.append([r["Band_3"], r["Band_4"], r["Band_5"], r["Band_7"]])
        lon_list.append(r["longitude"])
        lat_list.append(r["latitude"])

    features = np.array(features)

    # ========= Fallback StandardScaler (same as notebook) =========
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        y_pred = model(X_tensor, edge_index).squeeze().numpy()

    lon = np.array(lon_list)
    lat = np.array(lat_list)
    sal = y_pred

    # Interpolate to raster
    grid_x, grid_y = np.mgrid[lon.min():lon.max():200j, lat.min():lat.max():200j]
    grid_sal = griddata((lon, lat), sal, (grid_x, grid_y), method="cubic")

    # Mask outside convex hull
    points = np.vstack((lon, lat)).T
    hull = Polygon(points)
    mask = np.array([[hull.contains(Point(x, y)) for y in grid_y[0]] for x in grid_x[:,0]])
    grid_sal[~mask] = np.nan

    # Plot to PNG
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_sal.T, extent=(lon.min(), lon.max(), lat.min(), lat.max()), origin='lower', cmap='viridis')
    plt.colorbar(label='Salinity')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Predicted Salinity Raster")

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plt.close()

    return send_file(img, mimetype="image/png")
