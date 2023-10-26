import os
import time

import colorlover as cl
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_deck
from dash.dependencies import Input, Output, State
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.data_classes import Box, LidarPointCloud, RadarPointCloud
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk




def Header(name, app):
    title = html.H2(name, style={"margin-top": 5})
    

    return dbc.Row([dbc.Col(title, md=8)])


def unsnake(st):
    
    return st.replace("_", " ").title()


def build_deck(mode, pc_df, polygon_data):
    if mode == "first_person":
        view = pdk.View(type="FirstPersonView", controller=True)
        view_state = pdk.ViewState(latitude=0, longitude=0, bearing=-90, pitch=15)
        point_size = 10
    elif mode == "orbit":
        view = pdk.View(type="OrbitView", controller=True)
        view_state = pdk.ViewState(
            target=[0, 0, 1e-5],
            controller=True,
            zoom=23,
            rotation_orbit=-90,
            rotation_x=15,
        )
        point_size = 3

    else:
        view_state = pdk.ViewState(
            latitude=0,
            longitude=0,
            bearing=45,
            pitch=50,
            zoom=20,
            max_zoom=30,
            position=[0, 0, 1e-5],
        )
        view = pdk.View(type="MapView", controller=True)
        point_size = 1

    pc_layer = pdk.Layer(
        "PointCloudLayer",
        data=pc_df,
        get_position=["x", "y", "z"],
        get_color=[255, 255, 255],
        auto_highlight=True,
        pickable=False,
        point_size=point_size,
        coordinate_system=2,
        coordinate_origin=[0, 0],
    )

    box_layer = pdk.Layer(
        "PolygonLayer",
        data=polygon_data,
        stroked=True,
        pickable=True,
        filled=True,
        extruded=True,
        opacity=0.2,
        wireframe=True,
        line_width_min_pixels=1,
        get_polygon="polygon",
        get_fill_color="color",
        get_line_color=[255, 255, 255],
        get_line_width=0,
        coordinate_system=2,
        get_elevation="elevation",
    )

    tooltip = {"html": "<b>Label:</b> {name}"}

    r = pdk.Deck(
        [pc_layer, box_layer],
        initial_view_state=view_state,
        views=[view],
        tooltip=tooltip,
        map_provider=None,
    )

    return r


def compute_pointcloud_for_image(
    lv5,
    sample_token: str,
    dot_size: int = 2,
    pointsensor_channel: str = "LIDAR_TOP",
    camera_channel: str = "CAM_FRONT",
    out_path: str = None,
):
    """Scatter-plots a point-cloud on top of image.
    Args:
        sample_token: Sample token.
        dot_size: Scatter plot dot size.
        pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
        camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        out_path: Optional path to save the rendered figure to disk.
    Returns:
        tuple containing the points, array of colors and a pillow image
    """
    sample_record = lv5.get("sample", sample_token)

    # Here we just grab the front camera and the point sensor.
    pointsensor_token = sample_record["data"][pointsensor_channel]
    camera_token = sample_record["data"][camera_channel]

    points, coloring, im = lv5.explorer.map_pointcloud_to_image(
        pointsensor_token, camera_token
    )

    return points, coloring, im


def render_box_in_image(lv5, im, sample: str, camera_channel: str):
    camera_token = sample["data"][camera_channel]
    data_path, boxes, camera_intrinsic = lv5.get_sample_data(
        camera_token, flat_vehicle_coordinates=False
    )

    arr = np.array(im)

    for box in boxes:
        c = NAME2COLOR[box.name]
        box.render_cv2(arr, normalize=True, view=camera_intrinsic, colors=(c, c, c, c, c))

    new = Image.fromarray(arr)
    return new


def get_token_list(scene):
    token_list = [scene["first_sample_token"]]
    sample = lv5.get("sample", token_list[0])

    while sample["next"] != "":
        token_list.append(sample["next"])
        sample = lv5.get("sample", sample["next"])

    return token_list


def build_figure(lv5, sample, lidar, camera, overlay):
    points, coloring, im = compute_pointcloud_for_image(
        lv5, sample["token"], pointsensor_channel=lidar, camera_channel=camera
    )

    if "boxes" in overlay:
        im = render_box_in_image(lv5, im, sample, camera_channel=camera)

    fig = px.imshow(im, binary_format="jpeg", binary_compression_level=2)

    if "pointcloud" in overlay:
        fig.add_trace(
            go.Scattergl(
                x=points[0,],
                y=points[1,],
                mode="markers",
                opacity=0.4,
                marker_color=coloring,
                marker_size=3,
            )
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=0, b=0),
        paper_bgcolor="#506784",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode=False,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, range=(0, im.size[0]))
    fig.update_yaxes(showticklabels=False, showgrid=False, range=(im.size[1], 0))

    return fig


# Variables
CAMERAS = [
    "CAM_FRONT",
    "CAM_BACK",
    "CAM_FRONT_ZOOMED",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
]
LIDARS = ["LIDAR_TOP"]

NAME2COLOR = {
    "trailer": (0,0,0), # Black
    "pedestrian": (43.0, 131.0, 186.0), # Blue
    "car": (215.0, 25.0, 28.0), # Red
    "truck": (99.0, 230.0, 80.0), # Green
    "bus" : (173.0, 49.0, 165.0), # Pink
    "motorcycle" : (3.0, 2.0, 2.0), # Black
    "bicycle" : (209.0, 199.0, 13.0), # Yellow
    "construction" : (209.0, 75.0, 13.0), # Orange
    "construction_worker" : (209.0, 75.0, 13.0), # Orange
    "barrier" : (9.0, 6.0, 71.0), # Navy
    "traffic_cone" : (52.0, 201.0, 196.0), # Turquoise
    "pushable_pullable" : (97.0, 95.0, 84.0), # Grey
    "other_vehicle" : (101.0, 17.0, 191.0),     # Purple
    "debris" : (77, 69, 67), # Brown
    "bicycle_rack" : (209.0, 199.0, 13.0), # Yellow
    "child" : (255,255,255) # White
}

       # ["trailer", "pedestrian", "child","car", "truck", "bus", "motorcycle", "bicycle", "construction", "barrier", "traffic_cone", "pushable_pullable", "other_vehicle"],
       # cl.to_numeric(cl.scales['13']['qual']['Paired']),
    


# Create Lyft object
lv5 = LyftDataset(data_path="./data/sets/nuscenes", json_path="./data/sets/nuscenes/v1.0-mini", verbose=True)
# Load a single scene
scene = lv5.scene[0]
token_list = get_token_list(scene)
INITIAL_TOKEN = scene["first_sample_token"]



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

controls = [
    dbc.FormGroup(
        [
            dbc.Label("Camera Position"),
            dbc.Select(
                id="camera",
                options=[
                    {"label": unsnake(s.replace("CAM_", "")), "value": s}
                    for s in CAMERAS
                ],
                value=CAMERAS[0],
            ),
        ]
    ),
   
    dbc.FormGroup(
        [
            dbc.Label("Frame"),
            html.Br(),
            dbc.Spinner(
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            "Prev", id="prev", n_clicks=0, color="primary", outline=True
                        ),
                        dbc.Button("Next", id="next",  color="primary"),
                    ],
                    id="button-group",
                    style={"width": "100%"},
                ),
                spinner_style={"margin-top": 0, "margin-bottom": 0},
            ),
        ], style={'padding-right': '65px'}
    ),
    dbc.FormGroup(
        [
            dbc.Label("Progression"),
            dbc.Spinner(
                dbc.Input(
                    id="progression", type="range", min=0, max=len(token_list), value=0
                ),
                spinner_style={"margin-top": 0, "margin-bottom": 0},
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Lidar Position"),
            dbc.Select(
                id="lidar",
                value=LIDARS[0],
                options=[
                    {"label": unsnake(s.replace("LIDAR_", "")), "value": s}
                    for s in LIDARS
                ],
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Lidar View Mode"),
            dbc.Select(
                id="view-mode",
                value="map",
                options=[
                    {"label": unsnake(x), "value": x}
                    for x in ["first_person", "orbit", "map"]
                ],
            ),
        ]
    ),
]

deck_card = dbc.Card(
    dash_deck.DeckGL(id="deck-pointcloud", tooltip={"html": "<b>Label:</b> {name}"}),
    body=True,
    style={"height": "calc(85vh - 180px)"},
)

app.layout = dbc.Container(
    [
        
        html.H1("Reconstructing Components of Autonomous Driving Perception", style={'color':'#F3F6FA'}),
        html.Br(),
        html.Img(src=r'assets/fsd.jpg', alt='image', style={'height' : '93%', 'width' : '100%'}),
        html.Br(),
        html.Br(),
        html.Img(src=r'assets/mapping.jpg', alt='image', style={'height' : '93%', 'width' : '100%'}),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
       
        
        
        html.Img(src=r'assets/Av.png', alt='image', style={'height': '45%','width': '45%', 'margin-left' : '27%', 'display': 'inline-block', 'padding-right': '20px'}),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H3('Lane Detection', style={'color': 'orange'}),
        html.Br(),
        html.Video(controls = True, id = 'movie_player', src = r"assets\project_video_output_try17.mp4", autoPlay=False, style={'margin-left': '15%'}),
        html.Br(),
        html.Br(),
        html.H3('Traffic Light Detection', style={'color': 'orange'}),
        html.Br(),
        html.Br(),
        html.P('Using a Pre-trained CNN model we were able to produce an accurate traffic light detection & recognition model.', style={'margin-left': '25%', 'font-size': '23px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.Div([
        html.Img(src=r'assets/traffic_green.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        html.Img(src=r'assets/red_light.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        html.Img(src=r'assets/light3.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        html.Img(src=r'assets/light5.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        ], style={'margin-left': '15%'}),
        html.Br(),
        html.Br(),
        
        html.H3('Motion Prediction', style={'color': 'orange'}),
        html.Br(),
        html.Br(),
        
        html.P('Motion Prediction is a very important component of autonomous vehicles safety, being able to understand surrounding vehicles motion and trajectories is essential in order to assure that the AV`s motion will not be impacted.'
               'We used neural networks to make predictions of surrounding traffic agents using the Lyft motion dataset and Lyft 5 kit. The Lyft dataset contains:', style={'font-size': '22px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- Over 1000 hours of driving data following a single route', style={'font-size': '18px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- 170,000 different scenes each 25 seconds long', style={'font-size': '20px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- HD semantic map capturing road rules, including lane geometry, and other traffic elements', style={'font-size': '20px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- HD aerial picture of the area', style={'font-size': '20px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- Each traffic agent has its velocity, yaw (rotation), rotation rate, acceleration annotated', style={'font-size': '20px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.Br(),
        html.Div([
        html.Img(src=r'assets/map.png', alt='image', style={'height' : '45%', 'width' : '45%'}),
        html.Img(src=r'assets/data_overview.png', alt='image', style={'height' : '45%', 'width' : '45%'}),
        ], style={'display':'inline-block', 'margin-left': '15%'}),
        
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),

        html.H4('ResNet Architecture:', style={'font-size': '30px'}),
        html.P('We built the following ResNet 18 model before I realised there was an error within the lyft software kit. Hence I opted to continue the project with a pretrained resnet34 model', style={'font-size': '20px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.Div([
        html.P('ResNet18 CNN Architecture:', style={'font-size': '19px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- Convolutional Layer', style={'font-size': '19px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- Batch Normalization Layer', style={'font-size': '19px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- Activation Layer (relu type)', style={'font-size': '19px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- Pooling Layer (maxpool type)', style={'font-size': '19px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- Basic Block -- Layer 1', style={'font-size': '19px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- Basic Block -- Layer 2', style={'font-size': '19px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- Basic Block -- Layer 3', style={'font-size': '19px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- Basic Block -- Layer 4', style={'font-size': '19px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('- Pooling Layer (avgpool type)', style={'font-size': '19px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        html.P('-- Linear Layer (Fully Connected Layer)', style={'font-size': '19px', 'font-family': 'Arial', 'color': '#F3F6FA'}),
        ], style={'margin': '5px', 'display' : 'inline-block'}),
        html.Img(src=r'assets/resnet.jpg', alt='image', style={'height' : '45%', 'width' : '45%', 'margin-left': '15%', 'display' : 'inline-block', 'margin-top': '-15%'}),
        html.Br(),
        html.Br(),
        html.H4('Motion Prediction Results/Visualisations:', style={'font-size': '30px'}),
        html.Br(),
        html.Br(),
        html.Div([
        html.Img(src=r'assets/vis1.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        html.Img(src=r'assets/vis2.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        html.Img(src=r'assets/vis3.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        html.Img(src=r'assets/vis4.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        html.Img(src=r'assets/vis5.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        html.Img(src=r'assets/vis6.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        html.Img(src=r'assets/vis7.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        html.Img(src=r'assets/vis8.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        html.Img(src=r'assets/vis9.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        html.Img(src=r'assets/vi10.png', alt='image', style={'display': 'inline-block', 'height': '38%', 'width': '38%'}),
        ], style={'margin-left': '15%'}),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H3('Lidar Localisation', style={'color': 'orange'}),
        html.Br(),

       








if __name__ == "__main__":
    app.run_server(debug=True)
