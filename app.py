import pickle
from typing import Tuple, Union, List, Dict, Any
import cvxopt as opt
import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from cvxopt import solvers
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State


DATA_DIR = './data/'
MODELS_DIR = './models/'
STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
RiskTolerances = List[List[Union[int, float]]]


def clean_data() -> Tuple[pd.DataFrame]:
    """
    """
    investors = pd.read_csv(DATA_DIR + 'InputData.csv', index_col=0)
    assets = pd.read_csv(DATA_DIR + 'SP500Data.csv', index_col=0)
    drop_list = sorted(
        assets
        .isnull()
        .mean(axis=0)
        .sort_values(ascending=False)
        .loc[lambda x: x > 0.3]
        .index.tolist()
    )
    cleaned_assets = (
        assets
        .drop(labels=drop_list, axis=1)
        .fillna(method='ffill')
    )

    return cleaned_assets, investors


app = dash.Dash(__name__, external_stylesheets=STYLESHEETS)
assets, investors = clean_data()  # Global state


def predict_risk_tolerance(X_input: RiskTolerances) -> np.ndarray:
    """
    """
    with open(MODELS_DIR + 'model.sav', 'rb') as fh:
        model = pickle.load(fh)

    return model.predict(X_input)  # Estimate accuracy on validation set


def get_asset_allocation(
    riskTolerance: float,
    stock_ticker: List[str]
) -> Tuple[pd.DataFrame]:
    """
    """
    # Asset allocation given the Return, variance
    assets_selected = assets.loc[:, stock_ticker]
    return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T
    n = len(return_vec)
    returns = np.asmatrix(return_vec)
    mus = 1 - riskTolerance

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(return_vec))
    pbar = opt.matrix(np.mean(return_vec, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus * S, -pbar, G, h, A, b)
    w = portfolios['x'].T
    Alloc = pd.DataFrame(
        data=np.array(portfolios['x']),
        index=assets_selected.columns
    )

    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus * S, -pbar, G, h, A, b)
    returns_final = (np.array(assets_selected) * np.array(w))
    returns_sum = np.sum(returns_final, axis=1)
    returns_sum_pd = pd.DataFrame(returns_sum, index=assets.index)
    returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0, :] + 100
    return Alloc, returns_sum_pd


# Callback for the graph
# This function takes all the inputs and computes the cluster and the risk tolerance
@app.callback(
    [Output('risk-tolerance-text', 'value')],
    [
        Input('investor_char_button', 'n_clicks'),
        Input('Age', 'value'), Input('Nwcat', 'value'),
        Input('Inccl', 'value'), Input('Risk', 'value'),
        Input('Edu', 'value'), Input('Married', 'value'),
        Input('Kids', 'value'), Input('Occ', 'value')
    ]
)
def update_risk_tolerance(
    n_clicks: int, Age: int, Nwcat: int, Inccl: int,
    Risk: float, Edu: float, Married: float, Kids: float, Occ: float
) -> List[float]:
    """
    Get the x and y axis details
    """
    RiskTolerance = 0
    if n_clicks is not None:
        X_input = [[Age, Edu, Married, Kids, Occ, Inccl, Risk, Nwcat]]
        RiskTolerance = predict_risk_tolerance(X_input)

    # Using linear regression to get the risk tolerance within the cluster.
    return list([round(float(RiskTolerance * 100), 2)])


@app.callback(
    [Output('Asset-Allocation', 'figure'), Output('Performance', 'figure')],
    [
        Input('submit-asset_alloc_button', 'n_clicks'),
        Input('risk-tolerance-text', 'value')
    ],
    [State('ticker_symbol', 'value')]
)
def update_asset_allocationChart(
    n_clicks: int, 
    risk_tolerance: float, 
    stock_ticker: List[str]
) -> List[Dict[str, Any]]:
    """
    """
    Allocated, InvestmentReturn = get_asset_allocation(
        risk_tolerance,
        stock_ticker
    )

    return [
        {
            'data': [go.Bar(
                x=Allocated.index,
                y=Allocated.iloc[:, 0],
                marker=dict(color='red'))
            ],
            'layout': {'title': " Asset allocation - Mean-Variance Allocation"}
        },
        {
            'data': [go.Scatter(
                x=InvestmentReturn.index,
                y=InvestmentReturn.iloc[:, 0],
                name='OEE (%)',
                marker=dict(color='red'),
            )],
            'layout': {'title': "Portfolio value of $100 investment"},
        }
    ]


def serve_layout() -> html:
    """
    """
    options = [
        {
            'label': tic,  # Apple Co. AAPL
            'value': tic,
        }
        for tic in assets.columns
    ]

    main_html = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            # Dashboard Name
                            html.H3(children="Robo Advisor Dashboard"),
                            html.Div(
                                [html.H5(
                                    children="Step 1 : Enter Investor Characteristics "
                                )],
                                style={
                                    "display": "inline-block",
                                    "vertical-align": "top",
                                    "width": "30%",
                                    "color": "black",
                                    "background-color": "LightGray",
                                },
                            ),
                            html.Div(
                                [html.H5(
                                    children="Step 2 : Asset Allocation and portfolio performance"
                                )],
                                style={
                                    "display": "inline-block",
                                    "vertical-align": "top",
                                    "color": "white",
                                    "horizontalAlign": "left",
                                    "width": "70%",
                                    "background-color": "black",
                                },
                            ),
                        ],
                        style={"font-family": "calibri"},
                    ),
                    # All the Investor Characteristics
                    # ********************Demographics Features DropDown********
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Age:", style={"padding": 5}),
                                    dcc.Slider(
                                        id="Age",
                                        min=investors["AGE07"].min(),
                                        max=70,
                                        marks={25: "25", 35: "35",
                                               45: "45", 55: "55", 70: "70", },
                                        value=25,
                                    ),
                                    # html.Br(),
                                    html.Label("NetWorth:", style={
                                               "padding": 5}),
                                    dcc.Slider(
                                        id="Nwcat",
                                        # min = investors['NETWORTH07'].min(),
                                        min=-1000000,
                                        max=3000000,
                                        marks={
                                            -1000000: "-$1M",
                                            0: "0",
                                            500000: "$500K",
                                            1000000: "$1M",
                                            2000000: "$2M",
                                        },
                                        value=10000,
                                    ),
                                    # html.Br(),
                                    html.Label("Income:", style={
                                               "padding": 5}),
                                    dcc.Slider(
                                        id="Inccl",
                                        # min = investors['INCOME07'].min(), max = investors['INCOME07'].max(),
                                        min=-1000000,
                                        max=3000000,
                                        marks={
                                            -1000000: "-$1M",
                                            0: "0",
                                            500000: "$500K",
                                            1000000: "$1M",
                                            2000000: "$2M",
                                        },
                                        value=100000,
                                    ),
                                    # html.Br(),
                                    html.Label("Education Level (scale of 4):", style={
                                               "padding": 5}, ),
                                    dcc.Slider(
                                        id="Edu",
                                        min=investors["EDCL07"].min(),
                                        max=investors["EDCL07"].max(),
                                        marks={1: "1", 2: "2", 3: "3", 4: "4"},
                                        value=2,
                                    ),
                                    # html.Br(),
                                    html.Label("Married:", style={
                                               "padding": 5}),
                                    dcc.Slider(
                                        id="Married",
                                        min=investors["MARRIED07"].min(),
                                        max=investors["MARRIED07"].max(),
                                        marks={1: "1", 2: "2"},
                                        value=1,
                                    ),
                                    # html.Br(),
                                    html.Label("Kids:", style={"padding": 5}),
                                    dcc.Slider(
                                        id="Kids",
                                        min=investors["KIDS07"].min(),
                                        max=investors["KIDS07"].max(),
                                        # marks={ 1: '1', 2: '2', 3: '3', 4: '4'},
                                        marks=[
                                            {"label": j, "value": j}
                                            for j in investors["KIDS07"].unique()
                                        ],
                                        value=3,
                                    ),
                                    # html.Br(),
                                    html.Label("Occupation:", style={
                                               "padding": 5}),
                                    dcc.Slider(
                                        id="Occ",
                                        min=investors["OCCAT107"].min(),
                                        max=investors["OCCAT107"].max(),
                                        marks={1: "1", 2: "2", 3: "3", 4: "4"},
                                        value=3,
                                    ),
                                    # html.Br(),
                                    html.Label(
                                        "Willingness to take Risk:", style={"padding": 5}
                                    ),
                                    dcc.Slider(
                                        id="Risk",
                                        min=investors["RISK07"].min(),
                                        max=investors["RISK07"].max(),
                                        marks={1: "1", 2: "2", 3: "3", 4: "4"},
                                        value=3,
                                    ),
                                    # html.Br(),
                                    html.Button(
                                        id="investor_char_button",
                                        n_clicks=0,
                                        children="Calculate Risk Tolerance",
                                        style={
                                            "fontSize": 14,
                                            "marginLeft": "30px",
                                            "color": "white",
                                            "horizontal-align": "left",
                                            "backgroundColor": "grey",
                                        },
                                    ),
                                    # html.Br(),
                                ],
                                style={"width": "80%"},
                            ),
                        ],
                        style={
                            "width": "30%",
                            "font-family": "calibri",
                            "vertical-align": "top",
                            "display": "inline-block",
                        },
                    ),
                    #                     , "border":".5px black solid"}),
                    # ********************Risk Tolerance Charts********
                    html.Div(
                        [
                            # html.H5(children='Step 2 : Enter the Instruments for the allocation portfolio'),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "Risk Tolerance (scale of 100) :",
                                                style={"padding": 5},
                                            ),
                                            dcc.Input(
                                                id="risk-tolerance-text"),
                                        ],
                                        style={
                                            "width": "100%",
                                            "font-family": "calibri",
                                            "vertical-align": "top",
                                            "display": "inline-block",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "Select the assets for the portfolio:",
                                                style={"padding": 5},
                                            ),
                                            dcc.Dropdown(
                                                id="ticker_symbol",
                                                options=options,
                                                value=["GOOGL", "FB", "GS",
                                                       "MS", "GE", "MSFT", ],
                                                multi=True
                                                # style={'fontSize': 24, 'width': 75}
                                            ),
                                            html.Button(
                                                id="submit-asset_alloc_button",
                                                n_clicks=0,
                                                children="Submit",
                                                style={
                                                    "fontSize": 12,
                                                    "marginLeft": "25px",
                                                    "color": "white",
                                                    "backgroundColor": "grey",
                                                },
                                            ),
                                        ],
                                        style={
                                            "width": "100%",
                                            "font-family": "calibri",
                                            "vertical-align": "top",
                                            "display": "inline-block",
                                        },
                                    ),
                                ],
                                style={
                                    "width": "100%",
                                    "display": "inline-block",
                                    "font-family": "calibri",
                                    "vertical-align": "top",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [dcc.Graph(id="Asset-Allocation"), ],
                                        style={
                                            "width": "50%",
                                            "vertical-align": "top",
                                            "display": "inline-block",
                                            "font-family": "calibri",
                                            "horizontal-align": "right",
                                        },
                                    ),
                                    html.Div(
                                        [dcc.Graph(id="Performance")],
                                        style={
                                            "width": "50%",
                                            "vertical-align": "top",
                                            "display": "inline-block",
                                            "font-family": "calibri",
                                            "horizontal-align": "right",
                                        },
                                    ),
                                ],
                                style={
                                    "width": "100%",
                                    "vertical-align": "top",
                                    "display": "inline-block",
                                    "font-family": "calibri",
                                    "horizontal-align": "right",
                                },
                            ),
                        ],
                        style={
                            "width": "70%",
                            "display": "inline-block",
                            "font-family": "calibri",
                            "vertical-align": "top",
                            "horizontal-align": "right",
                        },
                    ),
                ],
                style={
                    "width": "70%",
                    "display": "inline-block",
                    "font-family": "calibri",
                    "vertical-align": "top",
                },
            ),
        ]
    )
    return main_html


if __name__ == '__main__':
    app.layout = serve_layout
    app.run_server()
