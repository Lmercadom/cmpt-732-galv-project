import plotly.graph_objects as go

def show_recomm(df):
    """

    :param df:
    :return:
    """
    with open('.mapbox_token', 'r') as f:
        mapbox_access_token = f.read()

    # TODO 1. plot additional atributes
    # TODO 2. show selected location on plot as a different marker
    fig = go.Figure(go.Scattermapbox(
        lat=df.business_latitude.tolist(),
        lon=df.business_longitude.tolist(),
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=9
        ),
        text=df.business_name.tolist(),
    ))
    fig.update_layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=df.business_latitude.tolist()[0],
                lon=df.business_longitude.tolist()[0]
            ),
            pitch=0,
            zoom=10
        ),
    )
    fig.show()