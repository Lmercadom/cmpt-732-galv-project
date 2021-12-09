import logging
import plotly.graph_objects as go

def show_recomm(df):
    """

    :param df:
    :return:
    """
    with open('.mapbox_token', 'r') as f:
        mapbox_access_token = f.read()

    # filter number of suggestions to 100.
    if len(df) > 100:
        logging.debug(f"Filtering out top 100 records by distance")
        df = df.sort_values(by='distance', ascending=True)
        df = df.iloc[:100,:]

    # TODO 1. plot additional atributes
    # TODO 2. show selected location on plot as a different marker
    fig = go.Figure(go.Scattermapbox(
        lat=df.latitude.tolist(),
        lon=df.longitude.tolist(),
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=9
        ),
        text=df.name.tolist(),
    ))
    fig.update_layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=df.latitude.tolist()[0],
                lon=df.longitude.tolist()[0]
            ),
            pitch=0,
            zoom=10
        ),
    )
    fig.show()