import argparse
from pywebio import start_server
from pywebio.input import *
from pywebio.output import *
from timeline import draw_timeline
import io
import matplotlib.pyplot as plt
import matplotlib
import pywebio
import networkx as nx
import en_core_web_sm
from pywebio import STATIC_PATH
from summa import keywords, summarizer
from pywebio.platform.flask import webio_view
from flask import Flask
app = Flask(__name__)
def extract_entities(text):
    nlp = en_core_web_sm.load()

    t = summarizer.summarize(text, ratio=0.4)
    doc = nlp(t)

    people = []
    time_expressions = []
    location = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.append(ent.text)
        elif ent.label_ == "DATE":
            time_expressions.append(ent.text)
        elif ent.label_ == "LOC":
            location.append(ent.text)

    if len(list(set(people))) > 15:
        people = []
        tr = summarizer.summarize(text, ratio=0.3)
        dr = nlp(tr)
        for ent in dr.ents:
            if ent.label_ == "PERSON":
                people.append(ent.text)

    if len(list(set(time_expressions))) > 10:
        time_expressions = []
        tr = summarizer.summarize(text, ratio=0.3)
        dr = nlp(tr)
        for ent in dr.ents:
            if ent.label_ == "DATE":
                time_expressions.append(ent.text)

    if len(list(set(location))) == 0:
        location = []
        dk = nlp(text)
        for ent in dk.ents:
            if ent.label_ == "LOC":
                location.append(ent.text)

    return list(set(people)), list(set(time_expressions)), list(set(location))


def show_timeline(eventst, time):
    matplotlib.use('agg')
    # Create a plot
    fig, ax = plt.subplots(figsize=(6, 10), constrained_layout=True)

    # Draw the vertical line
    _ = ax.axvline(x=0, color='gray', linestyle='--', linewidth=2)
    
    # Plot nodes representing events as text boxes along the line
    for i, event in enumerate(eventst):
        y = 1 - i / (len(eventst) - 1)  # Calculate y-coordinate for even spacing
        _ = ax.scatter(0, y, s=120, c='palevioletred', zorder=2)
        _ = ax.text(0.1, y, event, ha='left', fontfamily='serif', va='center', rotation=0, fontsize=12, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'))
        _ = ax.text(-0.1, y, time[i], ha='right', fontfamily='serif', rotation=0, fontsize=15, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'))
    
    # Customize the plot
    for spine in ["left", "top", "right", "bottom"]:
        _ = ax.spines[spine].set_visible(False)
    plt.title("Timeline Plot")
    plt.xticks([])
    plt.yticks([])
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.5, 0.5)
    
    # Display the plot
    buf = io.BytesIO()
    fig.savefig(buf)
    popup( "Timeline:",[
        pywebio.output.put_image(buf.getvalue()),
        put_button('Close', onclick=close_popup) ,
        ])


def show_table(location,eventsl):
    popup("Location information table:",[
                        put_table([location, eventsl]),
                        put_button('Close', onclick=close_popup) ,
                        ])
    
def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items

def visualize_graph(edge_list):
    matplotlib.use('agg')
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    pos=nx.spring_layout(G,seed=5,k=1)
    
    plt.clf()  # Clear the figure before creating a new visualization

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, ax=ax,node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax)

    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)
    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')

    edge_weights = nx.get_edge_attributes(G,'w')
    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
    my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad, font_color="red")
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)
    plt.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    
    # Display the image within a popup
    popup("Character Network:", [
        put_image(buf.getvalue()),
        put_button('Close', onclick=close_popup),
    ])

def vis():
    info = input_group("Fiction contents:", [
        textarea('Input texts:', name='text', required=True),
        actions("Actions", [
            {'label': 'Extract key words', 'value': 'save'},
            {'label': 'Reset', 'type': 'reset', 'color': 'warning'},
        ], name='action'),
    ])
    text = info["text"]
    people, time_expressions, location = extract_entities(text)

    extracted = input_group("Useful extracted:", [
        checkbox("Characters:", name="characters", options=people),
        input("number of missing characters:", name="nc", type=NUMBER, required=True),
        checkbox("Times:", name="time", options=time_expressions),
        input("number of missing time expressions:", name="nt", type=NUMBER, required=True),
        checkbox("Locations:", name="location", options=location),
        input("number of missing locations:", name="nl", type=NUMBER, required=True),
        actions("Actions:", [
            {'label': 'Add supplementary information', 'value': 'save'},
            {'label': 'Reset', 'type': 'reset', 'color': 'warning'},
        ], name='action'),
    ])

    people = extracted["characters"]
    time_expressions = extracted["time"]
    location = extracted["location"]

#addtional terms
    if extracted["nc"] >0:
        for i in list(range(extracted["nc"])):
            a=input("Additional Characters:")
            people.append(a)

    if extracted["nt"] >0:
        for i in list(range(extracted["nt"])):
            b=input("Additional Time:")
            time_expressions.append(b)

    if extracted["nl"] >0:
        for i in list(range(extracted["nl"])):
            c=input("Additional Locations:")
            location.append(c)



    #time related events:
    eventst=[]
    #rank the time
    time=[]
    for i in list(range(len(time_expressions))):
        s= select("sepecify sequence of the time expressions:", time_expressions, help_text=i+1, required=True)
        time.append(s)


    time.append("End")
    for i in list(range(len(time))):
        eventst.append(i)
        eventst[i]=input("Time related events:", help_text=time[i])

# add location related events
    eventsl=[]
    for i in list(range(len(location))):
        eventsl.append(i)
        eventsl[i]=input("Location related information:", help_text=location[i])

    #character network

    help_text = f"Note: if there are more than 1 relationships between same couple of characters, compress them in 2 dimensions. For better visualization, the two relations should be in reverse sequence. i.e. A,B:a; B,A:b; Characters are: {', '.join(people)}"
    edges_input=textarea("Type edges in the form of A,B:label :",help_text=help_text).split(";")
    edge_list=[]
    for edge_input in edges_input:
                source, rest = edge_input.split(',')
                target, label = rest.split(':')
                edge_list.append((source.strip(), target.strip(), {'w': label.strip()}))



    put_button("Show network", onclick=lambda:visualize_graph(edge_list))
    put_button("Show Location information table", onclick=lambda:show_table(location,eventsl))
    put_button("Show timeline", onclick=lambda:show_timeline(eventst,time))

app.add_url_rule('/tool', 'webio_view', webio_view(vis),
         methods=['GET', 'POST', 'OPTIONS'])  # need GET,POST and OPTIONS methods




if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("-p", "--port", type=int, default=8080)
   args = parser.parse_args()
   start_server(vis, port=args.port)





