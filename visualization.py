from PIL import Image, ImageDraw
from monzo import color_sequence
from vispy import scene
from vispy.color import ColorArray, get_colormap
import numpy as np

def scatter_view(df, x, y, value_col=None, continuous=False,
                 size=8, title="scatter", bgcolor="white"):
    pos = np.column_stack([df[x].to_numpy(float), df[y].to_numpy(float)])

    if value_col is None:
        face_color = "white"
    elif continuous:                      
        vals = df[value_col].to_numpy(float)
        norm = (vals - vals.min()) / (np.ptp(vals) or 1)
        face_color = get_colormap("viridis").map(norm)
    else:                                 
        idx = (df[value_col].to_numpy().astype(int) - 1) % len(color_sequence)
        face_color = ColorArray(color_sequence).rgba[idx]

    canvas = scene.SceneCanvas(keys="interactive", show=True,
                               title=title, bgcolor=bgcolor)
    grid = canvas.central_widget.add_grid()

    yaxis = scene.AxisWidget(orientation="left")
    yaxis.width_max = 60
    grid.add_widget(yaxis, row=0, col=0)

    xaxis = scene.AxisWidget(orientation="bottom")
    xaxis.height_max = 40
    grid.add_widget(xaxis, row=1, col=1)

    view = grid.add_view(row=0, col=1)
    view.camera = scene.PanZoomCamera(aspect=None)

    markers = scene.visuals.Markers()
    markers.set_data(pos, face_color=face_color, size=size,
                     edge_width=0, edge_color=None)
    markers.set_gl_state(depth_test=False, blend=True,
                         blend_func=("src_alpha", "one_minus_src_alpha"))
    view.add(markers)
    view.camera.set_range()

    connect = horizontal_segments(pos)
    if len(connect):
        lines = scene.visuals.Line(pos=pos, connect=connect,
                                color=(0.05, 0.4, 0.1, 0.2), width=1, method="gl", antialias=True)
        view.add(lines)

    xaxis.link_view(view)   
    yaxis.link_view(view)
    return canvas

def horizontal_segments(pos):
    x, y = pos[:, 0], pos[:, 1]
    order = np.lexsort((y, x))         
    xs = x[order]
    keep = np.empty(len(order), dtype=bool)
    keep[:-1] = xs[1:] != xs[:-1]       
    keep[-1] = True                     
    reps = order[keep]                  
    return np.column_stack([reps[:-1], reps[1:]])

def render_grid(values_2d, cell_size=20, gap=2, bg="#000000", path="grid.png"):
    cols = len(values_2d)
    rows = max(len(r) for r in values_2d)
    rows = max([i for i in range(rows) if len([j for j in range(cols) if values_2d[j][i] != "#000000"]) > 0]) + 20
    w = cols * cell_size + (cols + 1) * gap
    h = rows * cell_size + (rows + 1) * gap

    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    for r, row in enumerate(values_2d):
        for c, color in enumerate(row):
            y0 = gap + c * (cell_size + gap)
            x0 = gap + r * (cell_size + gap)
            draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill=color)

    img.save(path)
    return path