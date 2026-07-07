from monzo import color_sequence, Monzo, target_max_int
from vispy import scene
from vispy.color import ColorArray, get_colormap
from vispy.visuals.axis import _get_ticks_talbot
import numpy as np

from utils import factor, make_spf, sigmoid

def scatter_view(df, x, y, value_col=None, continuous=False, size=9, title="scatter", bgcolor="white", line_segments=False):
    factor_filter = False
    if factor_filter:
        spf = make_spf(target_max_int)
        def has_2_or_3_to_first_power(n):
            exps = dict(factor(n, spf))
            if (n % 10000 == 0):
                print(f"Processed {n} of {target_max_int} integers")
            return (exps.get(2, 0) == 1) or (exps.get(3, 0) == 1)
        keep_ints = {i for i in df["int"].unique() if has_2_or_3_to_first_power(i)}
        df = df[df["int"].isin(keep_ints)].reset_index(drop=True)
   
    pos = np.column_stack([df[x].to_numpy(float), df[y].to_numpy(float)])

    class TrackpadCamera(scene.PanZoomCamera):
        on_change = None  

        def viewbox_mouse_event(self, event):
            if event.type == "mouse_wheel":
                if not self.interactive:
                    return
                center = self._scene_transform.imap(event.pos)
                dx, dy = event.delta[0], event.delta[1]
                fx = (1 + self.zoom_factor) ** (-dx * 30)
                fy = (1 + self.zoom_factor) ** (-dy * 30)
                self.zoom((fx, fy), center)   # tuple = independent x/y zoom
                event.handled = True
            else:
                super().viewbox_mouse_event(event)

    def view_changed(self):
        super().view_changed()
        if self.on_change is not None:
            self.on_change()

    if value_col is None:
        face_color = "white"
    elif continuous:                      
        vals = df[value_col].to_numpy(float)
        norm = (vals - vals.min()) / (np.ptp(vals) or 1)
        face_color = get_colormap("viridis").map(norm)
    elif title == "modular coordinates":
        vals = df[value_col].to_numpy(float)
        norm = 1- sigmoid((vals - vals.min()) / (np.ptp(vals) or 1) -0.5)
        face_color = get_colormap("greens").map(norm)
    else:                                 
        idx = (df[value_col].to_numpy().astype(int) - 1) % len(color_sequence)
        face_color = ColorArray(color_sequence).rgba[idx]

    canvas = scene.SceneCanvas(keys="interactive", show=True, title=title, bgcolor=bgcolor)
    grid = canvas.central_widget.add_grid()

    yaxis = scene.AxisWidget(orientation="left",
                         axis_color="black", tick_color="black",
                         text_color="black", tick_font_size=5)
    yaxis.width_max = 60
    grid.add_widget(yaxis, row=0, col=0)

    xaxis = scene.AxisWidget(orientation="bottom",
                            axis_color="black", tick_color="black",
                            text_color="black", tick_font_size=5)
    xaxis.height_max = 40
    grid.add_widget(xaxis, row=1, col=1)

    view = grid.add_view(row=0, col=1)
    view.camera = TrackpadCamera(aspect=None)


    def horizontal_segments(pos):
        x, y = pos[:, 0], pos[:, 1]
        from collections import defaultdict
        max_y_idx = {}
        for idx in range(len(x)):
            xi, yi = x[idx], y[idx]
            if xi not in max_y_idx or yi > y[max_y_idx[xi]]:
                max_y_idx[xi] = idx

        sorted_xs = sorted(max_y_idx.keys())
        segments = []
        for i in range(len(sorted_xs) - 1):
            idx1 = max_y_idx[sorted_xs[i]]
            idx2 = max_y_idx[sorted_xs[i+1]]
            if np.abs(sorted_xs[i+1] - sorted_xs[i]) < 2:
                segments.append([idx1, idx2])
        return np.array(segments)

    def _ticks(lo, hi, px, dpi):
        if hi <= lo or px <= 0:
            return np.array([])
        return _get_ticks_talbot(lo, hi, px / dpi, density=7)   

    def update_grid():
        if True:
            return
        rect = view.camera.rect
        xs = _ticks(rect.left, rect.right, view.rect.width, canvas.dpi)
        ys = _ticks(rect.bottom, rect.top, view.rect.height, canvas.dpi)
        segs = []
        for xv in xs:
            segs += [[xv, rect.bottom], [xv, rect.top]]
        for yv in ys:
            segs += [[rect.left, yv], [rect.right, yv]]
        if segs:
            grid_lines.set_data(pos=np.array(segs, float), connect="segments")
        else:
            grid_lines.set_data(pos=np.zeros((0, 2), float))

    view.camera.on_change = update_grid
    canvas.events.resize.connect(lambda e: update_grid())
    grid_lines = scene.visuals.Line(color=(0.0, 0.0, 0.0, 0.03), width=2, method="gl", antialias=True, parent=view.scene)
    grid_lines.order = -1
    update_grid()

    xaxis.link_view(view)   
    yaxis.link_view(view)

    @canvas.events.key_press.connect
    def on_key_press(event):
        step = 1.25
        zin, zout = 1 / step, step
        if event.key == "Right":
            view.camera.zoom((zin, 1.0))   
        elif event.key == "Left":
            view.camera.zoom((zout, 1.0))  
        elif event.key == "Up":
            view.camera.zoom((1.0, zin))   
        elif event.key == "Down":
            view.camera.zoom((1.0, zout))  
        elif event.key == "R":
            view.camera.set_range()        

    markers = scene.visuals.Markers()
    markers.set_data(pos, face_color=face_color, size=size,
                     edge_width=0, edge_color=None, symbol='square')
    markers.set_gl_state(depth_test=False, blend=True,
                         blend_func=("src_alpha", "one_minus_src_alpha"))
    view.add(markers)
    view.camera.set_range()


    def print_find(what, a, b, c):
        print(f"{what}: ({a[0]}, {a[1]}), ({b[0]}, {b[1]}), ({c[0]}, {c[1]})")
        print(f"\tMonzo 1: {Monzo.get(a[0])}")
        print(f"\tMonzo 2: {Monzo.get(b[0])}")
        print(f"\tMonzo 3: {Monzo.get(c[0])}")

    draw_triangles = False
    draw_triples = False
    draw_quintuples = False

    if line_segments:
        connect = horizontal_segments(pos)
        if len(connect):
            lines = scene.visuals.Line(pos=pos, connect=connect,
                                    color=(0.05, 0.4, 0.2, 0.2), width=2, method="gl", antialias=True)
            lines.order = 1
            view.add(lines)
    elif title == "square norm":
        up = True
        relevant_pos = [p for p in pos if p[1] > 20]
        for a in pos:
            if draw_quintuples:
                if a[0] % 5 == 0:
                    k = 0
                    f = a[0] // 5
                    while pos[k][0] <= f and k < len(pos):
                        if pos[k][0] == f:
                            quint_line = scene.visuals.Line(pos=[a, pos[k]],
                                                            color=(0.05, 0.05, 0.5, 0.2), width=2, method="gl", antialias=True)
                            quint_line.order = 1
                            view.add(quint_line)
                        k += 1
            if draw_triples or draw_triangles:
                for b in [p for p in relevant_pos if p[0] > a[0] and p[1] == a[1]]:
                    if draw_triangles:
                        for c in [p for p in relevant_pos if a[0] + b[0] == 2 * p[0] and (a[1] < p[1] if up else a[1] > p[1])]:
                            # print_find("Triangle found", a, b, c)
                            triangle_lines = scene.visuals.Line(pos=[a, b, c, a],
                                                                color=(0.5, 0.05, 0.2, 0.05), width=1, method="gl", antialias=True)
                            triangle_lines.order = 1
                            view.add(triangle_lines)
                    if draw_triples:
                        for c in [p for p in relevant_pos if p[1] == a[1] and p[0] > b[0] and b[0] - a[0] == p[0] - b[0]]:
                            print_find("Triple found", a, b, c)
                            lines = scene.visuals.Line(pos=[a, c],
                                            color=(0.05, 0.05, 0.5, 0.2), width=2, method="gl", antialias=True)
                            lines.order = 1
                            view.add(lines)

    hover_text = scene.visuals.Text(
        "", color="red", anchor_x="left", anchor_y="bottom",
        font_size=10, parent=canvas.scene,
    )

    px_cache = {"pts": None}

    @canvas.events.mouse_move.connect
    def on_mouse_move(event):
        px_all = px_cache["pts"]
        d = np.linalg.norm(px_all - np.array(event.pos), axis=1)
        i = int(np.argmin(d))
        if d[i] < max(size, 12):                           
            hover_text.text = f"({pos[i, 0]:g}, {pos[i, 1]:g})"
            hover_text.pos = event.pos + np.array([10, -10])
        else:
            hover_text.text = ""

    def refresh_px():
        inv = markers.get_transform("visual", "canvas")
        px_cache["pts"] = inv.map(pos)[:, :2]

    old_on_change = view.camera.on_change
    view.camera.on_change = lambda: (old_on_change(), refresh_px())
    canvas.events.resize.connect(lambda e: refresh_px())
    canvas.events.draw.connect(lambda e: refresh_px())

    return canvas

