"""
4D Visualizer: Interactive 4D -> 2D Oblique Projection
------------------------------------------------------
This script implements an interactive visualization of 4D vectors projected 
into a 2D plane using oblique projection. 

Features:
- Projection of the four coordinate axes (x, y, z, w) into 2D space.
- Combination of basis contributions to form an arbitrary 4D vector.
- Visual representation of intermediate "boxes" (3D slices) in 2D projection.
- Toggle between AXES mode (rotate the projection axes) and OBJECT mode 
  (rotate the object/vectors in 4D).
- Interactive controls via radio buttons and sliders for plane selection, 
  mode switching, basis angles, and vector components.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from matplotlib.patches import FancyArrowPatch

# Colors for axes, box edges, and sum vector
AXIS_COLORS = {"x": "#154360", "y": "#196f3d", "z": "#922b21", "w": "#7d6608"}
BOX_COLORS  = {"x": "#2980b9", "y": "#27ae60", "z": "#e74c3c", "w": "#f1c40f"}
SUM_COLOR   = "#8e44ad"

# Plane index lookup for 4D rotations (e.g., "xy" means rotate in plane (0,1))
PLANE_IDX = {"xy": (0,1), "xz": (0,2), "xw": (0,3),
             "yz": (1,2), "yw": (1,3), "zw": (2,3)}

# Axis index mapping for convenience
AXIS_IDX  = {"x":0, "y":1, "z":2, "w":3}

def plane_rot4(i, j, theta):
    """Return a 4×4 rotation matrix in the (i,j)-plane by angle theta."""
    R = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    R[i,i] = c;  R[i,j] = -s
    R[j,i] = s;  R[j,j] =  c
    return R

def reorthonormalize(R):
    """
    Re-orthonormalize a 4x4 rotation matrix using QR decomposition.
    Ensures numerical stability after many incremental rotations.
    """
    Q, _ = np.linalg.qr(R)
    if np.linalg.det(Q) < 0:
        Q[:,0] *= -1.0
    return Q

def make_A(angles_deg=(0,30,90,120), scales=(1,1,1,1)):
    """
    Construct a 2×4 projection matrix A.
    Each column corresponds to the projection direction of a 4D basis vector.
    Angles are given in degrees on the 2D projection plane.
    """
    ang = np.radians(np.array(angles_deg, float))
    dirs = np.vstack((np.cos(ang), np.sin(ang)))
    return dirs * np.array(scales, float)

class FourDVisualizer:
    """
    Main application class for the 4D visualizer.
    Handles state (rotation, projection, target vector), UI controls, 
    and drawing routines.
    """
    def __init__(self):
        # Rotation state (4×4 matrix for OBJECT mode)
        self.R = np.eye(4)

        # Initial projection angles (degrees) for the four basis vectors
        self.angles_deg = [0,30,90,120]
        self.A = make_A(self.angles_deg)

        # Default mode: AXES (rotate the projection axes)
        self.mode = "AXES"

        # Initial rotation planes for dragging
        self.h_plane, self.v_plane = "xy","zw"

        # Mouse sensitivity and re-orthonormalization frequency
        self.sensitivity_x = 0.01; self.sensitivity_y = 0.01
        self.reortho_every = 10; self._drag_count = 0
        self.dragging = False; self.last_xy = None

        # Axis visualization length (scaling factor)
        self.axis_len = 1.2

        # Target vector (controlled by sliders)
        self.v_target = np.array([0.3,0.3,0.3,0.3], float)

        # Setup figure and axes
        self.fig, self.ax = plt.subplots(figsize=(9.2,8.2))
        self.fig.canvas.manager.set_window_title("4D Visualizer")
        plt.subplots_adjust(left=0.06,right=0.84,top=0.97,bottom=0.16)
        self.ax.set_aspect("equal"); self.ax.set_axis_off()
        self.ax.set_title("4D Visualizer: 4D → 2D Oblique Projection")

        # Build user interface (sliders, radio buttons)
        self._build_ui()
        self._init_plot()

        # Mouse interaction callbacks
        self.fig.canvas.mpl_connect('button_press_event',   self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event',  self.on_motion)

        plt.show()

    def _build_ui(self):
        """Create UI elements: plane selectors, mode switch, sliders."""
        planes = ["xy","xz","xw","yz","yw","zw"]

        # Right-side radio: horizontal plane selection
        ax_h = plt.axes([0.86, 0.60, 0.12, 0.30])
        self.h_radio = RadioButtons(ax_h, planes, active=planes.index("xy"))
        self.h_radio.on_clicked(lambda lab: self._set_plane("H", lab))
        ax_h.set_title("H-plane")

        # Right-side radio: vertical plane selection
        ax_v = plt.axes([0.86, 0.15, 0.12, 0.30])
        self.v_radio = RadioButtons(ax_v, planes, active=planes.index("zw"))
        self.v_radio.on_clicked(lambda lab: self._set_plane("V", lab))
        ax_v.set_title("V-plane")

        # Top-left radio: mode selection (AXES vs OBJECT)
        ax_mode = plt.axes([0.06, 0.905, 0.18, 0.045])
        self.mode_radio = RadioButtons(ax_mode, ("AXES","OBJECT"), active=0)
        self.mode_radio.on_clicked(lambda lab: setattr(self, "mode", lab))

        # Sliders for projection angles (θx, θy, θz, θw)
        self.sliders = []
        for i,lbl in enumerate(["θx","θy","θz","θw"]):
            ax_sl = plt.axes([0.06+i*0.18,0.15,0.10,0.03])
            s = Slider(ax_sl,lbl,0,360,valinit=self.angles_deg[i])
            s.on_changed(self._on_angle_changed(i))
            self.sliders.append(s)

        # Sliders for target vector components (Vx, Vy, Vz, Vw)
        self.v_sliders = []
        for k,lbl in enumerate(["x","y","z","w"]):
            axv = plt.axes([0.06+k*0.18,0.11,0.10,0.03])
            sv = Slider(axv,f"V{lbl}",0.0,1.0,valinit=self.v_target[k])
            sv.on_changed(self._on_v_changed(k))
            self.v_sliders.append(sv)

    def _on_angle_changed(self,idx):
        """Return callback for angle slider change."""
        def _cb(val):
            self.angles_deg[idx] = float(val)
            self.A = make_A(self.angles_deg)
            self._redraw()
        return _cb

    def _on_v_changed(self,k):
        """Return callback for vector component slider change."""
        def _cb(val):
            self.v_target[k] = float(val)
            self._redraw()
        return _cb

    def _set_plane(self, which, label):
        """Update selected drag plane (H or V)."""
        if which == "H": self.h_plane = label
        else: self.v_plane = label

    def _init_plot(self):
        """Initialize empty plot elements for axes, arrows, boxes, etc."""
        self.axis_lines,self.axis_texts=[],[]
        for a in ["x","y","z","w"]:
            ln, = self.ax.plot([],[],lw=2,color=AXIS_COLORS[a])
            tx = self.ax.text(0,0,a,color=AXIS_COLORS[a],ha="center",va="center")
            self.axis_lines.append(ln); self.axis_texts.append(tx)

        # Arrows at axis tips (positive direction only)
        self.axis_arrows = []
        for a in ["x","y","z","w"]:
            arr = FancyArrowPatch((0,0),(0,0),
                                  arrowstyle='-|>', mutation_scale=14,
                                  lw=0, color=AXIS_COLORS[a])
            self.ax.add_patch(arr)
            self.axis_arrows.append(arr)

        # Sum vector (origin → combined target vector)
        self.line_sum, = self.ax.plot([],[],lw=2.6,color=SUM_COLOR)
        self.sum_point = self.ax.scatter([],[],s=70,color=SUM_COLOR,zorder=5)

        # Box definitions (3D slices in 4D, projected to 2D)
        self.box_defs=[("xyz","x","y","z"),("xyw","x","y","w"),
                       ("xzw","x","z","w"),("yzw","y","z","w")]
        self.box_edges=[]; self.final_lines=[]
        for _ in self.box_defs:
            edges=[self.ax.plot([],[],lw=1.6)[0] for __ in range(12)]
            self.box_edges.append(edges)
            self.final_lines.append(self.ax.plot([],[],lw=2.2)[0])

        self.ax.set_xlim(-1.5,1.5); self.ax.set_ylim(-1.5,1.5)
        self._redraw()

    def _redraw(self):
        """Redraw all plot elements according to current state."""
        E4=np.eye(4)

        # Axes (kept fixed relative to the screen)
        P_axes=(self.A@E4).T

        # Draw each full axis line (negative to positive) and arrowhead at +tip
        for i,a in enumerate(["x","y","z","w"]):
            x1=self.axis_len*P_axes[i,0]; y1=self.axis_len*P_axes[i,1]
            self.axis_lines[i].set_data([-x1, x1], [-y1, y1])
            # Label remains near the positive tip
            self.axis_texts[i].set_position((1.1*x1,1.1*y1))

            # Positive-direction arrowhead only
            d = np.array([P_axes[i,0], P_axes[i,1]], dtype=float)
            n = np.linalg.norm(d)
            if n > 1e-9:
                u = d / n
                tip = np.array([1.035*x1, 1.035*y1])
                arrow_len = 0.08
                tail = tip - arrow_len * u
                self.axis_arrows[i].set_visible(True)
                self.axis_arrows[i].set_positions(tail, tip)
                self.axis_arrows[i].set_color(AXIS_COLORS[a])
            else:
                self.axis_arrows[i].set_visible(False)

        # Combined target vector in the chosen mode
        vR=self.R@self.v_target if self.mode=="OBJECT" else self.v_target
        a_cols=[self.A[:,k] for k in range(4)]
        p=[vR[k]*a_cols[k] for k in range(4)]
        SUM=p[0]+p[1]+p[2]+p[3]

        # Always draw sum vector
        self.line_sum.set_data([0.0,SUM[0]],[0.0,SUM[1]])
        self.sum_point.set_offsets([SUM[0],SUM[1]])

        # Boxes and remaining vectors
        for bi,(bname,b1,b2,sft) in enumerate(self.box_defs):
            show_box = True  # always show boxes
            idx1,idx2,idxs=AXIS_IDX[b1],AXIS_IDX[b2],AXIS_IDX[sft]
            remaining=list({"x","y","z","w"}-{b1,b2,sft})[0]
            idxr=AXIS_IDX[remaining]

            U,Vv,Sft=p[idx1],p[idx2],p[idxs]
            O0=np.array([0.0,0.0])
            U0,V0,UV=U,Vv,U+Vv
            O1=O0+Sft; U1=U0+Sft; V1=V0+Sft; UV1=UV+Sft

            edges=[(O0,U0,b1),(U0,UV,b2),(UV,V0,b1),(V0,O0,b2),
                   (O1,U1,b1),(U1,UV1,b2),(UV1,V1,b1),(V1,O1,b2),
                   (O0,O1,sft),(U0,U1,sft),(V0,V1,sft),(UV,UV1,sft)]
            if show_box:
                for k,(P0,P1,axname) in enumerate(edges):
                    ln=self.box_edges[bi][k]
                    ln.set_data([P0[0],P1[0]],[P0[1],P1[1]])
                    ln.set_color(BOX_COLORS[axname])
            else:
                for ln in self.box_edges[bi]:
                    ln.set_data([],[])

            # Always draw final remaining axis segment
            start=U+Vv+Sft; end=start+p[idxr]
            lnfin=self.final_lines[bi]
            lnfin.set_data([start[0],end[0]],[start[1],end[1]])
            lnfin.set_color(BOX_COLORS[remaining])

        self.fig.canvas.draw_idle()

    # --- Mouse interaction handlers ---
    def on_press(self,e):
        """Start drag rotation if mouse is pressed inside axes."""
        if e.inaxes==self.ax:
            self.dragging=True; self.last_xy=(e.x,e.y)

    def on_release(self,e):
        """Stop drag rotation, re-orthonormalize if in OBJECT mode."""
        self.dragging=False; self.last_xy=None
        if self.mode=="OBJECT":
            self.R=reorthonormalize(self.R)
        self._redraw()

    def _apply_drag_rotation(self,dth,plane):
        """Apply incremental 4D rotation in the given plane."""
        i,j=PLANE_IDX[plane]; G=plane_rot4(i,j,dth)
        if self.mode=="OBJECT":
            self.R=G@self.R
        else:
            self.A=self.A@G

    def on_motion(self,e):
        """Update rotation based on mouse motion while dragging."""
        if not self.dragging or self.last_xy is None or (e.x is None) or (e.y is None):
            return
        dx=e.x-self.last_xy[0]; dy=e.y-self.last_xy[1]
        self.last_xy=(e.x,e.y)
        self._apply_drag_rotation(self.sensitivity_y*dy,self.v_plane)
        self._apply_drag_rotation(self.sensitivity_x*dx,self.h_plane)
        self._drag_count+=1
        if self.mode=="OBJECT" and (self._drag_count%self.reortho_every==0):
            self.R=reorthonormalize(self.R)
        self._redraw()

if __name__=="__main__":
    FourDVisualizer()
