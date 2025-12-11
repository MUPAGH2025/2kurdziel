import numpy as np
from matplotlib import pyplot as plt
from open_atmos_jupyter_utils import show_anim
from PyMPDATA import ScalarField, Solver, Stepper, VectorField, Options, boundary_conditions
from matplotlib.colors import LinearSegmentedColormap

colors = [
    (0.00, 0.30, 0.60),  # deep ocean  
    (0.70, 0.90, 1.00),  # shallow water 
    (0.95, 0.85, 0.60),  # beach sand   
]
beach_cmap = LinearSegmentedColormap.from_list("beach", colors)

def plot_both(frame, *, zlim=(-0.25, 0.25), output, bathymetry):
    eta = output['h'][frame] - bathymetry
    xi, yi = np.indices(eta.shape)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 6))
    ax.plot_wireframe(xi+.5, yi+.5, eta, color='blue', linewidth=.6)
    ax.set(zlim=zlim, proj_type='ortho', title=f"Frame={frame}", zlabel="η")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
    plt.colorbar(
        ax.contourf(xi+.5, yi+.5, bathymetry, zdir='z', offset=zlim[0], cmap=beach_cmap),
        pad=.1, aspect=10, fraction=.02, label="bathymetry"
    )
    return fig

def anim_bathymetry(frame, *, bathy):
    fig = plt.figure(figsize=(7,4))
    cn = plt.contourf(bathy, levels=40, cmap=beach_cmap)
    plt.colorbar(cn, label="Wysokość dna [m]")
    plt.title("Batymetria dna")
    plt.xlabel("x [100 m]")
    plt.ylabel("y [100 m]")
    plt.tight_layout()
    return fig

def anim_cross_section_x(frame, *, output, bathy):
    eta = output["h"][frame] - bathy
    mid = eta.shape[0] // 2
    eta_line = eta[mid, :]
    fig = plt.figure(figsize=(7,4))
    plt.plot(eta_line, lw=2)
    plt.axhline(0, color='black')
    plt.xlabel("Pozycja x [100 m]")
    plt.ylabel("Wysokość fali η(x) [m]")
    plt.title(f"Przekrój wzdłuż X – krok czasowy: {frame}")
    plt.tight_layout()
    return fig

def anim_velocity(frame, *, output, bathy,skip=5):
    """
    Animacja przedstawiająca pole prędkości w m/s.
    """
    h = output["h"][frame]
    uh = output["uh"][frame]
    vh = output["vh"][frame]
    u = uh / h           # m/s
    v = vh / h           # m/s
    speed = np.sqrt(u**2 + v**2)
    fig = plt.figure(figsize=(7,4))
    plt.contourf(speed, levels=40, cmap="viridis")
    plt.colorbar(label="Prędkość |u| [m/s]")
    plt.quiver(
        np.arange(0, u.shape[1], skip),
        np.arange(0, u.shape[0], skip),
        u[::skip, ::skip],
        v[::skip, ::skip],
        color="white",
        scale=40
    )
    plt.title(f"Pole prędkości – klatka {frame}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    return fig

def slope_front_plot(slopes, c_front_list):
    # wykres prędkości czoła fali
    plt.figure(figsize=(6,4))
    plt.plot(slopes, c_front_list, "o-")
    plt.xlabel("Nachylenie brzegu [%]")
    plt.ylabel("Prędkość czoła fali [m/s]")
    plt.title("Prędkość czoła fali przy brzegu vs nachylenie")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def slope_ampl_plot(slopes, A_ref_list):
    # wykres amplitudy fali powracającej
    plt.figure(figsize=(6,4))
    plt.plot(slopes, A_ref_list, "o-")
    plt.xlabel("Nachylenie brzegu [%]")
    plt.ylabel("Amplituda fali powracającej [m]")
    plt.title("Amplituda fali odbitej vs nachylenie")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import os
os.makedirs("frames_svg", exist_ok=True)

def save_4_frames(animation_func, *, output=None, bathy=None, name="", n_frames=4):
    """
    Zapisuje 4 klatki animacji jako SVG + wyświetla każdą.
    animation_func(frame, output=..., bathy=...) → fig
    """
    total = len(output["h"]) if output is not None else 1
    frames = np.linspace(0, total-1, n_frames, dtype=int)

    print(f"\n=== Generuję {n_frames} klatek dla: {name} ===")

    for i, fr in enumerate(frames):
        fig = animation_func(fr, output=output, bathy=bathy)
        filename = f"frames_svg/{name}_{i}.svg"
        fig.savefig(filename, format="svg")
        plt.show()
        plt.close(fig)
        print(f"[OK] zapisano {filename}")

def save_4_frames_plot_both(output, bathy, name="surface3d", n_frames=4):
    """
    Zapisuje 4 klatki wizualizacji 3D (plot_both) jako SVG.
    """
    os.makedirs("frames_svg", exist_ok=True)

    total = len(output["h"])
    frames = np.linspace(0, total - 1, n_frames, dtype=int)

    print(f"\n=== Generuję {n_frames} klatki dla: {name} ===")

    for i, fr in enumerate(frames):
        fig = plot_both(fr, output=output, bathymetry=bathy)
        filename = f"frames_svg/{name}_{i}.svg"
        fig.savefig(filename, format="svg")
        plt.show()
        plt.close(fig)
        print(f"[OK] zapisano {filename}")
def save_4_frames_bathymetry(bathy, name="bathymetry", n_frames=1):
    os.makedirs("frames_svg", exist_ok=True)

    print(f"\n=== Generuję {n_frames} klatki dla: {name} ===")

    for i in range(n_frames):
        fig = anim_bathymetry(0, bathy=bathy)
        filename = f"frames_svg/{name}_{i}.svg"
        fig.savefig(filename, format="svg")
        plt.show()
        plt.close(fig)
        print(f"[OK] zapisano {filename}")

