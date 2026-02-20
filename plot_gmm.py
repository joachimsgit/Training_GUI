import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D


# ---------- Utilities ----------

def load_gmm_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    components = []
    for name, entry in data.items():
        mean = np.array([
            entry["contrast"]["b"],
            entry["contrast"]["g"],
            entry["contrast"]["r"]
        ])
        cov = np.array(entry["covariance_matrix"])
        components.append((name, mean, cov))
    return components


def plot_3d_gaussian(ax, mean, cov, n_std=2.0, resolution=30):
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort by eigenvalue size
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scale sphere to ellipsoid
    radii = n_std * np.sqrt(eigvals)
    ellipsoid = np.stack((x, y, z), axis=-1)
    ellipsoid = ellipsoid @ np.diag(radii)
    ellipsoid = ellipsoid @ eigvecs.T
    ellipsoid += mean

    ax.plot_wireframe(
        ellipsoid[..., 0],
        ellipsoid[..., 1],
        ellipsoid[..., 2],
        linewidth=0.5,
        alpha=0.6
    )


def plot_2d_gaussian(ax, mean, cov, idx1, idx2, n_std=2.0, label=None):
    cov_2d = cov[np.ix_([idx1, idx2], [idx1, idx2])]
    mean_2d = mean[[idx1, idx2]]

    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)

    ellipse = Ellipse(
        xy=mean_2d,
        width=width,
        height=height,
        angle=angle,
        fill=False
    )
    ax.add_patch(ellipse)
    ax.scatter(*mean_2d, s=20)

    if label is not None:
        ax.text(mean_2d[0], mean_2d[1], label)



# ---------- Main plotting ----------

def plot_gmm(components):
    fig = plt.figure(figsize=(14, 10))

    #set limits for better visualization
    b_lims = (-0.5, 0.1)
    g_lims = (-0.8, 0.1)
    r_lims = (-0.9, 0.1)

    # 3D plot
    ax3d = fig.add_subplot(221, projection="3d")
    ax3d.set_title("3D Gaussian Mixture Model")
    ax3d.set_xlabel("b")
    ax3d.set_ylabel("g")
    ax3d.set_zlabel("r")
    ax3d.set_xlim(b_lims)
    ax3d.set_ylim(g_lims)
    ax3d.set_zlim(r_lims)

    # 2D plots
    ax_xy = fig.add_subplot(222)
    ax_xz = fig.add_subplot(223)
    ax_yz = fig.add_subplot(224)

    ax_xy.set_title("b–g projection")
    ax_xz.set_title("b–r projection")
    ax_yz.set_title("g–r projection")

    ax_xy.set_xlim(b_lims)
    ax_xy.set_ylim(g_lims)

    ax_xz.set_xlim(b_lims)
    ax_xz.set_ylim(r_lims)

    ax_yz.set_xlim(g_lims)
    ax_yz.set_ylim(r_lims)

    

    for name, mean, cov in components:
        plot_3d_gaussian(ax3d, mean, cov)
        plot_2d_gaussian(ax_xy, mean, cov, 0, 1)
        plot_2d_gaussian(ax_xz, mean, cov, 0, 2)
        plot_2d_gaussian(ax_yz, mean, cov, 1, 2)

    for ax in (ax_xy, ax_xz, ax_yz):
        #ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True)

    for name, mean, cov in components:
        plot_3d_gaussian(ax3d, mean, cov)
        ax3d.text(mean[0], mean[1], mean[2], name)

        plot_2d_gaussian(ax_xy, mean, cov, 0, 1, label=name)
        plot_2d_gaussian(ax_xz, mean, cov, 0, 2, label=name)
        plot_2d_gaussian(ax_yz, mean, cov, 1, 2, label=name)

    plt.tight_layout()
    plt.show()


# ---------- Entry point ----------

if __name__ == "__main__":
    components = load_gmm_from_json("Materials/graphene_dataset2026-01-16/GMM/GMM_parameters.json")
    #components = load_gmm_from_json("Materials/graphene_dataset2026-01-16/graphene_dataset5-6-Layer2026-01-16/GMM/GMM_parameters.json")
    plot_gmm(components)
