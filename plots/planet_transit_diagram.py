import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def theta_entry(a_rs=3., rp_rs=0.5):
    return -np.arcsin((1 + rp_rs) / a_rs)


def circ_pos(center, theta, r):
    x, y = center
    y1 = y - r * np.cos(theta)
    x1 = x + r * np.sin(theta)

    return (x1, y1)


def draw_system(ax, a_rs=3., rp_rs=0.5, theta=None,
                rs=0.1, xs=(0.5, 0.5), label_radii=True,
                draw_planet=True, draw_sun=True,
                planet_circle_kwargs=dict(color='0.7'),
                sun_circle_kwargs=dict(color='r', alpha=0.5),
                draw_a=True, a_kwargs=dict(color='k', ls='-'),
                label_a=True):
    if theta is None:
        theta = theta_entry(a_rs=a_rs, rp_rs=rp_rs)

    xp = circ_pos(xs, theta, a_rs * rs)

    if draw_planet:
        planet = plt.Circle(xp, rp_rs * rs, **planet_circle_kwargs)
        ax.add_artist(planet)
    if draw_sun:
        star = plt.Circle(xs, rs, **sun_circle_kwargs)
        ax.add_artist(star)

    ax.plot([xs[0] - rs, xs[0] - rs], [0, xs[1]], color='k', ls=':')
    ax.plot([xs[0] + rs, xs[0] + rs], [0, xs[1]], color='k', ls=':')

    if draw_a:
        ax.plot(*zip(xs, xp), **a_kwargs)

    if draw_a and label_a:
        atext_xy = tuple(0.5 * (np.array(xs) + np.array(xp)))

        acoords = (-5 * np.cos(theta), -5 * np.sin(theta))
        ax.annotate("$a$", xy=atext_xy, xytext=acoords,
                    textcoords='offset points', xycoords='data',
                    ha='right', va='bottom' if theta < 0 else 'top',
                    fontsize=20)

    if label_radii:
        ax.plot([xs[0], xs[0] + rs], [xs[1], xs[1]], ls='--', color='k')
        ax.annotate("$R_{\\star}$", xy=(xs[0] + 0.5 * rs, xs[1]),
                    xytext=(0, 3),
                    textcoords='offset points', xycoords='data',
                    ha='center', va='bottom', fontsize=20)

        ax.plot([xp[0], xp[0] - rs * rp_rs],
                [xp[1], xp[1]], ls='--', color='k')
        ax.annotate("$R_p$", xy=(xp[0] - 0.5 * rs * rp_rs, xp[1]),
                    xytext=(-5, -5),
                    textcoords='offset points', xycoords='data',
                    ha='right', va='top', fontsize=20)

f, ax = plt.subplots()

x0 = (0.5, 0.8)
theta = -theta_entry()
rs = 0.2

draw_system(ax, theta=-theta, rs=rs, xs=x0)

draw_system(ax, theta=theta, rs=rs, xs=x0,
            draw_sun=False, label_radii=False, label_a=False)

arc_rad = 1.2 * rs
arc = mpatches.Arc(x0, 2 * arc_rad, 2 * arc_rad,
                   theta1=np.degrees(-np.pi/2 - theta),
                   theta2=np.degrees(-np.pi/2 + theta))

arc2 = mpatches.Arc(x0, 6 * rs, 6 * rs,
                    theta1=np.degrees(-np.pi/2 - 2 * theta),
                    theta2=np.degrees(-np.pi/2 + 2 * theta),
                    ls='--', color='k')

ax.add_patch(arc)
ax.add_patch(arc2)
ax.annotate('$\\theta$', xy=(x0[0], x0[1] - arc_rad), xytext=(0, -5),
            textcoords='offset points', fontsize=20, va='top', ha='center')


ax.axis('off')
ax.set_aspect('equal', 'datalim')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.show()
