import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpl_toolkits.axes_grid1 import make_axes_locatable


def animate2Dconfig_traj(run_dirs, Tselect=15):
    run_dir = run_dirs[0]
    # md = get_metadata(run_dir)
    # Ts = np.array(md["SweepParameterValues"])
    # print(Ts)
    fname = run_dir + 'c{}r0trajectory.npz'.format(Tselect)
    with open(fname, 'rb') as fin:
        traj = np.load(fin)
        config_traj = traj['prod_traj']
    B, N = config_traj.shape
    L = int(np.sqrt(N))
    snapshots = [config_traj[t].reshape((L, L)) for t in range(0, B)]
    snapshots = np.array(snapshots)
    # fuck need to be carefull cause this is a really long video
    # right?
    # yeah before it had a 5 sec limit, now it just goes on forever!
    # yep keep it limited from now on!
    snapshots = snapshots[0:500]
    animate2Dtrajectory(snapshots)


def animate2Dtrajectory(snapshots):
    frames, _, _ = snapshots.shape
    fps = 30
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # I like to position my colorbars this way, but you don't have to
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    cv0 = snapshots[0]
    im = ax.imshow(cv0)
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title('Frame 0')

    def animate(i):
        arr = snapshots[i]
        vmax = np.max(arr)
        vmin = np.min(arr)
        im.set_array(arr)
        im.set_clim(vmin, vmax)
        tx.set_text('Frame {0}'.format(i))
        # In this version you don't have to do anything to the colorbar,
        # it updates itself when the mappable it watches (im) changes
        # return [im, tx]

    ani = animation.FuncAnimation(
        fig, animate, frames=frames,
        interval=1000 / fps)
    ani.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.close()
