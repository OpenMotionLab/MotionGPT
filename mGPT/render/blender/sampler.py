import numpy as np

def get_frameidx(*, mode, nframes, exact_frame, frames_to_keep):
    if mode == "sequence":
        frameidx = np.linspace(0, nframes - 1, frames_to_keep)
        frameidx = np.round(frameidx).astype(int)
        frameidx = list(frameidx)
    elif mode == "frame":
        index_frame = int(exact_frame*nframes)
        frameidx = [index_frame]
    elif mode == "video":
        frameidx = range(0, nframes)
    else:
        raise ValueError(f"Not support {mode} render mode")
    return frameidx
