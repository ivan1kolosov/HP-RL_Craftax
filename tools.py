import bz2
import pickle
import time
from pathlib import Path

from craftax.craftax.constants import Achievement

def save_traj_history(traj_history):
    save_name = f"play_data/trajectories_{int(time.time())}"
    save_name += ".pkl"
    Path("play_data").mkdir(parents=True, exist_ok=True)    
    with bz2.BZ2File(save_name + ".pbz2", "w") as f:
        pickle.dump(traj_history, f)

def print_new_achievements(old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(
                f"{Achievement(i).name} ({new_achievements.sum()}/{len(Achievement)})"
            )
