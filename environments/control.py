import traci
import numpy as np
from environments.config import Settings

def get_ego_start_speed():
    if Settings.RANDOMIZE_START_SPEED:
        start_speed = np.random.normal(Settings.START_SPEED, Settings.START_SPEED_VARIANCE)
        start_speed = np.clip(start_speed, Settings.MIN_START_SPEED, Settings.MAX_START_SPEED)
    else:
        start_speed = Settings.START_SPEED
    return start_speed


def add_ego_car(start_velocity):
    # traci.vehicle.add("ego", "ramp_merge", "egocar", departSpeed=start_velocity, departPos=40, arrivalPos=50)
    traci.vehicle.add("ego", "ramp_merge", "egocar", departSpeed=start_velocity)
    traci.vehicle.setSpeedMode("ego", 0)
    traci.vehicle.setSpeed("ego", start_velocity)
    traci.vehicle.setColor("ego",color=(0,0,255))


