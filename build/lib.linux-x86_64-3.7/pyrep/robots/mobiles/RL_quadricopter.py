from pyrep.robots.mobiles.RL_drone_base import RLDrone_base
from pyrep.robots.mobiles.RL_drone_base_withoutCamera import RLDrone_base_nc


class RLQuadricopter(RLDrone_base):
    def __init__(self, count: int = 0, num_propeller:int = 4):
        super().__init__(count, num_propeller, 'Quadricopter')

class RLQuadricopter_nc(RLDrone_base_nc):
    def __init__(self, count: int = 0, num_propeller:int = 4):
        super().__init__(count, num_propeller, 'Quadricopter')
