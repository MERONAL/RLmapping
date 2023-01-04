# String constants for keys used in the environment.
STATUS = "status"
FAILED = "failed"
SUCCESS = "success"
RUNNING = "running"

REAL_PATH = 'real_path'
PLANNED_PATH = 'planned_path'

POSITION = 'position'
ORIENTATION = 'orientation'



# other constant

class LocalConstants(object):
    SuccessDistance = 0.5 # local avoidance success distance between car and target postion: 0.5m
    CollisionDistance = 0.45 # local avoidance collision distance: 0.45m, when laser scan distance is less than this value, it is considered to be a collision


class GlobalConstants(object):
    pass