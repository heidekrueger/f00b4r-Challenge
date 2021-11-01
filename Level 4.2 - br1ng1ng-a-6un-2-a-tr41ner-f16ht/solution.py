"""
Solution to bringing-a-gun-to-a-trainer-fight.

The basic outline is as follows: 
* We first construct a 2d-vector space in which straight lines correspond to
  the possible paths of bullets. To do so, we need to essentially mirror and
  concatenate the room indefinitely.
* We then determine all possible positions of all potential targets (i.e. yourself or the trainer)
  within the maximum distance from the initial position. The total number of these potential
  targets is bounded by 2*[max_distance / (room_dimensions) **2], i.e. quadratic in the problem
  size.
* With this _finite_ number of potential targets, we can cheaply enumerate their angles from the initial
  position and check which target will be hit first in all relevant directions.
"""

import math
from collections import defaultdict

def solution(dimensions, your_position, trainer_position, distance):
    """Returns the unique number of directions that you can shoot such that you will
    hit the bunny trainer without hitting yourself first.
    
    Args:
        dimensions (List[int]): x,y sizes of the room
        your_position (List[int]): your x,y coordinates
        trainer_position (List[int]): trainer's x,y coordinates
        distance (float): maximum distance that a bullet can travel
        
    Returns:
        n_directions (int): Unique number of directions in which a shot will hit the bunny trainer."""
    
    space = Space(dimensions, your_position, trainer_position)
    potential_targets = space.get_potential_targets(distance)
    
    # arrange the potential_targets by their angle and distance from your_position
    # we want a dictionary of format {angle : {distance : type of target}},
    # so that we can check the first target in each direction.

    directions = defaultdict(lambda: dict())
    for position, (target, distance) in potential_targets.items():
        directions[space.your_position.angle(position)][distance] = target
    
    # get those directions where the first thing being hit is a trainer
    trainer_hitting_directions = {
        angle for angle, targets_dict in directions.items()
        if targets_dict[min(targets_dict.keys())] == "trainer"
    }

    return len(trainer_hitting_directions)

class Space():
    """Represents a 2-dimensional Euclidean space equipped with utilities
    that describes the travel of half-rays through the room. 

    When we mirror the room at it's walls in both x- and y- directions
    indefinitely, we achieve a 2d-vector space in which the travel of the 
    bullet is described by a straight line.

           |             |
    -------|-----------dim_y-------------|-----------------|-----
           |             |xxxxxxxxxxxxxxx|                 |
           | ...         |xxxxxxxxxxxxxxx|                 |
           |             |xxxxxxxxxxxxxxx|                 |
    -------|-------------0-------------dim_x--------2*dim_x <=> 0--
           |             |               |                 |
           | ...         |               |                 |
           |             |               |                 |
           |------(-dim_y) <=> dim_y-----|-----------------|-------
           |             |
    """
    def __init__(self, dimensions, your_position, trainer_position):
        self.X = dimensions[0] # room X length
        self.Y = dimensions[1] # room Y length
        self.your_position = Point(*your_position)
        self.trainer_position = Point(*trainer_position)

    def get_potential_targets(self, max_distance):
        """Generates all potential targets of a bullet  in the space that are at 
        a maximum of `max_distance` from self.your position. By potential target 
        we mean any non-empty Point, i.e. one corresponding to`your_position` or
        `trainer_position` in the room.
        
        To do so, we first populate a 'large enough' rectangle in the first quadrant
        with all such points, then use symmetry (x- and y-axes are always walls of the room)
        to populate the other quadrants before filtering these candidate points on the 
        circle of radius max_distance around `self.your_position`.

        This method does NOT yet check, whether a potential target may be obstructed
        by another.

        Returns:
            targets: Dict[Point, Tuple[String, float]] indicating the type and distance 
                to `your_position` of a point of interest.
                Example: {Point(10,10): ("trainer", 40.0)}
        """
        ## determine the number of rooms needed in x and y directions
        # worst case: your_position is at the very right of the room and point is max_distance to its right,
        # then it is in the (math.ceil(max_distance / self.X))th room to the right of your_position.X, i.e. in room
        n_rooms_x = int(max_distance / self.X) + 2
        n_rooms_y = int(max_distance / self.Y) + 2

        candidates = dict()
        # add trainer and yourself from each mirror image of the room in Q1 to the candidates
        for room_x in range(n_rooms_x):
            for room_y in range(n_rooms_y):
                left_wall = room_x * self.X
                bottom_wall = room_y * self.Y

                # check if room is mirrored (in both dims), then determine 
                # distance either from left or right (respectively, bottom or top) wall

                yourself_x = left_wall + self.your_position.x if not room_x % 2 \
                    else left_wall + self.X - self.your_position.x
                yourself_y = bottom_wall + self.your_position.y if not room_y % 2 \
                    else bottom_wall + self.Y - self.your_position.y
                candidates[Point(yourself_x, yourself_y)] = "yourself"

                trainer_x = left_wall + self.trainer_position.x if not room_x % 2 \
                    else left_wall + self.X - self.trainer_position.x
                trainer_y = bottom_wall + self.trainer_position.y if not room_y % 2 \
                    else bottom_wall + self.Y - self.trainer_position.y
                candidates[Point(trainer_x, trainer_y)] = "trainer"

        # mirror onto other quadrants
        for p in candidates.keys():
            candidates[Point( p.x, -p.y)] = candidates[p]
            candidates[Point(-p.x,  p.y)] = candidates[p]
            candidates[Point(-p.x, -p.y)] = candidates[p]

        # filter by distance
        potential_targets = dict()
        for p in candidates.keys():
            d = self.your_position.distance(p)
            if d <= max_distance:
                potential_targets[p] = (candidates[p], d)
        return potential_targets

class Point(tuple):
    """Represents a point in 2-dimensional Euclidean vector space and
    some of its relevant operations."""
    def __new__(self, x, y):
        return tuple.__new__(Point, (x,y))
    
    def __add__(self, vector):
        return Point(self.x + vector.x, self.y + vector.y)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __sub__(self, vector):
        return self + (-vector)

    def __abs__(self):
        """The Euclidean length of self as a vector"""
        return math.sqrt(self.x**2 + self.y**2)

    @property
    def x(self):
        return self[0]
    
    @property
    def y(self):
        return self[1]

    def angle(self, target):
        """Returns the orientation (angle) of the vector from self to target
        in radians."""
        if target == self:
            return None
        v = target - self
        return math.atan2(v.y, v.x)

    def distance(self, other):
        """The Euclidean distance between self and an `other` Point"""
        return abs(self - other)

##################### Public Test Cases ###################

#print(solution([3,2], [1,1], [2,1], 4))
## --> 7

##print(solution([300,275], [150,150], [185,100], 500))
## --> 9