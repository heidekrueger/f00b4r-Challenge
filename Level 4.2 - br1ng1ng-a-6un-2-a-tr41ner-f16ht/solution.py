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
    points_of_interest = space.get_points_of_interest(distance)
    
    # arrange the points of interest by their angle and distance
    # we want a dictionary of format {angle : distance : type of point} 

    directions = defaultdict(lambda: dict())
    for p, (what, distance) in points_of_interest.items():
        directions[space.your_position.angle(p)][distance] = what, p
        
    n_trainer_hitting_directions = 0
    for angle, point_dict in directions.items():
        # check if the direction is proper and the first thing being 
        # hit in this direction is a trainer:
        if angle is not None and point_dict[min(point_dict.keys())][0] == "trainer":
            #print point_dict[min(point_dict.keys())][1]
            n_trainer_hitting_directions += 1
            
    return n_trainer_hitting_directions

class Space():
    """Represents the vector space describing the travel of bullet half-rays through
    the room.
                        dim_y-----------|-----------------|
                        |xxxxxxxxxxxxxxx|                 |
                        |xxxxxxxxxxxxxxx|                 |
                        |xxxxxxxxxxxxxxx|                 |
    --------------------0-------------dim_x--------2*dim_x <=> 0
                        |               |                 |
                        |               |                 |
                        |               |                 |
                -dim_y == dim_y--------------------------------
    """
    def __init__(self, dimensions, your_position, trainer_position):
        self.X = dimensions[0] # room X length
        self.Y = dimensions[1] # room Y length
        self.your_position = Point(*your_position)
        self.trainer_position = Point(*trainer_position)

    def get_points_of_interest(self, max_distance):
        """Generates all 'points of interest' in the space that are a maximum of `max_distance` from
        self.your position. A point of interest is any Point that corresponds to your_position or
        trainer_position in the room.
        
        To do so, we first populate a 'large enough' rectangle in the first quadrant with all such points,
        then use symmetry to populate the other quadrants before filtering these candidate points on the 
        circle of radius max_distance around `self.your_position`.

        Returns:
            points_of_interst: Dict[Point, Tuple[String, float]] indicating the type and distance 
                to `your_position` of a point of interest
        """

        ## we'll populate the first quadrant, then use symmetry for the other quadrants.
        ## determine the number of rooms needed in x and y directions
        # worst case: your_position is at the very right of the room and point is max_distance to its right,
        # then it is in the (math.ceil(max_distance / self.X))th room to the right of your_position.X, i.e. in room
        n_rooms_x = int(max_distance / self.X) + 2
        n_rooms_y = int(max_distance / self.Y) + 2

        candidates = dict()
        for room_x in range(n_rooms_x):
            for room_y in range(n_rooms_y):
                yourself_x = room_x * self.X + self.your_position.x if not room_x % 2 \
                    else (room_x + 1) * self.X - self.your_position.x
                yourself_y = room_y * self.Y + self.your_position.y if not room_y % 2 \
                    else (room_y + 1) * self.Y - self.your_position.y
                candidates[Point(yourself_x, yourself_y)] = "yourself"

                trainer_x = room_x * self.X + self.trainer_position.x if not room_x % 2 \
                    else (room_x + 1) * self.X - self.trainer_position.x
                trainer_y = room_y * self.Y + self.trainer_position.y if not room_y % 2 \
                    else (room_y + 1) * self.Y - self.trainer_position.y
                candidates[Point(trainer_x, trainer_y)] = "trainer"

        # mirror into other quadrants
        for p in candidates.keys():
            candidates[Point( p.x, -p.y)] = candidates[p]
            candidates[Point(-p.x,  p.y)] = candidates[p]
            candidates[Point(-p.x, -p.y)] = candidates[p]

        # filter by distance
        pois = dict()
        for p in candidates.keys():
            d = self.your_position.distance(p)
            if d <= max_distance:
                pois[p] = (candidates[p], d)
        return pois

class Point(tuple):
    """A point in the vector space"""
    def __new__(self, x, y):
        return tuple.__new__(Point, (x,y))

    @property
    def x(self):
        return self[0]
    
    @property
    def y(self):
        return self[1]

    def __add__(self, vector):
        return Point(self.x + vector.x, self.y + vector.y)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __sub__(self, vector):
        return self + (-vector)

    def __abs__(self):
        """The Euclidean length of self as a vector"""
        return math.sqrt(self.x**2 + self.y**2)

    def angle(self, target):
        """Return the orientation (angle) of the vector from self to target
        in radians."""
        if target == self:
            return None
        v = target - self
        return math.atan2(v.y, v.x)

    def distance(self, other):
        """The distance between self and an `other` Point"""
        return abs(self - other)



print(solution([3,2], [1,1], [2,1], 4))


print(solution([300,275], [150,150], [185,100], 500))