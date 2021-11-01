import math
import fractions

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

    all_dirs = get_directions(distance)
    trainer_hitting_dirs = {d for d in all_dirs if space.target(d, distance) == 'trainer'}
    #print trainer_hitting_dirs
    return len(trainer_hitting_dirs)

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

    def target(self, direction, max_distance):
        position = self.your_position
        
        while max_distance >= direction.step_length:
            position += direction
            room_position = position.get_room_coordinates(self)
            max_distance -= direction.step_length

            if room_position == self.your_position:
                return "yourself"
            elif room_position == self.trainer_position:
                return "trainer"
        
        # nothing hit after max distance
        return "nothing"

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

    def get_room_coordinates(self, space):
        """Reduces coordinates x,y in ray-space to coordinates x,y in the room.
        
        We have
        x (equiv) x + 2*X
        """
        # are we in an "ascending" or descending part of the space?
        desc_x = (self.x // space.X) % 2 # 0 if ascending, 1 if descending
        desc_y = (self.y // space.Y) % 2
        
        x = space.X - self.x % space.X if desc_x else self.x % space.X
        y = space.Y - self.y % space.Y if desc_y else self.y % space.Y

        return Point(x, y)

    def __add__(self, vector):
        return Point(self.x + vector.x, self.y + vector.y)


class Direction(tuple):
    def __new__(self, x, y):
        if x==0 and y>0:
            y = 1
        if y==0 and x>0:
            x = 1
        gcd = abs(fractions.gcd(x,y)) if x>0 and y>0 else 1
        return tuple.__new__(Direction, (x/gcd, y/gcd))

    @property
    def x(self):
        return self[0]
    
    @property
    def y(self):
        return self[1]

    @property
    def step_length(self):
        """The minimum vector length in this direction that constitutes a lattice step."""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def get_symmetries(self):
        """Returns the set of all eight symmetries of self along the horizontal, vertical and
        diagonal mirror axes of the circle"""
        return {Direction(x,y) for x,y in zip(
            [self.x, self.x,-self.x,-self.x, self.y, self.y,-self.y,-self.y],
            [self.y,-self.y, self.y,-self.y, self.x,-self.x, self.x,-self.x]
            )}

def get_directions(max_distance):
    """Returns all unique direction vectors (x,y) that contain an integral
    lattice step with no more than max_distance.

    To do so, we scan the lattice points within a 45 degree circle segment,
    to find valid directions, reduce those that are linearly dependent,
    then add the 8 existing symmetries across the circle.
    """

    initial_directions = set()
    for x in range(int(max_distance) + 1):
        # we need x**2 + y**2 <= max_distance**2
        for y in range(int(math.sqrt(max_distance ** 2 - x**2)) + 1):
            ## there's room for optimization here! Can we skip y that aren't coprime to x?
            initial_directions.add(Direction(x,y))
            
    directions = set()
    for d in initial_directions:
        directions.update(d.get_symmetries())
    return directions



print(solution([3,2], [1,1], [2,1], 4))


print(solution([300,275], [150,150], [185,100], 500))