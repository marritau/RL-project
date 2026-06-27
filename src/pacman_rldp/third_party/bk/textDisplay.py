
import time
try: 
    from . import pacman
except:
    pass

DRAW_EVERY = 1
SLEEP_TIME = 0 # This can be overwritten by __init__
DISPLAY_MOVES = False
QUIET = False # Supresses output

class NullGraphics:
    """Auto-generated class docstring for NullGraphics."""
    def initialize(self, state, isBlue = False):
        """Auto-generated function docstring for initialize."""
        pass

    def update(self, state):
        """Auto-generated function docstring for update."""
        pass

    def checkNullDisplay(self):
        """Auto-generated function docstring for checkNullDisplay."""
        return True

    def pause(self):
        """Auto-generated function docstring for pause."""
        time.sleep(SLEEP_TIME)

    def draw(self, state):
        """Auto-generated function docstring for draw."""
        print(state)

    def updateDistributions(self, dist):
        """Auto-generated function docstring for updateDistributions."""
        pass

    def finish(self):
        """Auto-generated function docstring for finish."""
        pass

class PacmanGraphics:
    """Auto-generated class docstring for PacmanGraphics."""
    def __init__(self, speed=None):
        """Auto-generated function docstring for __init__."""
        if speed != None:
            global SLEEP_TIME
            SLEEP_TIME = speed

    def initialize(self, state, isBlue = False):
        """Auto-generated function docstring for initialize."""
        self.draw(state)
        self.pause()
        self.turn = 0
        self.agentCounter = 0

    def update(self, state):
        """Auto-generated function docstring for update."""
        numAgents = len(state.agentStates)
        self.agentCounter = (self.agentCounter + 1) % numAgents
        if self.agentCounter == 0:
            self.turn += 1
            if DISPLAY_MOVES:
                ghosts = [pacman.nearestPoint(state.getGhostPosition(i)) for i in range(1, numAgents)]
                print("%4d) P: %-8s | Score: %-5d | Ghosts: %s" % (
                    self.turn,
                    str(pacman.nearestPoint(state.getPacmanPosition())),
                    state.score,
                    ghosts,
                ))
            if self.turn % DRAW_EVERY == 0:
                self.draw(state)
                self.pause()
        if state._win or state._lose:
            self.draw(state)

    def pause(self):
        """Auto-generated function docstring for pause."""
        time.sleep(SLEEP_TIME)

    def draw(self, state):
        """Auto-generated function docstring for draw."""
        print(state)

    def finish(self):
        """Auto-generated function docstring for finish."""
        pass

