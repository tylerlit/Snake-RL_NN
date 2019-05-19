# Tyler Little CMSC 478 - Machine Learning
# Final Project
#
# this file contains the Snake game class

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL._bytes import as_8_bit # reads ascii keystrokes as octal
from random import randint 
import numpy as np
import random
import copy

class Game:
	def __init__(self, gui = False, rl = True):

		# render game
		self.gui = gui
		# use reinforcement learning to play game / collect data
		self.rl = rl

		#constants for rendering
		self.windowsize = 250 # in pixels
		self.interval = 100 # milliseconds
		self.playing = True

		# height and width of playing board
		self.n = 25

		self.score = 0
		self.maxscore = 0

		if self.rl:

			# set parameters
			# learning rate and discount factor
			self.lr = .001
			self.df = .25

			# this is the maximum steps needed to reach food
			self.OPTsteps = 8 + (4 * 3 * self.n ) + (((self.n * self.n) - (self.n * 4)) * 4)

			# keep set of moves to food when q learning is finished
			self.moves = []

		self.snake = [(self.n // 2, self.n // 2)]
		self.snake_dir = (1, 0)

		# spawn random food position
		self.food = []
		x, y = randint(0, self.n-1), randint(0, self.n-1)
		self.food.append((x, y))

		self.playgame()
	
	def move(self, value):

		print(self.food)
		if self.rl:
			if self.moves == []:
				# initialize virtual grid
				grid = [[-.1 for x in range(self.n)] for y in range(self.n)]
				# create q table with set of actions for each individual state and set each value to 0
				a = 4 # 0 = up, 1 = down, 2 = left, and 3 = right
				table = [[[0 for z in range(a)] for x in range(self.n)] for y in range(self.n)]

				step = 0

				# save old snake
				oldsnake = copy.deepcopy(self.snake)

				# give food good reward give snake head bad reward
				x = [x for x in self.snake[0]]
				(fx, fy) = self.food[0][0], self.food[0][1]
				grid[fx][fy] = 1000
				# grid[x[0]][x[1]] = -10

				print("gathering test data...")

				while(True):

					# need to save status of snake
					dead = False

					#
					## START Q LEARNING
					#

					# save previous agent location for q update
					oldx = copy.deepcopy(x)

					# observe q table scores for all actions in current state
					scores = []
					for i in table[x[0]][x[1]]:
						scores.append(i)

					# get highest scoring move (pick randomly if there are multiple maxes)
					#
					m = max(scores)
					choose = [i for i, j in enumerate(scores) if j == m]
					direction = random.choice(choose)

					# traverse in that direction if the snake can live
					# up
					if (direction == 0):
						self.snake_dir = (0, 1)

					# down
					if (direction == 1):
						self.snake_dir = (0, -1)

					# left
					if (direction == 2):
						self.snake_dir = (-1, 0)

					# right
					if (direction == 3):
						self.snake_dir = (0, 1)

					self.moves.append(self.snake_dir)
					head = self.snake[0]
					newpos = (self.snake_dir[0] + head[0], self.snake_dir[1] + head[1])

					self.snake.insert(0, newpos)

					# check if snake lived and update move list and reset if it didnt
					(hx, hy) = self.snake[0][0], self.snake[0][1]
					if hx < 0 or hx >= self.n or hy < 0 or hy >= self.n:
						dead = True
						print("DIED!!!!! at")
						print(self.snake)
						self.snake.pop(0)
					elif (len(self.snake) != len(set(self.snake))) and len(self.moves) > 1:
						dead = True
						print("DIED!!!!! at")
						print(self.snake)
						self.snake.pop(0)
					else:
						self.snake.pop()

					# get new head and find values of state
					x = [x for x in self.snake[0]]

					# get highest scoring move in new state
					scores = []
					for i in table[x[0]][x[1]]:
						scores.append(i)

					m = max(scores)
					choose = [i for i, j in enumerate(scores) if j == m]
					qprime = random.choice(choose)

					 
					if dead:
						reward -= 1
					else:
						reward = grid[oldx[0]][oldx[1]]
						# table[oldx[0]][oldx[1]][direction] -= 1
					# update q table
					table[oldx[0]][oldx[1]][direction] += (self.lr * ((reward + (self.df * scores[qprime])) - table[oldx[0]][oldx[1]][direction]))

					##
					## END Q LEARNING
					##

					step += 1

					print(snake)
					if dead:
						# self.moves.pop()
						# step -= 1
						self.moves.clear()
						self.snake = copy.deepcopy(oldsnake)
						step = 0
					# check if agent is finished
					if x == [fx, fy]:
						if step <= self.OPTsteps:
							self.moves.pop()
							print("done")
							print(self.score)
							self.snake = copy.deepcopy(oldsnake)
							break
						self.moves.clear()
						self.snake = copy.deepcopy(oldsnake)
						step = 0
			else:
				self.snake_dir = self.moves[0]
				self.moves.pop(0)

		# get new position of snake head
		head = self.snake[0]
		newpos = (self.snake_dir[0] + head[0], self.snake_dir[1] + head[1])

		self.snake.insert(0, newpos)
		self.snake.pop()

		# get new new position of snake head
		(hx, hy) = self.snake[0][0], self.snake[0][1]

		# check if snake collided with wall or itself
		if hx < 0 or hx >= self.n or hy < 0 or hy >= self.n or \
		(len(self.snake) != len(set(self.snake))):
			self.snake.clear()
			self.food.clear()
			self.playing = False
			return

		# check if snake found food
		(fx, fy) = self.food[0][0], self.food[0][1]

		if hx == fx and hy == fy:
			self.snake.append((fx, fy))
			self.food.remove((fx, fy))
			x, y = randint(0, self.n-1), randint(0, self.n-1)
			self.food.append((x, y))
			self.score += 1

		if self.gui:
			glutTimerFunc(self.interval, self.move, 0) # trigger next update


	#################################################################
	## FUNCTIONS BELOW ONLY DEAL WITH RENDERING/PLAYING THE GAME!! ##
	#################################################################

	def draw_snake(self):
		glColor3f(1.0, 1.0, 1.0) # set color to white
		for x in self.snake:
			self.draw_rect(x[0], x[1], 1, 1) # draw (x, y) with width=1 and height=1

	def draw_food(self):
		glColor3f(0.5, 0.5, 1.0) # set color to blue
		for x in self.food:
			self.draw_rect(x[0], x[1], 1, 1) # draw (x, y) with width=1 and height=1

	def refresh2d_custom(self, width, height, internal_width, internal_height):
		glViewport(0, 0, width, height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0.0, internal_width, 0.0, internal_height, 0.0, 1.0)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

	def draw_rect(self, x, y, width, height):
		glBegin(GL_QUADS) # start drawing a rectangle
		glVertex2f(x, y) # bottom left point
		glVertex2f(x + width, y) # bottom right point
		glVertex2f(x + width, y + height) # top right point
		glVertex2f(x, y + height) # top left point
		glEnd()	

	def draw(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # clear the screen
		glLoadIdentity() # reset position
		self.refresh2d_custom(self.windowsize, self.windowsize, self.n, self.n)
		self.draw_food()
		self.draw_snake()
		glutSwapBuffers() 

	def playgame(self):

		if self.gui:
			glutInit()
			glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
			glutInitWindowSize(self.windowsize, self.windowsize)
			glutInitWindowPosition(0, 0)
			window = glutCreateWindow("Snake ML Project")
			glutDisplayFunc(self.draw) # set draw function callback
			glutIdleFunc(self.draw) # draw all the time
			glutTimerFunc(self.interval, self.move, 0) # trigger next update every 100ms
			if not self.rl:		
				glutKeyboardFunc(self.keyboard) # tell opengl that we want to check keys
			glutMainLoop() # start everything


	def keyboard(self, *args):
		w_key = as_8_bit('\167')
		a_key = as_8_bit('\141')
		s_key = as_8_bit('\163')
		d_key = as_8_bit('\144')
		arg_trans = args[0]
		if args[0] == w_key :
			self.snake_dir = (0, 1) # up
		if args[0] == s_key:
			self.snake_dir = (0, -1) # down
		if args[0] == a_key:
			self.snake_dir = (-1, 0) # left
		if args[0] == d_key:
			self.snake_dir = (1, 0) # right

if __name__ == "__main__":
    game = Game(gui = True, rl = True)
