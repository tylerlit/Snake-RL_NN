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
import time
import csv
import net

class Game:
	def __init__(self, gui = False, rl = True):
		# render game
		self.gui = gui
		# use reinforcement learning to play game / collect data
		self.rl = rl

		nn = net.Net()
		self.model = nn.getmodel()

		#constants for rendering
		self.windowsize = 500 # in pixels
		self.interval = 100 # milliseconds
		self.playing = True

		# height and width of playing board
		self.n = 10

		self.score = 0

		# set parameters
		# learning rate and discount factor
		self.lr = .5
		self.df = .5

		# keep set of moves to food when q learning is finished
		self.moves = []

		self.snake = [(self.n // 2, self.n // 2)]
		self.snake_dir = (1, 0)

		# spawn random food position
		self.food = []
		x, y = randint(0, self.n-1), randint(0, self.n-1)
		self.food.append((x, y))

		# this is the maximum steps needed to reach food
		(hx, hy) = self.snake[0][0], self.snake[0][1]
		(fx, fy) = self.food[0][0], self.food[0][1]
		self.OPTsteps = abs(hx - fx) + abs(hy - fy)

		self.playgame()
	
	# get information about game to create input for network
	def getstuff(self):

		up_wall = (self.n - 1) - self.snake[0][1]
		up_food = self.food[0][1] - self.snake[0][1]
		counter = 0
		for i in self.snake:
			if self.snake[0][1] == i[1]:
				counter += 1
				if counter == 2:
					up_snake = i[1] - self.snake[0][1]
		if counter == 1:
			up_snake = 0

		left_wall = self.snake[0][0]
		left_food = self.snake[0][0] - self.food[0][0]
		counter = 0
		for i in self.snake:
			if self.snake[0][0] == i[0]:
				counter += 1
				if counter == 2:
					left_snake = self.snake[0][0] - i[0]
		if counter == 1:
			left_snake = 0

		down_wall = self.snake[0][1]
		down_food = self.snake[0][1] - self.food[0][1]
		counter = 0
		for i in self.snake:
			if self.snake[0][1] == i[1]:
				counter += 1
				if counter == 2:
					down_snake = self.snake[0][1] - i[1]
		if counter == 1:
			down_snake = 0

		right_wall = (self.n - 1) - self.snake[0][0]
		right_food = self.food[0][0] - self.snake[0][0]
		counter = 0
		for i in self.snake:
			if self.snake[0][0] == i[0]:
				counter += 1
				if counter == 2:
					right_snake = i[0] - self.snake[0][0]
		if counter == 1:
			right_snake = 0

		x = [up_wall, up_food, up_snake, \
					left_wall, left_food, left_snake, down_wall, \
					down_food, down_snake, right_wall, right_food, right_snake]
		return x
		# try:
		# 	with open("data2.csv", 'a') as f:
		# 		write = csv.writer(f, delimiter = ',')
		# 		write.writerow([up_wall, up_food, up_snake, \
		# 			left_wall, left_food, left_snake, down_wall, \
		# 			down_food, down_snake, right_wall, right_food, right_snake])
		# 	with open("labels2.csv", 'a') as f:
		# 		write = csv.writer(f, delimiter = ',')
		# 		write.writerow(self.moves[0])

		# except FileNotFoundError:
		# 	print("cant find data")

	def move(self, value):

		if self.rl:
			if self.moves == []:

				# test time of RL
				starttime = time.time()

				# collect data to analyze performance of reinforcement learning
				# steps = []

				# initialize virtual grid
				grid = [[5 for x in range(self.n)] for y in range(self.n)]
				# create q table with set of actions for each individual state and set each value to 0
				a = 4 # 0 = up, 1 = down, 2 = left, and 3 = right
				table = [[[0 for z in range(a)] for x in range(self.n)] for y in range(self.n)]

				# save old snake
				oldsnake = copy.deepcopy(self.snake)

				# give food good reward
				(fx, fy) = self.food[0][0], self.food[0][1]
				grid[fx][fy] = 100

				step = 0

				print("gathering test data...")

				while(True):
					# print("fud")
					# print(self.food)
					# need to save status of snake
					dead = False
					moved = True

					#
					## START Q LEARNING
					#

					# save previous agent head location for q update
					x = [x for x in self.snake[0]]
					oldx = copy.deepcopy(x)

					# observe q table scores for all actions in current state
					scores = []
					for i in table[x[0]][x[1]]:
						scores.append(i)

					# get highest scoring move (pick randomly if there are multiple maxes)
					#
					m = max(scores)
					choose = [i for i, j in enumerate(scores) if j == m]
					olddirection = self.snake_dir
					direction = random.choice(choose)

					# traverse in that direction
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
						self.snake_dir = (1, 0)

					# move in that direction 
					head = self.snake[0]
					newpos = (self.snake_dir[0] + head[0], self.snake_dir[1] + head[1])

					# see if snake lives
					self.snake.insert(0, newpos)
					(hx, hy) = self.snake[0][0], self.snake[0][1]
					if hx < 0 or hx >= self.n or hy < 0 or hy >= self.n or \
					(len(self.snake) != len(set(self.snake))):
						moved = False
						self.snake.pop(0)
					else:
						self.snake.pop()

					# add to list of moves if it does
					if moved:
						self.moves.append(self.snake_dir)
					
					# get new head and find values of state
					x = [x for x in self.snake[0]]

					# get highest scoring move in new state
					scores = []
					for i in table[x[0]][x[1]]:
						scores.append(i)

					m = max(scores)
					choose = [i for i, j in enumerate(scores) if j == m]
					qprime = random.choice(choose)

					reward = grid[oldx[0]][oldx[1]]

					if grid[oldx[0]][oldx[1]] > 0:
						grid[oldx[0]][oldx[1]] = -1
					else:	
						grid[oldx[0]][oldx[1]] -= 1

					# table[oldx[0]][oldx[1]][direction] -= 1

					# update q table
					table[oldx[0]][oldx[1]][direction] += \
					(self.lr * ((grid[oldx[0]][oldx[1]] + (self.df * scores[qprime])) - \
					table[oldx[0]][oldx[1]][direction]))

					##
					## END Q LEARNING
					##

					step += 1

					# print("snek = ")
					# print(self.snake)
					# if (moved):
					# 	starttime = time.time()
					if (time.time() - starttime) > 7:
						exit()
					# check if agent is finished

					if [x for x in self.snake[0]] == [fx, fy]:
						
						# for testing RL
						# steps.append(step)

						# print("OMG!!!!")
						# print("FOUND FOOD!!!!")
						# print("OMG!!!!")

						if step <= self.OPTsteps:

							# for testing RL
							# print(steps)
							
							# print(self.OPTsteps)

							# exit()

							self.moves.pop()
							print("done")
							print(self.score)
							self.snake = copy.deepcopy(oldsnake)
							break

						self.moves.clear()
						self.snake = copy.deepcopy(oldsnake)
						step = 0
			else:

				# get stuff about game
				# more specifically, make inputs and labels and export to files
				# self.getstuff()

				# print("snek = ")
				# print(self.snake[0])
				# print("food = ")
				# print(self.food)
				# print(self.moves[0])

				# use moves learned from q learning
				self.snake_dir = self.moves[0]
				self.moves.pop(0)

		# END RL 
		# AGENT KNOWS WHAT TO DO AFTER THIS POINT!!

		# OR DOES IT???
		# CALL NN MODEL TO PREDICT BASED OFF STUFF FROM GAME
		stuff = self.getstuff()
		pred = self.model.predict([stuff])
		pred = pred[0]
		move = [int(x) for x in pred]
		if move == [1,0,0,0]:
			self.snake_dir = (0, 1)
		elif move == [0,1,0,0]:
			self.snake_dir = (0, -1)
		elif move == [0,0,1,0]:
			self.snake_dir = (-1, 0)
		elif move == [0,0,0,1]:
			self.snake_dir = (1, 0)
		else:
			self.snake_dir = (0,0)

		print(move)

		# get new position of snake head
		head = self.snake[0]
		newpos = (self.snake_dir[0] + head[0], self.snake_dir[1] + head[1])

		self.snake.insert(0, newpos)
		self.snake.pop()

		# print(self.snake)

		# get new new position of snake head
		(hx, hy) = self.snake[0][0], self.snake[0][1]

		# check if snake collided with wall or itself
		if hx < 0 or hx >= self.n or hy < 0 or hy >= self.n or \
		(len(self.snake) != len(set(self.snake))):
			print("ahhh died snek =")
			print(self.snake)
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


	############################################################################
	## FUNCTIONS BELOW ONLY DEAL WITH RENDERING/PHYSICALLY PLAYING THE GAME!! ##
	############################################################################

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
    game = Game(gui = True, rl = False)
