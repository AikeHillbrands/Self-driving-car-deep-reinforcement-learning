import numpy as np
import math
import matplotlib.pyplot as plt
import Engine
import random as rnd

view_distance = 100
view_rays = 5
hit_wall = -0.5
hit_reward = 2
finished_level = 5

class env:
    def __init__(self):
        self.engine = None
        self.reset()
        self.observation_space = (view_rays,1)
        self.action_space = 4 #0: do nothing 1:accelerate 2: break 3: right 4: left

    #inits a new world
    def new_world(self,wall = np.array(((2,7),(7,6),(11,8),(15,8),(18,12),(22,12),(18,5),(11,5),(5,3),(2,3),(2,7)),dtype="float32"), rewards = np.array(((3,5),(8,5),(13,6),(18,8),(20,11)),dtype="float32")):
        self.wall = wall
        self.reward_gates = np.array(((3,5),(5,5),(8,5),(11,7),(13,6),(18,8),(20,11)),dtype="float32")

    #inits a new agent
    def new_agent(self):
        self.agent_dir = env.rotate(env.normalize(self.reward_gates[-2]-self.reward_gates[-1]),(rnd.random()-0.5)*0.3)
        self.agent_pos = self.reward_gates[-1,:]
        self.agent_speed = 0
        self.agent_rewards = [p for p in self.reward_gates]

    def reset(self):
        if self.engine != None:
            self.engine.close()
            self.engine = None
        self.new_world()
        self.new_agent()
        view = self.calculate_view()[0]
        return view

    @staticmethod
    def normalize(v): #makes a vector 1 long
        s = 0
        for a in v:
            s += a*a
        return np.true_divide(v[:], math.sqrt(s))

    #returns vectors which point left and right around the direction
    def view_vectors(self):
        results = []
        for i in range(view_rays):
            results.append(env.rotate(self.agent_dir, (math.pi/(view_rays-1)*i) - (math.pi/2) ))
        return np.array(results)

    #rotates the vector v for a radians
    @staticmethod
    def rotate(v,a):
        return np.array((v[0]*math.cos(a)-v[1]*math.sin(a),v[1]*math.cos(a)+v[0]*math.sin(a)))

    #returns the distance between two points
    @staticmethod
    def dist(p1,p2):
        p = np.subtract(p1,p2)
        return(np.linalg.norm(p))

    #moves the agent and returns the new view 
    def action(self,act,render = False):
        maxarg = np.argmax(act)
        if maxarg == 0:
            self.agent_speed+=0.01
        elif maxarg == 1:
            self.agent_speed-=0.01
        elif maxarg == 2:
            self.agent_dir = env.rotate(self.agent_dir,-0.1)
        elif maxarg == 3:
            self.agent_dir = env.rotate(self.agent_dir,0.1)
        
        #speed cap
        if self.agent_speed>0.1:
            self.agent_speed = 0.1
        if self.agent_speed<-0.01:
            self.agent_speed = -0.01

        self.agent_pos += np.multiply(self.agent_dir,self.agent_speed)

        reward = 0
        #checks if there is a reward gate in the radius of 2 and then gives a reward to the agent and removes the reward gate for further steps
        for p in self.agent_rewards:
            if env.dist (self.agent_pos,p) < 2:
                reward += hit_reward
                for i in range(len(self.agent_rewards)):
                    if(np.array_equal(p,self.agent_rewards[i])):
                        self.agent_rewards.pop(i)
                        break
    

        #when render is True the enviorement is rendered
        if render:
            if self.engine == None:
                self.engine = Engine.Engine()
            self.engine.new_frame()
            self.engine.draw_lines(self.wall)
            self.engine.draw_car(self.agent_pos,self.agent_dir)
            self.engine.render()
            self.engine.wait(1000/60)

        result = self.calculate_view()
        result.append(reward)
        if result[1] == True: #the agent crashed and is done therefore
            result[2] += hit_wall
        if len(self.agent_rewards)==0: #the agent completed the level and gets an extra reward and is done
            result[2] += finished_level
            result[1]= True
        return result #[observation, done, reward]

    #calculates the distances from each view ray to the next wall
    def calculate_view(self):
        rays = self.view_vectors()
        view = []
        crashed = False #the vision rays are also used to check if the distance to a wall is to small
        for ray in rays:
            #The shortest distance from each view ray the walls is calculated and the shortest one is saved
            shortest_dist = view_distance
            for i in range(len(self.wall)):
                w1 = self.wall[i-1]
                w2 = self.wall[i]
                p = self.agent_pos
                n = ray
                div = w1[0]*n[1] - w1[1]*n[0] - w2[0]*n[1] + w2[1]*n[0]
                k = (w1[0] * n[1]) - (w1[1] *n[0]) + (n[0] * p[1]) - (n[1] * p[0])
                if not env.is_paralel(w2,w1,n):
                    if 0 <= k / (div) and k / (div)<=1:
                        dist = (w1[0]*(w2[1]-p[1]) - w1[1]*(w2[0]-p[0]) + w2[0] * p[1] - w2[1] * p[0] ) / (div)
                        if dist < shortest_dist and dist > 0:
                            shortest_dist = dist
                        if abs(dist)<0.2: #when the distance to a wall is smaller than 0.2 the env returns that the car crashed
                            crashed = True

            view.append(shortest_dist)
        return [np.array(view).reshape(view_rays,1),crashed]
    
    @staticmethod
    def is_paralel(w1,w2,n): #checks if the line betwen w1 and w2 is paralel
        if abs(w1[0]*n[1] - w1[1]*n[0] - w2[0]*n[1] + w2[1]*n[0])< 0.000001:
            return True
        return False

#Testing
def test():
    e = env(np.array(((3,7),(7,6),(11,8),(15,8),(18,12),(22,12),(18,5),(11,5),(5,3),(2,3)),dtype="float32"),np.array(((3,5),(8,5),(13,6),(18,8),(20,11)),dtype="float32"))

    running = True
    while running:
        observation = e.action(0.01,0.01,render=True)
        print(observation[1:])
        running = not observation[1]

#test()


