import numpy as np
import keras
import Enviorement
import matplotlib.pyplot as plt
from keras.models import load_model
import random as rnd

'''
The whole program was created to learn something about deep reinforcement learning. I used an algorithm similar to the q-learning algorithm.
To render the environment pygame is required.
'''

env = Enviorement.env()
batch_size = 500
max_steps = 250
epochs = 3
games_episode=20
episodes = 20
save_model = False

'''A simple model is built'''
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(64,input_dim=env.observation_space[0], activation = "relu"))
    model.add(keras.layers.Dense(64,activation = "relu"))
    model.add(keras.layers.Dense(env.action_space,activation = "relu"))
    model.compile(loss="mse",optimizer=keras.optimizers.Adam(lr=0.0003)) #It turned out that a very small LR is the most productive because it doesnt overfit to the discounted actions
    return model

'''Trains the model on the data'''
def train_model(model,data,epochs = epochs):
    x = data[0]
    y = data[1]
    model.fit(x=x,y=y,batch_size = batch_size,epochs = epochs,verbose=1)

'''A random action is created'''
def random_action():
    return np.random.rand(env.action_space)

'''A game is played and the rewards, states and actions are returned so that it can be used for testing and training'''
def play_episode(model,render = False,rand=0):
    states, actions, rewards = [],[],[]
    state = env.reset()
    speed = None
    for i in range(max_steps):
        action = []
        if rnd.random()<rand: #to encourage the agent to explore. In the early episodes a lot of the actions are taken randomly
            action = random_action()
        else:
            action = model.predict(np.array(state).reshape(1,env.observation_space[0]))[0]
        states.append(state)

        actions.append(action)
        state,done,reward = env.action(action,render=render)

        #When the agents average speed over the last time steps is below 0.02 the game is stopped because the agent seems to be stuck
        if speed == None:
            speed = env.agent_speed
        else:
            speed = speed*0.8+env.agent_speed*0.2 #average for the last steps is calculated
        if speed < 0.02 and i > max_steps*0.1: #that the agent has time to start, the first 10% of the game are ignored for stopping
            done = True

        rewards.append(reward)
        if done:
            break
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    rewards[0]=0
    return states,actions,rewards

'''The main methode of the whole programm. It executes the game and collects training data for the algorithm and trains the model'''     
def q_learning(model):
    reward_history = [] #The reward sums are saved over the training to visualize the progress
    for k in range(episodes): #Each episode is a series of games to collect the training data
        print("Episode",k,"/",episodes)
        first_run=True
        episode_states, episode_actions = None,None
        for i in range(games_episode): #This loop collects the data of an individual game and processes the data
            states,actions,rewards = play_episode(model,rand=1-0.7*(k/episodes))
            reward_history.append(np.sum(rewards)) #just for a overview in the end
            discounted_rewards = discount_rewards(rewards) #Rewards are discounted
            actions = calculate_new_action_values(actions,discounted_rewards) #The actions used for the training are calculated

            states = states.reshape(states.shape[:2])

            if first_run:
                first_run = False
                episode_states = states
                episode_actions = actions
            else:
                episode_states = np.vstack((episode_states,states)) #The data is stacked to the collection of the whole episode for the training
                episode_actions = np.vstack((episode_actions,actions))
        train_model(model,(episode_states,episode_actions))

        state,actions,reward = play_episode(model,render=True) #A demo game is shown with the discounted rewards afterwards
        reward = discount_rewards(reward)
        actions = calculate_new_action_values(actions,reward)

    plt.plot(reward_history) #The trainig progress can be seen due to the summed rewards of every game
    plt.show()
    

'''This method computes the actions the agent is trained on.
When an action brought a negativ reward, the action with the highest value is discouraged and all the other actions are encouraged.
When the action has a positive reward all the possible actions are discouraged but the action with the highest value is encouraged'''
def calculate_new_action_values(actions,rewards): 
    new_actions = actions[:]
    rew = rewards.reshape((rewards.shape[0],1))
    for i in range(new_actions.shape[0]):
        new_actions[i]-=rew[i]*0.5
        new_actions[i,new_actions[i].argmax()]+=1*rew[i]
    return new_actions

'''when the agent receives a reward, the previous steps are also reinforced'''
def discount_rewards(rewards):
    result = rewards[:]
    for i in reversed(range(result.shape[0]-1)):
        result[i] += result[i+1]*0.99
    return result

'''Creates the model and trains it'''
def create_and_train(): 
    model = build_model()
    q_learning(model)

    print("Save model? y/n")
    if input() == "y":
        model.save("model.h5")
        print("Model saved!")
    else:
        print("Model NOT saved!")

'''When you want to test the saved model, you can use this method and adjust how many games you want to see'''
def test_model(games = 5):
    model = load_model('model.h5')
    
    for i in range(games):
        play_episode(model,render=True,rand=0.1)


'''Select whatever you want to do. When you want to save a new model, you need to set save_model to True'''
#create_and_train()
test_model()

