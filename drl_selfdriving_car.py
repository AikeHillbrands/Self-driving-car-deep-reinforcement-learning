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
games_episode=10
episodes = 5
save_model = False

'''A simple model is built'''
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(64,input_dim=env.observation_space[0], activation = "relu"))
    #model.add(keras.layers.Dense(64,activation = "relu"))
    model.add(keras.layers.Dense(env.action_space[0],activation = "tanh"))
    model.compile(loss="mse",optimizer=keras.optimizers.Adam())
    return model

'''Trains the model on the data'''
def train_model(model,data,epochs = epochs):
    x = data[0]
    y = data[1]
    model.fit(x=x,y=y,batch_size = batch_size,epochs = epochs,verbose=1)

'''A random action is created'''
def random_action():
    return (rnd.random()*2-1,rnd.random()*2-1)

'''A game is played and the rewards, states and actions are returned so that it can be used for testing and training'''
def play_episode(model,render = False,rand=0):
    states, actions, rewards = [],[],[]
    state = env.reset()
    for i in range(max_steps):
        action = []
        if rnd.random()<rand: #to encourage the agent to explore in the early episodes a lot of the actions are taken randomly
            action = random_action()
        else:
            action = model.predict(np.array(state).reshape(1,env.observation_space[0]))[0]
        states.append(state)
        actions.append(action)
        state,done,reward = env.action(action,render=render)
        reward-=0.001 #punishment for doing nothing
        rewards.append(reward)
        if done:
            break
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    return states,actions,rewards

'''The main methode of the whole programm. It executes the game and collects training data for the algorithm and trains the model'''     
def q_learning(model):
    reward_history = [] #The reward sums are saved over the training to visualize the progress
    for k in range(episodes): #Each episode is a series of games to collect the training data
        print("Episode",k,"/",episodes)
        first_run=True
        episode_states, episode_actions = None,None
        for i in range(games_episode): #This loop collects the data of an individual game and processes the data
            states,actions,rewards = play_episode(model,rand=1-(k/episodes))
            reward_history.append(np.sum(rewards))
            rewards = discount_rewards(rewards) #Rewards are discounted
            actions = calculate_new_action_values(actions,rewards) #The actions used for the training are calculated
            states = states.reshape(states.shape[:2])
            if first_run:
                first_run = False
                episode_states = states
                episode_actions = actions
            else:
                episode_states = np.vstack((episode_states,states)) #The data is stacked to the collection of the whole episode for the training
                episode_actions = np.vstack((episode_actions,actions))
        train_model(model,(episode_states,episode_actions))

        state,done,reward = play_episode(model,render=True) #A demo game is shown with the discounted rewards afterwards
        plt.plot(discount_rewards(reward))
        plt.show()
    plt.plot(reward_history) #The trainig progress can be seen due to the summed rewards of every game
    plt.show()
    

'''This method computes the actions the agent is trained on.
When an action brought a negativ reward, the opposite is trained'''
def calculate_new_action_values(actions,rewards): 
    new_actions = actions[:]
    rew = rewards.reshape((rewards.shape[0],1))
    new_actions = np.multiply(new_actions,rew)
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
    if save_model:
        model.save("model.h5")

'''When you want to test the saved model, you can use this method and adjust how many games you want to see'''
def test_model(games = 5):
    model = load_model('model.h5')
    for i in range(games):
        play_episode(model,render=True)


'''Select whatever you want to do. When you want to save a new model, you need to set save_model to True'''
#create_and_train()
test_model()
