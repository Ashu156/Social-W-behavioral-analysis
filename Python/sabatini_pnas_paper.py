#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:27:54 2024

@author: rig5
"""

#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
# path = '/home/rig5/2ABT_behavior_models-main'
# import plot_models_v_mouse as bp
# import model_policies as models
from sklearn.model_selection import train_test_split

#%% conditional_probs.py



import numpy as np
import pandas as pd


def list_to_str(seq):
    
    '''take list of ints/floats and convert to string'''
    
    seq = [str(el) for el in seq] # convert element of sequence to string
    
    return ''.join(seq) # flatten list to single string

#%%

def encode_as_ab(row, symm):
    
    '''
    converts choice/outcome history to character code where where letter represents choice and case outcome
    INPUTS:
        - row: row from pandas DataFrame containing named variables 'decision_seq' and 'reward_seq' (previous N decisions/rewards) 
        - symm (boolean): if True, symmetrical encoding with A/B for direction (A=first choice in sequence)
                          if False, R/L encoding right/left choice
    OUTPUTS:
        - (string): string of len(decision_seq) trials encoding each choice/outcome combination per trial
    
    '''
    
    if int(row.decision_seq[0]) & symm: # symmetrical mapping based on first choice in sequence 1 --> A
        mapping = {('0','0'): 'b', ('0','1'): 'B', ('1','0'): 'a', ('1','1'): 'A'} 
    elif (int(row.decision_seq[0])==0) & symm: # symmetrical mapping for first choice 0 --> A    
        mapping = {('0','0'): 'a', ('0','1'): 'A', ('1','0'): 'b', ('1','1'): 'B'} 
    else: # raw right/left mapping (not symmetrical)
        mapping = {('0','0'): 'u', ('0','1'): 'U', ('1','0'): 'm', ('1','1'): 'M'} 

    return ''.join([mapping[(c,r)] for c,r in zip(row.decision_seq, row.reward_seq)])

#%%

def add_history_cols(df, N):
    
    '''
    INPUTS:
        - df (pandas DataFrame): behavior dataset
        - N (int): number trials prior to to previous trial to sequence (history_length)
        
    OUTPUTS:
        - df (pandas DataFrame): add columns:
            - 'decision_seq': each row contains string of previous decisions t-N, t-N+1,..., t-1
            - 'reward_seq': as in decision_seq, for reward history
            - 'history': encoded choice/outcome combination (symmetrical)
            - 'RL_history': encoded choice/outcome combination (raw right/left directionality)
       
    '''
    from numpy.lib.stride_tricks import sliding_window_view
    
    df['decision_seq']=np.nan # initialize column for decision history (current trial excluded)
    df['reward_seq']=np.nan # initialize column for laser stim history (current trial excluded)

    df = df.reset_index(drop=True) # need unique row indices (likely no change)

    for session in df.Session.unique(): # go by session to keep boundaries clean

        d = df.loc[df.Session == session] # temporary subset of dataset for session
        df.loc[d.index.values[N:], 'decision_seq'] = \
                                    list(map(list_to_str, sliding_window_view(d.Decision.astype('int'), N)))[:-1]

        df.loc[d.index.values[N:], 'reward_seq'] = \
                                    list(map(list_to_str, sliding_window_view(d.Reward.astype('int'), N)))[:-1]

        df.loc[d.index.values[N:], 'history'] = \
                                    df.loc[d.index.values[N:]].apply(encode_as_ab, args=([True]), axis=1)

        df.loc[d.index.values[N:], 'RL_history'] = \
                                    df.loc[d.index.values[N:]].apply(encode_as_ab, args=([False]), axis=1)
        
    return df
        

#%%

def calc_conditional_probs(df, symm, action=['Switch'], run=0):

    '''
    calculate probabilities of behavior conditional on unique history combinations
    
    Inputs:
        df (pandas DataFrame): behavior dataset
        symm (boolean): use symmetrical history (True) or raw right/left history (False)
        action (string): behavior for which to compute conditional probabilities (should be column name in df)
        
    OUTPUTS:
        conditional_probs (pandas DataFrame): P(action | history) and binomial error, each row for given history sequence
    '''

    group = 'history' if symm else 'RL_history' # define columns for groupby function

    max_runs = len(action) - 1 # run recursively to build df that contains summary for all actions listed

    conditional_probs = df.groupby(group).agg(
        paction=pd.NamedAgg(action[run], np.mean),
        n = pd.NamedAgg(action[run], len),
    ).reset_index()
    conditional_probs[f'p{action[run].lower()}_err'] = np.sqrt((conditional_probs.paction * (1 - conditional_probs.paction))
                                                  / conditional_probs.n) # binomial error
    conditional_probs.rename(columns={'paction': f'p{action[run].lower()}'}, inplace=True) # specific column name
    
    if not symm:
        conditional_probs.rename(columns={'RL_history':'history'}, inplace=True) # consistent naming for history
    
    if max_runs == run:
    
        return conditional_probs
    
    else:
        run += 1
        return pd.merge(calc_conditional_probs(df, symm, action, run), conditional_probs.drop(columns='n'), on='history')


#%%
def sort_cprobs(conditional_probs, sorted_histories):
    
    '''
    sort conditional probs by reference order for history sequences to use for plotting/comparison
    
    INPUTS:
        - conditional_probs (pandas DataFrame): from calc_conditional_probs
        - sorted_histories (list): ordered history sequences from reference conditional_probs dataframe
    OUTPUTS:
        - (pandas DataFrame): conditional_probs sorted by reference history order
    '''
    
    from pandas.api.types import CategoricalDtype
    
    cat_history_order = CategoricalDtype(sorted_histories, ordered=True) # make reference history ordinal
    
    conditional_probs['history'] = conditional_probs['history'].astype(cat_history_order) # apply reference ordinal values to new df
    
    return conditional_probs.sort_values('history') # sort by reference ordinal values for history



#%% model_fitting.py




import jax.numpy as jnp
from jax import jit, lax, value_and_grad
from tqdm.auto import trange
import numpy as np


def fit_with_sgd(ll_func, training_data, num_steps=10000, step_size=1e-1, init_parameters = (1.0, 1.0, 1.0)):
    
    '''
    fit behavior model with sgd, basic architecture
    
    INPUTS:
        - ll_func (function): log likelihood function for given model
        - traning_data (nested lists): [choices, rewards] by session 
        - num_steps (int)
        - step_size (float)
        - init_parameters (tuple): starting parameters, varies in length by model
        
    OUTPUTS:
        - (np array) optimized parameters
        - nll: negative log likelihood
    '''
    
    # simple gradient ascent algorithm
    from jax.example_libraries.optimizers import sgd

    init_fun, update_fun, get_params = sgd(step_size)
    opt_state = init_fun(init_parameters)

    loss_fn = lambda parameters: -ll_func(parameters, training_data)
    loss_fn_and_grad = jit(value_and_grad(loss_fn))

    def step(itr, opt_state):
        value, grads = loss_fn_and_grad(get_params(opt_state))
        opt_state = update_fun(itr, grads, opt_state)
        return value, opt_state

    for i in trange(num_steps, disable=True):
        nll, opt_state = step(step, opt_state)
        if i % int(round(num_steps/4)) == 0:
            print("iteration ", i, "neg ll: ", nll)
            
    return np.asarray(opt_state[0]), nll   


#%%

'''RECURSIVELY FORMULATED LOGISTIC REGRESSION'''

@jit

def _log_prob_single_rflr(parameters, choices, rewards):
    
    alpha, beta, tau = parameters  # unpack parameters
    gamma = jnp.exp(-1 / tau)
    
    def update(carry, x):
        # unpack the carry
        ll, phi = carry

        # unpack the input
        prev_choice, choice, reward = x

        # update
        psi = phi + alpha * (2 * prev_choice - 1)
        ll += choice * psi - jnp.log(1 + jnp.exp(psi))
        phi = gamma * phi + beta * reward * (2 * choice - 1)

        new_carry = (ll, phi) 
        return new_carry, None
    
    
    ll = 0.0
    phi = beta * rewards[0] * (2 * choices[0] - 1)
    (ll, phi), _ = lax.scan(update, (ll, phi), (choices[:-1], choices[1:], rewards[1:]))
    
    return ll

#%%    
def log_probability_rflr(parameters, sessions):
    
    # compute probability of next choice
    ll = 0.0
    n = 0
    for choices, rewards in sessions:        
        # initialize "belief state" for this session
        
        ll += _log_prob_single_rflr(parameters, choices, rewards)
        n += len(choices) - 1
            
    return ll / n

#%% model_policies.py



import numpy as np
from functools import partial
from sklearn.linear_model import LogisticRegression
from scipy.special import expit as sigmoid
from scipy.stats import bernoulli
from tqdm.auto import trange
from scipy.special import logit


'''GENERAL'''

def model_to_policy(model_probs, sessions_data, policy='stochastic', **kwargs):
    
    '''relies on predefined histories and sorting
    INPUTS:
        model_probs (nested_list): posterior estimates from models for choice as probabilities
        sessions_data: usually for test dataset
        policy (str): decision policy off of model estimate; can be 'stochastic','greedy', or 'softmax'
        
    OUTPUTS:
        model_choice (nested list): predicted choices after applying policy
        model_switch: predicted switches, as trial-to-trial differences in predicted next choice from mouse previous choice
    '''
    
    def stochastic_pol(x): return int(np.random.rand() < x)
    
    def greedy_pol(x): return int(round(x))
    
    def softmax_pol(x, T): 
        choice_prob = sigmoid( logit(x) / T)
        return int(np.random.rand() < choice_prob) # same as stochastic_pol where probs have been filtered through softmax
    
    def make_choice(model_probs, mouse_choices, policy, **kwargs):
        
        '''
        use specified policy to make choice from each model estimate
        INPUTS:
            model_probs (nested list)
            mouse_choices (nested list)
            policy (str)
        OUTPUTS:
            predicted_choices (nested list)
            predicted_switches (nested list)
        '''
        
        predicted_choices, predicted_switches=[], []
        for session_probs, session_choice_history in zip(model_probs, mouse_choices):
            session_choices = [policy(model_prob[1]) for model_prob in session_probs]
            predicted_choices.append(session_choices)

            session_switches = [int(predicted_choice!=last_choice) for predicted_choice, last_choice \
                                in zip(session_choices[1:], session_choice_history[:-1])]
            session_switches.insert(0,0) # define first choice as not switch
            predicted_switches.append(session_switches)

        return predicted_choices, predicted_switches

    mouse_choices=[]
    for session_choices, session_rewards in sessions_data:
        mouse_choices.append(session_choices)
    
    if policy=='stochastic': model_choice, model_switch = make_choice(model_probs, mouse_choices, policy=stochastic_pol)
    elif policy=='greedy': model_choice, model_switch = make_choice(model_probs, mouse_choices, policy=greedy_pol)
    elif policy=='softmax': 
        T=kwargs.get('temp')
        model_choice, model_switch = make_choice(model_probs, mouse_choices, policy=partial(softmax_pol, T=T))
        
    return model_choice, model_switch

#%% 
def log_likelihood_model_policy(policies, sessions):
       
    '''
    evaluate the per trial likelihood of each session
    INPUTS:
        - policies (nested lists): choice probabilities output by model
        - sessions (nested lists): [choice, reward] for each session (just need choice)
        
    OUTPUTS:
        - ll/n (float): average log-likelihood across all trials
    '''
    
    ll = 0
    n = 0
    
    for i in trange(len(sessions), disable=True):
        choices, rewards = sessions[i]
        policy = policies[i][:, 1] # P(choice==1)

        # Update the log likelihood estimate
        ll += bernoulli.logpmf(choices, policy).sum()
        n += len(choices) # number of trials
        
    return ll / n

#%%
def log_likelihood_empirical_policy(policy_df, test_sessions, memory):
    ll = 0
    n = 0

    for row in test_sessions.iterrows():
        pleft = policy_df[policy_df.history==row[1].RL_history].pdecision.item()
        pleft = np.clip(pleft, 1e-4, 1-(1e-4))
        ll += row[1].Decision * np.log(pleft) + (1 - row[1].Decision) * np.log(1 - pleft)
        n += 1
    return ll / n

#%%
'''LOGISTIC REGRESSION'''

# pm1 = lambda x: 2 * x - 1
# feature_functions = [
#     lambda cs, rs: pm1(cs),                # choices
#     lambda cs, rs: rs,                     # rewards
#     lambda cs, rs: pm1(cs) * rs,           # +1 if choice = 1 and reward, 0 if no reward, -1 if choice=0 and reward
#     lambda cs, rs: np.ones(len(cs))        # overall bias term
    
# ]


# Changed feature function ofr 50% reward contingency

pm1 = lambda x:  2 * x - 1
feature_functions = [
    lambda cs, rs: pm1(cs),                # choices
    lambda cs, rs: rs,                     # rewards
    lambda cs, rs: pm1(cs) - rs,           # 0 if choice = 1 and reward, -1 if choice = 0 and no reward, +1 if choice = 1 and no reward
    lambda cs, rs: np.ones(len(cs))        # overall bias term
    
]
#%%
def encode_session(choices, rewards, memories, featfun):
    
    '''Helper to encode sessions in features and outcomes'''
    
    assert len(memories) == len(featfun)  
    
    # Construct the features
    features = []
    for fn, memory in zip(featfun, memories): 
        for lag in range(1, memory+1):
            # encode the data and pad with zeros
            x = fn(choices[:-lag], rewards[:-lag])
            x = np.concatenate((np.zeros(lag), x))
            features.append(x)
    features = np.column_stack(features)
    return features, choices

#%%    
def fit_logreg_policy(sessions, memories, featfun=feature_functions):
    
    '''
    fit logistic regression to training data
    INPUTS:
        - sessions (nested lists): [choices, rewards] from each session of training dataset
        - memories (list): memory length for each feature from featfun
        -featfun: list of functions to encode input features for logistic regression
    OUTPUTS:
        -lr (LogisticRegression): fit model
    '''
    
    encoded_sessions = [encode_session(*session, memories, featfun = featfun) for session in sessions]
    
    # This line of code uses a list comprehension to extract the first element of each session 
    # in the encoded_sessions list and then stacks these elements vertically into a NumPy array 
    # using np.row_stack().
    
    X = np.row_stack([session[0] for session in encoded_sessions])
    
    # Similarly for choices
    y = np.concatenate([session[1] for session in encoded_sessions])
    
    # Construct the logistic regression model and fit to training sessions
    # lr = LogisticRegression(C=1.0, fit_intercept=False)
    
    # lr.fit(X, y)
    
    lr = LogisticRegression(fit_intercept=False)
    from sklearn.model_selection import GridSearchCV

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')

    grid_search.fit(X, y)
    
    print("Best parameters:", grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X, y)
    print("Test accuracy:", accuracy)
    
    # Construct the logistic regression model and fit to training sessions
    lr = grid_search.best_estimator_
    
    lr.fit(X, y)
    
    
    return lr

#%%
def compute_logreg_probs(sessions, lr_args, featfun=feature_functions):
    
    '''
    use fit logistic regression to calculate choice probabilities from mouse data
    INPUTS:
        - sessions (nested lists): [choices, rewards] from each session of mouse data 
        - lr_args: fit logistic regression and memory lengths for features
        - featfun: list of functions to encode input features for logistic regression
    OUTPUTS:
        - policies (nested lists): choice probabilities by trial for each session 
    '''
    lr, memories = lr_args
    
    policies = []
    for choices, rewards in sessions:
        X, y = encode_session(choices, rewards, memories, featfun=featfun)
        policy = lr.predict_proba(X)#[:, 1]
        policies.append(policy)
    return policies

#%%
'''HMM VARIATIONS'''

# Now implement the Thompson sampling mouse using the HMM world model.  Use SSM to implement the observation and transition distribution.
from ssm.observations import Observations
from ssm.transitions import StationaryTransitions
from ssm import HMM

class MultiArmBanditObservations(Observations):
    """
    Instantiation of the k-arm bandit transition model.
    
    data:  (T, 1) array of rewards (int 0/1)
    input: (T, 1) array of choices (int 0,...,K-1)
    
    reward_prob:  probability of reward when choosing correct arm by assumption, reward delivered with probability 1-reward_prob when choosing incorrect arm.
    """
    
    def __init__(self, K, D, M=0, reward_prob=0.8, **kwargs):
        assert D == 1, "data must be 1 dim"
        assert M == 1, "inputs must be 1 dim"
        super(MultiArmBanditObservations, self).__init__(K, D, M=M, **kwargs)
        self.reward_prob = reward_prob
        
    def log_likelihoods(self, data, input, mask, tag):
        """
        data: sequence of binary choices
        input: array of binary rewards
        """
        assert data.dtype == int and data.ndim == 2 and data.shape[1] == 1
        assert input.ndim == 2 and input.shape[0] == data.shape[0] and input.shape[1] == 1
        assert input.min() >= 0 and input.max() <= self.K-1
        rewards, choices = data[:, 0], input[:, 0]
        
        # Initialize the output log likelihood
        T = len(data)

        lls = np.zeros((T, self.K))
        for k in range(self.K):
            p = self.reward_prob * (choices == k) + (1-self.reward_prob) * (choices != k)
            lls[:, k] = bernoulli.logpmf(rewards, p)
        return lls
        
    def m_step(self, expectations, datas, inputs, masks, tags,
               sufficient_stats=None,
               **kwargs):
        pass
        
        
class MultiArmBanditTransitions(StationaryTransitions):
    """
    Instantiation of the k-arm bandit transition model
    
    self_transition_prob: probability of transitioning from current state by assumption, it is equally likely to transition to any other state.
    """
    def __init__(self, K, D, M=1, self_transition_prob=0.98):
        super(MultiArmBanditTransitions, self).__init__(K, D, M=M)
        P = self_transition_prob * np.eye(K)
        P += (1 - self_transition_prob) / (K-1) * (1 - np.eye(K))
        self.log_Ps = np.log(P)
        
    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        pass

#%%
def compute_hmm_probs(behavior_features, parameters):
    
    '''
    Bayesian inference in a hidden Markov model to compute belief state (posteriors) from mouse history of choices and rewards
    
    INPUTS:
        - behavior_features (nested lists): [choices, rewards] from each session of mouse dataset
        - parameters (dictionary):  contains transition 'q' and emission 'p' probabilities
    OUTPUTS:
        - beliefs (nested lists): state probabilities by trial for each session
    '''
    
    #unpack parameters
    q=parameters['q']
    p=parameters['p']
    
    # Construct an HMM "world model" for estimating world state
    K, D, M = 2, 1, 1
    world_model = HMM(K=K, D=D, M=M, 
                      observations=MultiArmBanditObservations(K, D, M=M, reward_prob=p),
                      transitions=MultiArmBanditTransitions(K, D, M=M, self_transition_prob=q))
    
    beliefs = []
    for i in trange(len(behavior_features), disable=True): # by session
        choices, rewards = behavior_features[i]
        
        # Run the HMM filter to estimate world state
        belief = world_model.filter(rewards[:, None], input=choices[:, None])
        
        beliefs.append(belief)
    
    return beliefs

#%%
def hmm_stickiness(choices, alpha, beta, tau):
    ''' Compute the stickiness (delta t+1, equation 32) - i.e. deviation of logistic regression from optimal behavior'''
    
    decay = np.exp(-1/tau)
    s1 = alpha + beta/2
    s2 = -alpha * decay

    choices = pm1(choices) # encode choices as -1, 1
    T = len(choices) # history length of choice influence
    stickiness = np.zeros(T)
    stickiness[1] = s1 * choices[0]
    for t in range(2, T):
        stickiness[t] = decay * stickiness[t-1]
        stickiness[t] += s1 * choices[t-1] 
        stickiness[t] += s2 * choices[t-2] 
    return stickiness
#%%
def compute_stickyhmm_probs(behavior_features, parameters):
    
    '''
    Bayesian inference in a sticky hidden Markov model to compute belief state (posteriors) from mouse history of choices and rewards
    
    INPUTS:
        - behavior_features (nested lists): [choices, rewards] from each session of mouse dataset
        - parameters (dictionary):  contains transition 'q' and emission 'p' probabilities along with 'alpha', 'beta', 'tau' for stickiness
    OUTPUTS:
        - policies (nested lists): choice probabilities by trial for each session after adding stickiness to HMM beliefs
    
    '''
    
    # unpack parameters
    q=parameters['q']
    p=parameters['p']
    alpha=parameters['alpha']
    beta=parameters['beta']
    tau=parameters['tau']
    
    from scipy.special import expit as sigmoid
    from scipy.special import logit
    
    # Construct an HMM "world model" for estimating world state
    K, D, M = 2, 1, 1
    world_model = HMM(K=K, D=D, M=M, 
                      observations=MultiArmBanditObservations(K, D, M=M, reward_prob=p),
                      transitions=MultiArmBanditTransitions(K, D, M=M, self_transition_prob=q))
    
    policies = []
    deltas = []
    for i in trange(len(behavior_features), disable=True): # by session
        choices, rewards = behavior_features[i]
        
        # Run the HMM filter to estimate world state
        belief = world_model.filter(rewards[:, None], input=choices[:, None])
        
        # Add stickiness to the model
        stickiness = hmm_stickiness(choices, alpha, beta, tau)
        psi = logit(belief[:, 1]) + stickiness # add stickiness to belief
        
        policy = np.zeros((len(psi),2))
        policy[:, 1] = sigmoid(psi)
        policy[:, 0] = 1 - policy[:, 1]
        
        policies.append(policy)#[:, 1])
        deltas.append(stickiness) # deviation from original HMM
    
    return policies

#%%
'''RECURSIVELY FORMULTATED LOGISTIC REGRESSION (RFLR)'''
        
def RFLR(behavior_features, parameters):
    
    '''
    trial-by-trial calculation of choice probabilities using a recursively formulated logistic regression;
    reinitializes every session, uses mouse behavior as model history
    INPUTS:
        - behavior_features (nested lists): [choices, rewards] from each session of mouse dataset
        - parameters (tuple): alpha, beta, tau as fit with sgd
    OUTPUTS:
        - psi_sessions (nested lists): choice probabilities by trial for each session
            
    '''

    alpha, beta, tau = parameters  # unpack parameters
    
    gamma = np.exp(-1 / tau)
    
    psi_sessions=[]

    for choices, rewards in behavior_features:
        
        # initialize psi
        psi=np.zeros((len(choices), 2))
        psi[0,:]=[0.5,0.5] 
    
        # recode choices
        cbar = 2 * choices - 1
        
        # initialize "belief state"
        phi = beta * rewards[0] * cbar[0]
        
        for t in range(1, len(choices)):
            
            # evaluate probability of this choice
            # psi[t,:] = 1-sigmoid(phi + (alpha * cbar[t-1])), sigmoid(phi + (alpha * cbar[t-1]))
            psi[t, 0] = 1 - sigmoid(phi + (alpha * cbar[t-1]))
            psi[t, 1] = sigmoid(phi + (alpha * cbar[t-1]))
            
            # update belief state for next time step
            phi = gamma * phi + (beta*(rewards[t] * cbar[t])) 

        psi_sessions.append(psi)

    return psi_sessions

#%%
'''Q-LEARNING'''

def fq_learning_model(behavior_features, parameters):
    
    '''
    trial by trial calculation choice probabilities using a F-Q-learning algorithm
    reinitializes every session, uses mouse behavior as model history
    
    INPUTS:
        - df (nested lists): [choices, rewards] from each session of mouse dataset
        - parameters (tuple): alpha (choice history bias), k (learning=forgetting rate), T (softmax temperature) as derived from Logistic Regression
        
    OUTPUTS:
        - psi_sessions (nested lists): choice probabilities by trial for each session
    '''
    
    def update_q(q, choice, reward):
        
        '''update Q-values based on choice direction and choice outcome'''
        
        q[choice] = (k * (reward - q[choice])) + q[choice] # ((1-beta)*q[choice]) # 
        q[1-choice] = (1-k)*q[1-choice]
        
        return q
    
    alpha, k, T = parameters  # unpack parameters
        
    psi_sessions, q_sessions = [], [] 

    for choices, rewards in behavior_features:
        
        psi = np.zeros((len(choices), 2))
        psi[0,:] = [0.5,0.5]
        q = np.zeros_like(psi)
        q[0,:] = [0, 0]

        for t in range(1, len(choices)):
            
            # evaluate probability of this choice
            psi[t,1] = sigmoid( ( (q[t-1,1] - q[t-1,0])/T ) + (alpha * (2 * choices[t-1] - 1)) )
            psi[t,0] = 1 - psi[t,1]
            
            # update q for next trial
            q[t,:] = update_q(q[t-1,:], choices[t], rewards[t]) # q_t+1
               
        psi_sessions.append(psi)
        q_sessions.append(q)

    return q_sessions, psi_sessions

#%% model_simulations.py


import numpy as np
from scipy.special import expit as sigmoid
import numpy.random as npr
from scipy.special import logit

def observe_reward(choice, state, phigh):
    
    if choice==state:
        return np.random.choice(2,p=[1-phigh, phigh])
    else:
        return np.random.choice(2, p=[phigh,1-phigh])

def markov_process(curr_state,tprob):
    
    if curr_state:
        return np.random.choice(2,p=[tprob, 1-tprob])
    else:
        return np.random.choice(2,p=[1-tprob, tprob])
            
    
def make_choice(psi):
    
    return int(np.random.rand() < sigmoid(psi))
    
def rflr_simulation(rflr_params, task_params, nTrials=30000):
    
    alpha, beta, tau = rflr_params  # unpack parameters
    phigh, tprob = task_params
    
    gamma = np.exp(-1 / tau)
    
    nSessions = round(nTrials/750) # setting session length by mean mouse session
            
    nRewards=0 # cumulative reward count   
    N=0 # keep track of how many trials
    
    sessions = []
    session_states = []
    
    for iSession in range(1,nSessions):
        
        nTrials = np.random.randint(650,850)
    
        choices = [np.random.randint(2)] # randomly initialize first choice
        states = [np.random.randint(2)] # randomly initialize first state
        rewards = [observe_reward(choices[0], states[0], phigh)]
        psis = []
        # initialize "belief state"
        phi = beta * rewards[0] * (2 * choices[0] - 1)
        
        nRewards += rewards[0]
        N+=1
        
        for t in range(1, nTrials):

            states.append(markov_process(states[t-1], tprob))
            
            # update belief state for next time step
            psi = phi + (alpha * (2*choices[t-1]-1)) # compute probability of next choice

            # make choice and observe outcome off belief state
            choices.append(make_choice(psi))
            rewards.append(observe_reward(choices[t], states[t], phigh)) # now have moved to current time step
            psis.append(psi)
            
            phi = gamma * phi + (beta*(rewards[t] * (2*choices[t]-1))) # update evidence

            nRewards+=rewards[t]
            N+=1
            
        sessions.append([np.array(choices), np.array(rewards), np.array(psis)])
        session_states.append(states)
        
    return nRewards/N, sessions, session_states


class Mouse(object):
    """
    A mouse is just an agent that chooses left or right based on past experience and possibly some model 
    of the world.  It maintains some state that summarizes the past experience and updates that state
    when it receives new feedback; i.e. the outcomes of its choices.
    """
    def make_choice(self):
        raise NotImplementedError
        
    def receive_feedback(self, choice, reward):
        raise NotImplementedError

class BayesianMouse(Mouse):
    """
    This mouse maintains an estimate of the posterior distribution of the 
    rewarded port based on past choices and rewards.
    """
    def __init__(self, params):
        """
        Specify the HMM model parameters including 'p_switch' and 'p_reward'
        """
        # transition matrix specifies prob for each (current state, next state) pair
        p_switch = params['p_switch']
        self.transition_matrix = (1 - p_switch) * np.eye(2)
        self.transition_matrix += p_switch * (1 - np.eye(2))
        
        # reward probability specifies prob for each (choice, state) pair
        p_reward = params['p_reward']
        self.reward_probability = p_reward * np.eye(2)
        self.reward_probability += (1 - p_reward) * (1 - np.eye(2))
        self.posterior = 0.5 * np.ones(2)

    def make_choice(self, policy):
        
        prediction = np.dot(self.transition_matrix.T, self.posterior)
        
        if policy=='greedy':
            return np.where(prediction==prediction.max())[0][0]
        elif policy=='thompson':
            return npr.rand() < prediction[1]

        
    def receive_feedback(self, choice, reward):
        # For simulation only: update posterior distribution given new information
        assert self.posterior.shape == (2, )
        pr = self.reward_probability[int(choice)]
        lkhd = pr if reward else (1 - pr)
        
        self.posterior = np.dot(self.transition_matrix.T, self.posterior) * lkhd
        self.posterior /= self.posterior.sum()
        assert self.posterior.shape == (2, )
        
        
def simulate_experiment(params, mouse, states, policy, sticky=False):
    """
    Simulate an experiment with a given set of parameters and a mouse model.
    
    'params' is a dictionary with keys:
        'p_switch':  the probability that the rewarded port switches
        'p_reward':  the probability that reward is delivered upon correct choice
        
    'mouse' is an instance of a Mouse object defined below
    
    """
    sessions = []

    # Run the simulation
    for session_states in states: # switched from trange for reps
        
        choices, rewards, beliefs = [],[], []
        mouse.posterior = 0.5 * np.ones(2) # reset posterior for each session
        for state in session_states:
            # Make choice according to policy
            choices.append(mouse.make_choice(policy))

            # Deliver stochastic reward
            if choices[-1] == state:
                rewards.append(npr.rand() < params["p_reward"])
            else:
                rewards.append(npr.rand() < (1 - params["p_reward"]))
            
            mouse.receive_feedback(choices[-1], rewards[-1]) # update posterior
            
            if (sticky==True) & (len(choices)>2):
                mouse.update_stickiness(choices[-2:])
            beliefs.append(mouse.posterior)
        sessions.append([np.array(choices, dtype='int'), np.array(rewards, dtype='int'), np.array(beliefs)])
    
    return rewards, sessions

#%% plot_models_v_mouse.py



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
  
    
def get_block_position_summaries(data):
    
    bpos = pd.DataFrame()

    for group_col in ['block_pos_rev', 'blockTrial']:

        block_start, block_end = [0,20] if group_col=='blockTrial' else [-20,-1]

        summary_stats = data.groupby(group_col).agg(
            phigh = pd.NamedAgg(column = 'highPort', aggfunc = 'mean'),
            phigh_std = pd.NamedAgg(column = 'highPort', aggfunc = 'std'),
            pswitch = pd.NamedAgg(column = 'Switch', aggfunc = 'mean'),
            pswitch_std = pd.NamedAgg(column = 'Switch', aggfunc = 'std'),
            n = pd.NamedAgg(column = 'Switch', aggfunc = 'count')).loc[block_start:block_end]

        summary_stats.index.name='block_pos'

        bpos=pd.concat((bpos,summary_stats))

        
    return bpos.reset_index()
           
    
def plot_by_block_position(bpos, subset='condition', **kwargs):
    
    sns.set(style='ticks', font_scale=1.6, rc={'axes.labelsize':18, 'axes.titlesize':18}) 

    color_dict=kwargs.get('color_dict', {key:val for key, val in zip(bpos[subset].unique(), np.arange(len(bpos[subset].unique())))})
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10.5,3.5))
    ax1.vlines(x=0,ymin=0,ymax=1.05,linestyle='dotted',color='black')
    ax2.vlines(x=0,ymin=0,ymax=1 ,linestyle='dotted', color='black')

    for subset_iter in bpos[subset].unique(): 

        if type(color_dict[subset_iter])==np.int64:
            
            trace_color=sns.color_palette()[color_dict[subset_iter]]
            
            if subset_iter=='mouse':
                trace_color='gray'
                
        else:
            trace_color=color_dict[subset_iter]
        d = bpos.loc[bpos[subset] == subset_iter]
        
        ax1.plot(d.block_pos,d.phigh,label=subset_iter, alpha=0.8, linewidth=2, color=trace_color)
        ax1.fill_between(d.block_pos, y1=d.phigh - d.phigh_std / np.sqrt(d.n), 
                                y2=d.phigh + d.phigh_std / np.sqrt(d.n), alpha=0.2, color=trace_color)

        ax1.set_yticks([0,0.5, 1.0])

        ax2.plot(d.block_pos,d.pswitch, label=subset_iter, alpha=0.8, linewidth=2, color=trace_color)
        ax2.fill_between(d.block_pos,y1=d.pswitch - d.pswitch_std / np.sqrt(d.n), 
                                    y2=d.pswitch + d.pswitch_std / np.sqrt(d.n), alpha=0.2, color=trace_color)
        
        ax2.set_yticks(np.arange(0,0.6,step=0.1))#[0,0.1, 0.4])
    ax1.set(xlim=(-10,20), ylim=(0,1), xlabel='Block Position', ylabel='P(high port)')
    ax2.set(xlim=(-10,20), ylim=(0,np.max(bpos.pswitch)+0.05), xlabel='Block Position', ylabel='P(switch)') 
    
    if len(bpos[subset].unique())<5:
        ax1.legend(loc=[0.5,-0.03], fontsize=16,frameon=False)
    sns.despine()
    plt.tight_layout()
    
    
def plot_scatter(df_mouse, df_model):
    
    sns.set(style='ticks', font_scale=1.6, rc={'axes.labelsize':18, 'axes.titlesize':18})   
    sns.set_palette('dark')
    
    plt.figure(figsize=(4,4))
    plt.subplot(111, aspect='equal')
    plt.scatter(df_mouse.pswitch, df_model.pswitch, alpha=0.6, edgecolor=None, linewidth=0)
    plt.plot([0, 1], [0, 1], ':k')
    
    plt.xlabel('P(switch)$_{mouse}$')
    plt.ylabel('P(switch)')
    plt.xticks(np.arange(0, 1.1, 0.5))
    plt.yticks(np.arange(0, 1.1, 0.5))
    
    plt.tight_layout()
    sns.despine()
  
    
def plot_sequences(df, overlay=[], **kwargs):
    
    sns.set(style='ticks', font_scale=1.7, rc={'axes.labelsize':20, 'axes.titlesize':20})
    sns.set_palette('deep')

    overlay_label = kwargs.get('overlay_label', '')
    main_label = kwargs.get('main_label', '')
    yval = kwargs.get('yval','pswitch')
    # yval = kwargs.get('yval','pdecision')
    
    df = df.astype('object') # to deal with histories being treated as categorical from sorting
    
    fig, ax = plt.subplots(figsize=(14,4.2))
    if len(overlay)>0:
        overlay = overlay.astype('object')
        sns.barplot(x='history',y=yval, data=overlay, label=overlay_label, color=sns.color_palette()[0], ax=ax, alpha=1.0)
        ax.errorbar(x='history',y=yval, yerr=yval+'_err', data=overlay, fmt=' ', label=None, color=sns.color_palette('dark')[0])
        
    sns.barplot(x='history',y=yval,data=df, color='k', alpha=kwargs.get('alpha',0.4), label=main_label, ax=ax, edgecolor='gray')
    ax.errorbar(x='history',y=yval, yerr=yval+'_err', data=df, fmt=' ', color='k', label=None)

    
    if len(overlay_label)>0:
        ax.legend(loc='upper left', frameon=False)
    ax.set(xlim=(-1,len(df)), ylim=(0,1), ylabel='P(switch)', title=kwargs.get('title', None))
    plt.xticks(rotation=90)
    sns.despine()
    plt.tight_layout()
    
    
def internal_prob(a, b, n):
    
    return np.nansum(a * b * n) / np.nansum(n)


def calc_confusion_matrix(df_mouse, col, df_model=None):

    #[[actual repeat * predict repeat, actual repeat * predict switch],
    #[actual switch * predict repeat, actual switch * predict swich]]
    # and can sub in right / left
    
    if df_model is None:
        df_model = df_mouse.copy()
    else:
        assert(np.all(df_mouse.history.values == df_model.history.values))
    
    N = df_mouse.n.values # same counts for model
    a = df_mouse[col].values
    b = df_model[col].values

    raw_confusion = np.array([[internal_prob(1-a, 1-b, N), internal_prob(1-a, b, N)],
                               [internal_prob(a, 1-b, N), internal_prob(a, b, N)]])

    norm_confusion = raw_confusion / raw_confusion.sum(axis=1)[:,np.newaxis]

    return norm_confusion


def plot_confusion(df, df_model, cm_fig=None, col='pswitch', color='Blues', seq_nback=3, delta=True):
    
    sns.set(style='white', font_scale=1.3, rc={'axes.labelsize':16, 'axes.titlesize':16})
    
    if cm_fig is None:
        cm_fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.4, 2.2))
    else:
        ax = cm_fig.add_subplot(1, 2+delta, 2+delta)
        cm_fig.set_size_inches(8.4, 2.5)

    cm = calc_confusion_matrix(df, col, df_model)
        
    ax.imshow(cm, cmap=color)

    fmt='.2f'
    thresh = cm.max()/ 2.
    for i, row in enumerate(cm):
            
            for j, square in enumerate(row):
                
                ax.text(j, i, format(square, fmt),
                        ha="center", va="center",
                        color="white" if square > thresh else "black")
                
    column_dict = {'pswitch': ['repeat', 'switch'], 'pdecision':['left','right']}
    
    ax.set_xticks((0,1))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xticklabels(('{} '.format(column_dict[col][0]),' {}'.format(column_dict[col][1])))
    ax.set_yticks((0,1))
    ax.set_yticklabels(('{}'.format(column_dict[col][0]),'{}'.format(column_dict[col][1])))
    ax.set(xlabel='predicted', ylabel='actual', ylim=(-0.5, 1.5)) 
    ax.invert_yaxis()
    plt.tick_params(top=False, pad=-2)
    plt.tight_layout()
    
    return cm_fig


#%% resample_and_model_reps.py


import numpy as np
import pandas as pd
import itertools

# import plot_models_v_mouse as bp
# import conditional_probs as cprobs
# import model_policies as models
from sklearn.utils import resample


def pull_sample_dataset(session_id_list, data):
        
    '''
    INPUTS:
        - session_id_list (list): list of session names
        - data (pandas DataFrame): dataset
            
    OUTPUTS: 
        - sample_features (list of lists): [choices, rewards] x session
        - sample_target (list of lists): [target port] x session
        - sample_block_pos_core (pandas DataFrame): in the same format as DATA, containing only sessions in/sorted by session_id_list 
            
    Note: can be used to sample for train and test sets or for resampling from dataset for bootstrapping
    '''
    
    # choices and rewards (second idx, {0,1}) by session (first idx {0:nSessions}) for models
    sample_features = [[data[data.Session==session].Decision.values.astype('int'), \
                        data[data.Session==session].Reward.values.astype('int')] for session in session_id_list]
    sample_target = [data[data.Session==session].Target.values.astype('int') for session in session_id_list] # for expected reward only

    # makde test_df ordered same as test_sessions
    sample_block_pos_core = pd.concat([data[data.Session == session] for session in session_id_list] ).reset_index(drop=True)
    
    return sample_features, sample_target, sample_block_pos_core



def reconstruct_block_pos(blocks, model_choice, model_switch):
    
    '''
    takes mouse dataframe and replaces Switch, Decision, and highPort columns 
    with model predictions and get summary at each block position
    
    INPUTS:
        - blocks (pandas DataFrame): df with row for each trial, includes block position column
        - model_choice (nested lists): list of model choice predictions for each session
        - model_switch (nested lists): as with model_choice, for switch predictions
        
    OUTPUTS:
        - block_pos_model (pandas DF): same as BLOCKS, but with Switch, Decision, and highPort columns
                                       replaced with model predictions; model label marks model predictions
    '''
    
    block_pos_model = blocks.copy() 
    block_pos_model['model'] = 'model' # label all rows to fill as model predictions

    block_pos_model['Switch']=list(itertools.chain(*model_switch)) # fill with model switch predictions
    block_pos_model['Decision'] = list(itertools.chain(*model_choice)) # fill with model choice predictions
    block_pos_model['highPort']= list(itertools.chain(*model_choice)) == block_pos_model.Target # fill with model higher prob port
    
    return block_pos_model # return model version of data


def build_model_dfs(block_pos_model):
    
    '''initializes df for each form of analysis'''
            
    block_pos_model_summary = get_block_position_summaries(block_pos_model)
    
    symm_cprobs_model = calc_conditional_probs(block_pos_model, symm=True, action=['Switch', 'Decision']).reset_index()    

    port_cprobs_model = calc_conditional_probs(block_pos_model, symm=False, action=['Switch','Decision']).reset_index()

    return symm_cprobs_model, port_cprobs_model, block_pos_model_summary


def append_model_reps(block_pos_model, df_reps=None):
    
    '''builds up dataframes across repetitions of model runs'''

    cprobs_symm, cprobs_port, bpos_model = build_model_dfs(block_pos_model)
    
    phigh_reps, pswitch_reps, cprobs_symm_reps, cprobs_port_reps = df_reps #unpack df_reps
    
    phigh_reps = phigh_reps.merge(bpos_model[['block_pos','phigh']], on='block_pos', how='left', sort=False, suffixes=('', str(len(phigh_reps.columns))))
    pswitch_reps = pswitch_reps.merge(bpos_model[['block_pos', 'pswitch']], on='block_pos', how='left', sort=False, suffixes=('', str(len(pswitch_reps.columns))))
    cprobs_symm_reps = cprobs_symm_reps.merge(cprobs_symm[['history', 'pswitch']], on='history', how='left', sort=False, suffixes=('', str(len(cprobs_symm_reps.columns))))
    cprobs_port_reps = cprobs_port_reps.merge(cprobs_port[['history', 'pdecision']], on='history', how='left', sort=False, suffixes=('', str(len(cprobs_port_reps.columns))))
    
    return phigh_reps, pswitch_reps, cprobs_symm_reps, cprobs_port_reps

    
def reps_wrapper(model_func, session_list, data, n_reps, action_policy='stochastic', bs=True, **kwargs):
    
    '''
    Resamples sessions to run repetitions of model predictions using given model
    
    INPUTS:
        - model_func (function): partial model function with fit parameters given
        - session_list (list): list of sessions to be resampled
        - data (pandas DF): mouse behavior data 
        - n_reps (int): number of repetitions to run
        - action_policy (string): 'greedy', 'stochastic', 'softmax'
        - bs (bool): True if resampling, False if reps on same sessions
        **kwargs:
            - inv_temp (float): temperature parameter for softmax policy
            
    OUTPUTS:
        - bpos_model (pandas DF): summary across reps of P(high port) and P(switch) at each block position
        - cprobs_symm (pandas DF): summary across reps of P(switch | history) for symmetrical history 
        - cprobs_port (pandas DF): as above, but for lateralized history (right/left directionality preserved)
    '''
    
    phigh = get_block_position_summaries(data)[['block_pos']].copy() # block_pos column only
    pswitch = phigh.copy() # block_pos column only
    
    cprobs_symm = calc_conditional_probs(data, symm=True, action=['Switch', 'Decision'])
    cprobs_symm = cprobs_symm.sort_values(by='pswitch')[['history']].copy() # full dataset sorted histories 
    
    cprobs_port = calc_conditional_probs(data, symm=False, action=['Switch','Decision'])
    cprobs_port = cprobs_port.sort_values(by='pswitch')[['history']].copy() # full dataset sorted histories
    
    for i in range(n_reps):
        
        if i%100 == 0: print(f'rep {i}')

        if bs:
            resampled_sessions = resample(session_list) # resample test dataset with replacement
        else:
            resampled_sessions = session_list # if just running reps on test dataset without bootstrapping
        resampled_choice_reward, _, resampled_block_pos_core = pull_sample_dataset(resampled_sessions, data)
        model_probs = model_func(resampled_choice_reward) # calculate model probs on resampled dataset
        model_choices, model_switches = model_to_policy(model_probs, resampled_choice_reward, policy=action_policy, **kwargs)
        block_pos_model = reconstruct_block_pos(resampled_block_pos_core, model_choices, model_switches)
        
        # append to existing dfs from current resample
        phigh, pswitch, cprobs_symm, cprobs_port = append_model_reps(block_pos_model, df_reps=[phigh, pswitch, cprobs_symm, cprobs_port])

    block_pos = phigh.pop('block_pos') # new df with block positions
    pswitch = pswitch.drop(columns='block_pos')
    bpos_model = pd.DataFrame(data={'block_pos':block_pos, 'phigh': phigh.mean(axis=1),'phigh_std':phigh.std(axis=1),#/np.sqrt(n_reps), 
                                    'pswitch':pswitch.mean(axis=1), 'pswitch_std':pswitch.std(axis=1), 'n':n_reps}) #/np.sqrt(n_reps)}


    history = cprobs_symm.pop('history')
    cprobs_symm = pd.DataFrame(data={'history':history, 'n_reps':n_reps,
                                     'pswitch': np.nanmean(cprobs_symm,axis=1),'pswitch_err':np.nanstd(cprobs_symm,axis=1)})#/np.sqrt(n_reps)})
    
    history = cprobs_port.pop('history')
    cprobs_port = pd.DataFrame(data={'history':history, 'n_reps':n_reps,
                                     'pdecision': np.nanmean(cprobs_port,axis=1),'pdecision_err':np.nanstd(cprobs_port,axis=1)})#/np.sqrt(n_reps)})
    
    
    return bpos_model, cprobs_symm, cprobs_port


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        demo_models.ipynb
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
import numpy as np
import pandas as pd
import seaborn as sns

# import plot_models_v_mouse as bp
# import model_policies as models
from sklearn.model_selection import train_test_split
# import conditional_probs as cprobs
# import resample_and_model_reps as reps
# import model_fitting as fit

#%% Load data


data = pd.read_csv("C:\\Users\\shukl\\OneDrive\\Documents\\MATLAB\\Jadhav lab\\Analyzed\\allTrials_WT.csv")
data['Condition'] = '50-50'
data['Session'] = data['session']
data['Rat'] = 'FXM' + data['rat'].astype(str)
data['Session'] = data['Rat'] + '_' + data['Session'].astype(str)
data['Target'] = 1.0
data['Decision'] = data['match']
# data['Switch'] = data['switch']
# Add 'Switch' column
data['Switch'] = (data['match'] != data['match'].shift(1)).astype(int)

# data = data.dropna()

# data = pd.read_csv(os.path.join('bandit_data.csv'))

data.head()


#%%

probs='50-50' # P(high)-P(low)
seq_nback = 3 # history length for conditional probabilites
train_prop=0.7 # for splitting sessions into train and test
seed = np.random.randint(1000) # set seed for reproducibility

data = data.loc[data.Condition==probs] # segment out task condition


#%%

data = add_history_cols(data, seq_nback) # set history labels up front

train_session_ids, test_session_ids = train_test_split(data.Session.unique(), 
                                                       train_size=train_prop, random_state=seed) # split full df for train/test

# data['block_pos_rev'] = data['blockTrial'] - data['blockLength'] # reverse block position from transition
data['model']='mouse'
data['highPort'] = data['Decision']==data['Target'] # boolean, chose higher probability port

train_features, _, _ = pull_sample_dataset(train_session_ids, data)
test_features, _, block_pos_core = pull_sample_dataset(test_session_ids, data)

# bpos_mouse = get_block_position_summaries(block_pos_core)
# bpos_mouse['condition'] = 'mouse'


#%% Plot conditional switch probabilities for just the mouse behavior first

# full dataset for sorting
df_mouse_symm_reference = calc_conditional_probs(data, symm = False, 
                                                        action=['Switch'])#.sort_values('pswitch')
df_mouse_symm = calc_conditional_probs(block_pos_core, symm = False, action=['Switch', 'Decision'])
# df_mouse_symm = sort_cprobs(df_mouse_symm_reference, df_mouse_symm_reference.history.values)
plot_sequences(df_mouse_symm_reference, alpha = 0.5) 

#%% Fit models on training data, predict choice probabilities for held-out data

# Model predictions (model_probs) for each held out session stored for plotting below.

# Logistic regression
# Fit logistic regression on training set, with coefficients for choice history (up to trial n-L1), choice-reward interaction (up to trial n-L2), and reward history (up to trial n-L3).

L1 = 2 # choice history
L2 = 0 # choice * reward history
L3 = 0 # reward history
memories = [L1, L3, L2, 1]

lr = fit_logreg_policy(train_features, memories) # refit model with reduced histories, training set
model_probs = compute_logreg_probs(test_features, lr_args=[lr, memories])

# Recursively formulated logistic regression (RFLR)
params, nll = fit_with_sgd(log_probability_rflr, train_features) # quick fit on RFLR parameters
alpha, beta, tau = params
print(f'alpha = {alpha[0]:.2f}')
print(f'beta = {beta[0]:.2f}')
print(f'tau = {tau[0]:.2f}')

model_probs = RFLR(test_features, params)

#%% Hidden Markov model (HMM)

q = 0.55 # 1-p(block transition)
p = 0.5 # p(reward | high port)

model_probs = compute_hmm_probs(test_features, parameters={'q':q, 'p':p})


#%% forgetting Q-learning model (F-Q model)

# Using alpha, beta, and tau parameters derived from fit RFLR above.

T = (1-np.exp(-1/tau))/beta 
k = 1-np.exp(-1/tau) 
a = alpha 

model_probs = fq_learning_model(test_features, parameters=[a, k, T])

#%% Predict action and plot

# Apply policy ('greedy', 'stochastic', or 'softmax') to select actions from model_probs and plot behavior. Can add replicate runs for stochastic policies.

model_choices, model_switches = model_to_policy(model_probs, test_features, policy='softmax', temp = 0.95) # , temp = 0.95

block_pos_model = reconstruct_block_pos(block_pos_core, model_choices, model_switches)
# bpos_model = get_block_position_summaries(block_pos_model)
# bpos_model['condition'] = 'model' # label model predictions as such
# bpos_model_v_mouse = pd.concat((bpos_mouse, bpos_model)) # agg df with model predictions and mouse data
color_dict = {'mouse': 'gray', 'model': sns.color_palette()[0]}#plot_config['model_seq_col']}

# plot_by_block_position(bpos_model_v_mouse, subset='condition', color_dict = color_dict)

symm_cprobs_model = calc_conditional_probs(block_pos_model, symm=False, action=['Switch'])
# symm_cprobs_model = sort_cprobs(symm_cprobs_model, df_mouse_symm.history.values)

plot_sequences(df_mouse_symm_reference, overlay=symm_cprobs_model, main_label='mouse', overlay_label='model')

plot_scatter(df_mouse_symm_reference, symm_cprobs_model)


#%%  Implementing the Bayesian agent simulations

# Define parameters
params = {'p_switch': 0.5, 'p_reward': 0.5}
# Create an instance of BayesianMouse with specified parameters
mouse = BayesianMouse(params)

# Define states for the experiment
# Example: states = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]
# Each sublist represents the states for one session of the experiment
states = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]

# Define the policy ('greedy' or 'thompson')
policy = 'greedy'

# Call simulate_experiment function
rewards, sessions = simulate_experiment(params, mouse, states, policy)

# Analyze the results
# For example, you can analyze rewards and beliefs across sessions
for session in sessions:
    choices, rewards, beliefs = session[0], session[1], session[2]
    print("Choices:", choices)
    print("Rewards:", rewards)
    print("Beliefs:", beliefs)
    # Perform further analysis as needed


