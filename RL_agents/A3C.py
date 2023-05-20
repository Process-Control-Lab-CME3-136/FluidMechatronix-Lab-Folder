import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend as K
import multiprocessing as mp
from multiprocessing import Queue
from threading import Thread, Lock
import os


# ================================ Global Vars ==========================================================

max_steps = 300
MAX_EP = 500
sp_steps = 50
N_TD = 2
MAX_CLIP_NORM = 0.6
DELTA_U_SCALE = 5

SP_LIST = [0.8305668525314133, 15.736208991570876, 23.548405704989108,
           4.903718796523407, 16.05754370497057, 1.6723041377816614]
# [(np.random.random()*env.state_space_high-5)*0.8 for i in range(max_steps//sp_steps)] # SP list with 1 SP per 10 steps

tf.keras.utils.disable_interactive_logging()

alpha_a = 3.234e-6
alpha_c = 7.565e-5
l1_dims = 256
l2_dims = 256
Entropy = 0.003
activation = 'tanh'
reward_scale = 0.1
alpha_u = 0.6
SAVE_DIR = 'Agent_'
DIR_NAME = 'A3C_Agents'


entropy_decay = -3/2000  # 3/tau
gamma = 0.99

a_opt = tf.optimizers.RMSprop(learning_rate=alpha_a, name='a_opt')
# KalmanOptimizer(learning_rate=alpha_a)
c_opt = tf.optimizers.RMSprop(learning_rate=alpha_c, name='c_opt')

# ========================================================================================================


class Environment:
    def __init__(self, alpha_u=1e-2, *args, **kwargs):
        self.state_space_high = 36.0
        self.state_space_low = -10.0

        self.action_space_high = 50.0
        self.action_space_low = 0.0

        self.num_actions = 1
        self.num_states = 1
        self.reward = 0.0
        self.state = 0.0
        self.action = 0.0
        self.alpha_u = alpha_u
        self.SP = []

    def get_next_state(self, u):
        x = -2.1761 + -0.202*self.state + 0.014*u**2
        return x

    def reset(self):
        self.reward = 0.0
        self.state = 0.0
        return self.state

    def step(self, state, action, step):

        # clip state space and calculate reward
        next_state = K.clip(self.step(action),
                            self.state_space_low, self.state_space_high)

        self.reward = -0.25 * \
            np.abs(self.SP[step]-next_state) - \
            self.alpha_u*np.abs(self.action-action)
        # print('state',self.state, 'action', action, 'next_state', next_state)
        self.state = next_state
        self.action = action

        return self.state, self.reward


# Helper functions to normalize/de-normalize state
def normalize(state, env):
    return (state-env.state_space_low)/(env.state_space_high-env.state_space_low)


def de_normalize(state, env):
    return state*(env.state_space_high-env.state_space_low)+env.state_space_low


class ActorCriticNet():
    def __init__(self, num_actions=1, num_states=1, l1_dims=128, l2_dims=32, activation='tanh', name='actor_critic'):
        super(ActorCriticNet, self).__init__()
        self.num_states = num_states
        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
        self.model_name = name
        self.num_actions = num_actions
        self.name = name
        self.seed = 123456
        self.activation = activation

    def make_ac_network(self):
        # iniatilizers referenced from this paper: https://arxiv.org/pdf/1704.08863.pdf
        with tf.name_scope('actor'):

            # Input: [state, set-point], output: [mu, sigma] for action "u"
            self.inp_layer1 = tf.keras.Input(
                (self.num_states+1,), name='input_layer')
            self.nn1 = tf.keras.layers.Dense(self.l1_dims, activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.glorot_normal(
                seed=self.seed), name='dense_layer1')(self.inp_layer1)
            self.nn2 = tf.keras.layers.Dense(self.l2_dims, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.glorot_normal(
                seed=self.seed), name='dense_layer2')(self.nn1)
            self.mu = tf.keras.layers.Dense(
                self.num_actions, name='mu')(self.nn2)

            self.sigma = tf.keras.layers.Dense(
                self.num_actions, activation=tf.nn.softplus, name='sigma')(self.nn2)
            self.actor = tf.keras.Model(
                inputs=self.inp_layer1, outputs=[self.mu, self.sigma])

        with tf.name_scope('critic'):

            # Input: [state, set-point], output: value
            self.inp_layer2 = tf.keras.Input(
                (self.num_states+1,), name='input_layer')
            self.nn3 = tf.keras.layers.Dense(self.l1_dims, activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.glorot_normal(
                seed=self.seed), name='dense_layer1')(self.inp_layer2)
            self.nn4 = tf.keras.layers.Dense(self.l2_dims, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal(
                seed=self.seed), name='dense_layer2')(self.nn3)

            self.v = tf.keras.layers.Dense(1, name='value')(self.nn4)
            self.critic = tf.keras.Model(
                inputs=self.inp_layer2, outputs=self.v)

            return self.actor, self.critic


class A3CAgent:
    def __init__(self, num_actions, num_states, a_bounds, alpha=0.001, gamma=0.99, Entropy=0.001, max_steps=500, sp_steps=50):

        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions
        self.num_states = num_states
        self.a_bounds = a_bounds
        self.a_lower_bound = a_bounds[0]
        self.a_upper_bound = a_bounds[1]
        self.a_mu = np.mean(a_bounds)
        self.Entropy = Entropy
        self.max_steps = max_steps
        self.sp_steps = sp_steps
        self.is_loaded = False

        self.global_actor, self.global_critic = ActorCriticNet(
            num_actions, num_states, l1_dims,).make_ac_network()
        # self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))
        self.global_actor.compile(a_opt)
        self.global_critic.compile(c_opt)
        self.coord = tf.train.Coordinator()

    # load global_actor/critic model for training/testing

    def load_model(self, dir_path):
        # check if h5 module exists
        try:
            import h5py
        except:
            raise ImportError()

        try:
            print(f'loading model weights from {dir_path}')
            self.global_actor, self.global_critic = ActorCriticNet(
                self.num_actions, self.num_states).make_ac_network()
            self.global_actor.load_weights(dir_path+'_actor.h5')
            self.global_critic.load_weights(dir_path+'_critic.h5')

            self.is_loaded = True
        except FileNotFoundError as e:
            print(e.args)

    def save(self, model_idx):
        self.global_actor.save_weights(os.path.join(
            DIR_NAME, SAVE_DIR+str(model_idx)+'_actor.h5'))
        self.global_critic.save_weights(os.path.join(
            DIR_NAME, SAVE_DIR+str(model_idx)+'_critic.h5'))

    def train(self):
        with tf.device('/cpu:0'):
            queue = Queue()
            self.workers = [Worker(self.num_actions, self.num_states,
                                   self.a_bounds, self.global_actor,
                                   self.global_critic, self.alpha, self.gamma, self.Entropy,
                                   self.sp_steps, self.max_steps, queue, i, self.is_loaded) for i in range(1)]
        self.rewards_global = []
        Worker.coord = self.coord

        try:
            for i, worker in enumerate(self.workers):
                print(f'Worker:{i} start')
                worker.start()

            while True:
                reward = queue.get()
                if reward is not None:
                    self.rewards_global.append(reward)
                else:
                    break

            self.coord.join(self.workers)
            self.save(19)

        except Exception as e:
            print(e)
            for worker in self.workers:
                worker.join()


# Helper class to keep track of state, action and reward buffers
class Memory:

    def __init__(self):
        self.buf_s, self.buf_a, self.buf_r = [], [], []
        # self.ep_states = []
        # self.ep_actions = []

    def store(self, state, action, reward):
        self.buf_s.append(state)
        self.buf_a.append(action)
        self.buf_r.append(reward)

    def clear(self):
        self.buf_s, self.buf_a, self.buf_r = [], [], []


# referenced from: https://blog.tensorflow.org/2018/07/deep-reinforcement-learning-keras-eager-execution.html
class Worker(Thread):

    global_moving_avg_reward = 0.0
    lock = Lock()
    global_episode = 1
    best_score = -250

    def __init__(self, num_actions, num_states,
                 a_bounds, global_actor,
                 global_critic, alpha, gamma, Entropy,
                 sp_steps, max_steps, queue, idx, is_loaded, mem):
        super(Worker, self).__init__()
        self.num_actions = num_actions
        self.num_states = num_states
        self.a_lower_bound = a_bounds[0]
        self.a_upper_bound = a_bounds[1]
        self.a_mu = np.mean(a_bounds)
        self.alpha = alpha
        self.gamma = gamma
        self.Entropy = Entropy
        self.sp_steps = sp_steps
        self.max_steps = max_steps

        self.env = Environment()
        self.env.SP = SP_LIST
        self.global_actor = global_actor
        self.global_critic = global_critic

        self.local_actor, self.local_critic = ActorCriticNet(
            num_actions, num_states).make_ac_network()

        self.local_actor.compile(a_opt)
        self.local_critic.compile(c_opt)
        self.global_mem = mem

        if is_loaded:
            # load weights for model from global_ACNet model
            self.pull_global()

        self.queue = queue
        self.worker_idx = idx
        self.tfd = tfp.distributions

    def record_info(self, global_episode, reward, worker_idx, global_ma_reward, queue, states, actions):
        alpha = 0.9
        # update global moving average reward
        global_ma_reward = global_ma_reward*alpha + reward*(1-alpha)
        self.queue.put(reward)
        print(f'episode: {global_episode: 4d}, reward: {reward: 4.2f}, worker:{worker_idx: 2d}. Global moving avg reward:{global_ma_reward: 5.2f}')

        if global_episode % 10 == 0:  # save state and action results every 10 episodes

            # since one episode is run on a worker each, no lock needed
            self.global_mem.ep_states.append(states)
            self.global_mem.ep_actions.append(actions)

            # only for colab
            self.global_actor.save_weights(SAVE_DIR+str(10)+'_actor.h5')
            self.global_critic.save_weights(SAVE_DIR+str(10)+'_critic.h5')

        return global_ma_reward

    def choose_action(self, state):

        state = tf.convert_to_tensor(state)
        state = state[np.newaxis, :]

        self.mu, self.sigma = self.actor(state)
        # print(np.squeeze(self.mu), np.squeeze(self.sigma))
        # scale mu and sigma by action space
        # self.mu, self.sigma = K.clip(self.mu*self.a_bounds[1], self.a_bounds[0], self.a_bounds[1]), self.sigma + 1e-5

        self.mu, self.sigma = K.clip(
            self.mu*DELTA_U_SCALE, -DELTA_U_SCALE, DELTA_U_SCALE), self.sigma + 1e-5
        normal_dist = self.tfd.Normal(loc=self.mu, scale=self.sigma)

        self.A = tf.clip_by_value(tf.squeeze(
            normal_dist.sample(1), axis=0), -DELTA_U_SCALE, DELTA_U_SCALE)
        # pick random action from normal dist of action space

        return self.A.numpy()[0, 0]

    @tf.function(reduce_retracing=True)
    def learn(self, buf_s, buf_a, buf_v_td):
        """Critic Update"""
        with tf.GradientTape() as tape:
            v = self.local_critic(buf_s, training=True)
            v_next = buf_v_td
            td = tf.subtract(v_next, v, name='td_error')
            # mse error for v_pred-td_target
            self.c_loss = tf.reduce_mean(tf.square(td))

            # get gradients from local critic
            self.c_gradient = tape.gradient(
                self.c_loss, self.local_critic.trainable_variables)
        del tape

        """Actor Update"""
        with tf.GradientTape() as tape:
            self. mu, self.sigma = self.actor(buf_s)
            self.mu, self.sigma = K.clip(
                self.mu*DELTA_U_SCALE, -DELTA_U_SCALE, DELTA_U_SCALE), self.sigma + 1e-5
            normal_dist = self.tfd.Normal(loc=self.mu, scale=self.sigma)
            # print(np.squeeze(self.mu), np.squeeze(self.sigma))
            log_prob = normal_dist.log_prob(buf_a)
            exp_v = log_prob*td  # td error from critic
            entropy = normal_dist.entropy()
            exp_v = self.Entropy*entropy + exp_v
            self.a_loss = tf.reduce_mean(-exp_v)

            self.a_gradient = tape.gradient(
                self.a_loss, self.actor.trainable_weights)
            self.a_gradient = tf.clip_by_global_norm(
                self.a_gradient, MAX_CLIP_NORM)[0]
        del tape

        with self.lock:
            # update global critic weights
            c_opt.apply_gradients(
                zip(self.c_gradient, self.global_critic.trainable_variables))
            # self.local_critic.set_weights(self.global_critic.get_weights()) # pull global critic weights

            a_opt.apply_gradients(
                zip(self.a_gradient, self.global_actor.trainable_variables))
            # self.local_actor.set_weights(self.global_actor.get_weights())

        return self.c_gradient, self.a_gradient, self.c_loss, self.a_loss, v, self.mu, self.sigma

    @tf.function
    def pull_global(self):  # run by a local, pull weights from the global nets
        for l_p, g_p in zip(self.local_actor.trainable_weights, self.global_actor.trainable_weights):
            l_p.assign(g_p)
        for l_p, g_p in zip(self.local_critic.trainable_weights, self.global_critic.trainable_weights):
            l_p.assign(g_p)

    def run(self):

        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()
        tf.keras.utils.disable_interactive_logging()
        Worker.coord.clear_stop()

        mem = Memory()
        values_arr = []

        with tf.device('/device:GPU:0'):
            while Worker.global_episode < MAX_EP:
                state = env.reset()
                mem.clear()
                terminate = 0
                ep_reward = 0
                step = 1
                action = 0
                mse_err = 0

                # store states and actions.
                while True:
                    sp = env.SP[step//(sp_steps+1)]
                    sp_norm = normalize(sp, env)
                    del_action = agent.choose_action([state, sp_norm])
                    action += del_action
                    action = K.clip(action, env.action_space_low,
                                    env.action_space_high)

                    next_state, r = env.step(action, sp)

                    ep_reward += r
                    mse_err += (state-sp_norm)**2
                    mem.store([state, sp_norm], action, ep_reward)

                    if not step % self.sp_steps and self.worker_idx == 0:
                        print(
                            f'step: {step: 4}, state:{np.round(state, 3): 5.3f}, next_state:{np.round(next_state, 3): 5.3f}, action: {np.round(action, 3): 5.3f}')

                    if Worker.coord.should_stop() or step == self.max_steps:
                        # print worker info
                        Worker.global_moving_avg_reward = self.record_info(
                            Worker.global_episode, ep_reward, self.worker_idx, Worker.global_moving_avg_reward, self.queue)
                        # ep_states, ep_actions)

                        if ep_reward > Worker.best_score:
                            # save best model
                            with Worker.lock:
                                Worker.best_score = ep_reward
                                print(
                                    f'Saving best model to {SAVE_DIR}{self.worker_idx}, episode_reward:{ep_reward}')
                                self.global_actor.save_weights(
                                    SAVE_DIR+str(10)+'_actor.h5')
                                self.global_critic.save_weights(
                                    SAVE_DIR+str(10)+'_critic.h5')

                        Worker.global_episode += 1
                        # terminate worker
                        self.queue.put(None)
                        break

                    if step % N_TD == 0:  # update step, update actor and critic weights
                        next_state = tf.convert_to_tensor(
                            np.vstack([next_state]))
                        v_next = self.local_critic.predict([state, sp_norm])

                        buf_v_td = []
                        # get discounted rewards
                        for r in mem.buf_r[::-1]:
                            v_next = r + gamma*v_next
                            buf_v_td.append(v_next)

                        buf_v_td.reverse()

                        buf_s = tf.convert_to_tensor(np.vstack(mem.buf_s))
                        buf_a = tf.convert_to_tensor(np.vstack(mem.buf_a))
                        buf_v_td = tf.convert_to_tensor(
                            np.vstack(buf_v_td).astype('float32'))
                        # print(buf_s, buf_a, buf_v_td)
                        # get gradients from local model and update global weights
                        c_grad, a_grad, c_loss, a_loss, value, mu, sigma = self.learn(
                            buf_s, buf_a, buf_v_td)
                        mem.clear()

                        self.pull_global()
                        next_state = next_state.numpy()[0, 0]
                        values_arr.append(value.numpy()[0])

                    state = next_state
                    step += 1

                # only returning |state-SP| + alpha*|action-prev_action|
                ep_reward_scaled = ep_reward/reward_scale
                agent.Entropy = Entropy * \
                    np.exp(entropy_decay*Worker.global_episode)
                print(
                    f'episode:{Worker.global_episode+1: 3d}, reward(unscaled):{ep_reward: 4.2f}')

                with agent.summary_writer.as_default():
                    tf.summary.scalar(
                        'episodic reward', ep_reward_scaled, step=Worker.global_episode)
                    tf.summary.scalar('MSE Error', mse_err,
                                      step=Worker.global_episode)

                    if Worker.global_episode % 5 == 0:
                        tf.summary.histogram(
                            'actor/layer1/weights', agent.actor.layers[1].weights[0][0], step=Worker.global_episode)
                        tf.summary.histogram(
                            'actor/layer2/weights', agent.actor.layers[2].weights[0][0], step=Worker.global_episode)
                        tf.summary.histogram(
                            'critic/layer1/weights', agent.actor.layers[1].weights[0][0], step=Worker.global_episode)
                        tf.summary.histogram(
                            'critic/layer2/weights', agent.actor.layers[2].weights[0][0], step=Worker.global_episode)

                        tf.summary.histogram(
                            'gradients actor', a_grad[0], step=Worker.global_episode)
                        tf.summary.histogram(
                            'gradients critic', c_grad[0], step=Worker.global_episode)

                        tf.summary.scalar('Value', tf.math.reduce_sum(
                            values_arr).numpy(), step=Worker.global_episode)
                        tf.summary.scalar('mu', tf.reduce_mean(
                            mu), step=Worker.global_episode)
                        tf.summary.scalar('sigma', tf.reduce_mean(
                            sigma), step=Worker.global_episode)
                        tf.summary.scalar('c_loss', tf.reduce_mean(
                            c_loss), step=Worker.global_episode)
                        tf.summary.scalar('a_loss', tf.reduce_mean(
                            a_loss), step=Worker.global_episode)
                        tf.summary.flush()


if __name__ == '__main__':

    env = Environment()
    a_bounds = [env.action_space_low, env.action_space_high]
    s_bounds = [env.state_space_low, env.state_space_high]

    env.reward_scale = reward_scale
    env.alpha_u = alpha_u
    env.SP = SP_LIST

    a_opt = tf.optimizers.RMSprop(learning_rate=alpha_a, name='a_opt')
    c_opt = tf.optimizers.RMSprop(learning_rate=alpha_c, name='c_opt')

    agent = A3CAgent(num_actions=env.num_actions, a_bounds=a_bounds,
                     num_states=env.num_states, Entropy=0.01, max_steps=max_steps, sp_steps=sp_steps)

    print(f"Set points: {env.SP}")
    print(np.mean(env.SP), np.std(env.SP))

    try:

        if not os.path.exists(DIR_NAME):
            os.mkdir(DIR_NAME)

        model_idx = 5
        path = os.path.join(os.getcwd(), DIR_NAME)
        agent.load_model(os.path.join(path, SAVE_DIR+str(model_idx)))
        agent.train()

    except KeyboardInterrupt as e:
        # save the current model and exit
        print(f'Saving actor and critic models to {path}, index: {6}')
        # check if folder DIR_NAME exists

        agent.coord.request_stop()
        agent.save(6)

    except Exception as e:
        agent.coord.request_stop()
