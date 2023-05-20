import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend as K
# import optuna
import datetime
import os
import csv

# ============================ Global Variables ===========================================
max_steps = 300
num_episodes = 2000
sp_steps = 50
N_TD = 2
MAX_CLIP_NORM = 0.6
DELTA_U_SCALE = 5

sp_list = [0.8305668525314133, 15.736208991570876, 23.548405704989108,
           4.903718796523407, 16.05754370497057, 1.6723041377816614]
# [(np.random.random()*env.state_space_high-5)*0.8 for i in range(max_steps//sp_steps)] # SP list with 1 SP per 10 steps

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
tf.keras.utils.disable_interactive_logging()

alpha_a = 3.234e-6
alpha_c = 7.565e-5
l1_dims = 256
l2_dims = 256
Entropy = 0.003
activation = 'tanh'
reward_scale = 0.1
alpha_u = 0.6


entropy_decay = -3/2000  # 3/tau
gamma = 0.99

# Entropy = Entropy*np.exp(-entropy_decay*800)

LOAD_DIR_ACTOR = 'Agent_actor_1.h5'
LOAD_DIR_CRITIC = 'Agent_critic_1.h5'

SAVE_DIR = 'Agent_'

# ============================================================================================


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

    def step(self, action, sp):
        
        # clip state space and calculate reward
        
        next_state = K.clip(self.get_next_state(action), self.state_space_low, self.state_space_high)
        action = K.clip(action, 0, 50)
        # get reward for un-normalized state and action as step is a function for un-normalized state and action
        self.reward = -self.reward_scale*(np.abs(sp-self.state) + self.alpha_u*np.abs(self.action-action))
        #print('state',self.state, 'action', action, 'next_state', next_state)
        norm_next_state = (next_state-self.state_space_low)/(self.state_space_high-self.state_space_low)
        self.state = next_state
        self.action = action

        # return normalized state
        return norm_next_state, self.reward


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
            self.mu = tf.keras.layers.Dense(self.num_actions, name='mu')(self.nn2)

            self.sigma = tf.keras.layers.Dense(self.num_actions, activation=tf.nn.softplus, name='sigma')(self.nn2)
            self.actor = tf.keras.Model(inputs=self.inp_layer1, outputs=[self.mu, self.sigma])

        with tf.name_scope('critic'):

            # Input: [state, set-point], output: value
            self.inp_layer2 = tf.keras.Input(
                (self.num_states+1,), name='input_layer')
            self.nn3 = tf.keras.layers.Dense(self.l1_dims, activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.glorot_normal(
                seed=self.seed), name='dense_layer1')(self.inp_layer2)
            self.nn4 = tf.keras.layers.Dense(self.l2_dims, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal(
                seed=self.seed), name='dense_layer2')(self.nn3)

            self.v = tf.keras.layers.Dense(1, name='value')(self.nn4)
            self.critic = tf.keras.Model(inputs=self.inp_layer2, outputs=self.v)

            return self.actor, self.critic


class A2CAgent:
    def __init__(self, num_actions, num_states, a_bounds, a_opt, c_opt, activation='tanh', alpha=0.001, gamma=0.99, Entropy=0.001, l1_dims=128, l2_dims=128):
        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions
        self.a_lower_bound = a_bounds[0]
        self.a_upper_bound = a_bounds[1]
        self.a_mu = np.mean(a_bounds)
        self.a_opt = a_opt
        self.c_opt = c_opt
        self.Entropy = Entropy
        self.log_dir = 'test'
        self.actor, self.critic = ActorCriticNet(
            num_actions, num_states, l1_dims, l2_dims, activation).make_ac_network()
        # self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))
        self.actor.compile(self.a_opt)
        self.critic.compile(self.c_opt)
        self.tfd = tfp.distributions

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        self.log_dir = os.path.join(
            self.log_dir, 'logs_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.mkdir(self.log_dir)

        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.summary_writer2 = tf.summary.create_file_writer(
            self.log_dir+"_SP")

    def choose_action(self, state):
        # state = tf.convert_to_tensor(state)

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
            v = self.critic(buf_s)
            v_next = tf.stop_gradient(buf_v_td)
            td = tf.subtract(v_next, v, name='td_error')
            # mse error for v_pred-td_target
            self.c_loss = tf.reduce_mean(tf.square(td))
        self.c_gradient = tape.gradient(
            self.c_loss, self.critic.trainable_weights)
        self.c_gradient = tf.clip_by_global_norm(
            self.c_gradient, MAX_CLIP_NORM)[0]
        self.c_opt.apply_gradients(
            zip(self.c_gradient, self.critic.trainable_weights))
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
        self.a_opt.apply_gradients(
            zip(self.a_gradient, self.actor.trainable_weights))
        del tape

        return self.c_gradient, self.a_gradient, self.c_loss, self.a_loss, v, self.mu, self.sigma

    def test(self, num_episodes, num_steps, sp_steps, is_sp_random):

        # Hard coding file paths for actor and critic weights
        fp_actor = 'Agent_actor_3.h5'
        fp_critic = 'Agent_critic_3.h5'
        csv_path = 'test_result.csv'
        if os.path.exists(csv_path):
            print('Removing previous data')
            os.remove(csv_path)

        # Reward scale same as training
        alpha_u = 0.6
        reward_scale = 0.1
        env = Environment(alpha_u=alpha_u, reward_scale=reward_scale)

        # This scale covers all possible state space values
        state_space_scale = env.state_space_high

        if is_sp_random:
            sp_list = [np.random.random()*state_space_scale -
                       10 for i in range(num_steps//sp_steps)]
            print(f"Set Point list generated randomly: {sp_list}")
        else:
            seed = 100
            np.random.seed(seed)
            # List size varies with num_steps and sp_steps
            sp_list = [
                np.random.random()*state_space_scale for i in range(num_steps//sp_steps)]
            print(f"Set Point list generated with seed={seed}: {sp_list}")

        env.SP = sp_list
        # load ac agent
        try:
            self.actor.load_weights(fp_actor)
            self.critic.load_weights(fp_critic)
        except:

            raise FileNotFoundError()

        for episode in range(num_episodes):

            state = env.reset()
            episode_reward = 0.0
            step = 1
            action = 0

            state_trajectory = []
            actions = []
            sp_trajectory = []

            while step < num_steps:

                sat_step = num_steps
                sp = env.SP[step//(sp_steps+1)]
                sp_norm = normalize(sp, env)

                del_action = agent.choose_action([state, sp_norm])
                action += del_action

                action = K.clip(action, env.action_space_low,
                                env.action_space_high)
                next_state, r = env.step(action, sp)
                episode_reward += r

                if episode == 0:
                    # save state, action, sp data to csv for plotting
                    state_trajectory.append(de_normalize(
                        tf.squeeze(state).numpy(), env))
                    sp_trajectory.append(sp)
                    actions.append(tf.squeeze(action).numpy())

                state = next_state
                step += 1

            print(
                f'Test | episode:{episode+1: 3d} | reward:{episode_reward: 4.2f}')

            if episode % 10 == 0:
                # save the lists to csv file
                with open(csv_path, 'a') as csv_fp:
                    writer = csv.writer(csv_fp, dialect='excel')
                    writer.writerows(
                        list(zip(*[state_trajectory, sp_trajectory, actions])))


if __name__ == "__main__":

    env = Environment()
    a_bounds = [env.action_space_low, env.action_space_high]
    s_bounds = [env.state_space_low, env.state_space_high]
    env.reward_scale = reward_scale
    env.alpha_u = alpha_u
    env.SP = sp_list

    a_opt = tf.optimizers.RMSprop(learning_rate=alpha_a, name='a_opt')
    # KalmanOptimizer(learning_rate=alpha_a)
    c_opt = tf.optimizers.RMSprop(learning_rate=alpha_c, name='c_opt')

    agent = A2CAgent(num_actions=env.num_actions, a_bounds=a_bounds,
                     a_opt=a_opt, c_opt=c_opt, num_states=env.num_states, Entropy=Entropy, l1_dims=l1_dims, l2_dims=l2_dims, activation=activation)

    # The actor and critic file paths must be specified before running.
    agent.actor.load_weights(LOAD_DIR_ACTOR)
    agent.critic.load_weights(LOAD_DIR_CRITIC)

    rewards = np.zeros(num_episodes)
    buf_s, buf_a, buf_r = [], [], []
    values_arr = []
    # states, actions = [], []

    print(f"Set points: {env.SP}")
    print(f"SP mean:{np.mean(env.SP)}, std:{np.std(env.SP)}")

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    tf.keras.backend.clear_session()
    # tf.summary.trace_on(graph=True, profiler=True)

    try:
        with tf.device('/device:GPU:0'):
            # store states and actions.
            for episode in range(num_episodes):
                state = env.reset()
                terminate = 0
                ep_reward = 0
                step = 1
                action = 0
                mse_err = 0

                while not terminate:
                    # state = (state-env.state_space_low)/(env.state_space_high-env.state_space_low)  # Normalized state
                    sp = env.SP[step//(sp_steps+1)]
                    sp_norm = normalize(sp, env)
                    del_action = agent.choose_action([state, sp_norm])

                    action += del_action
                    # if (step)%10 == 0: print(action, K.clip(action, env.action_space_low, env.action_space_high))
                    action = K.clip(action, env.action_space_low,
                                    env.action_space_high)
                    next_state, r = env.step(action, sp)
                    # print(de_normalize(state, env), de_normalize(next_state, env))
                    ep_reward += r
                    mse_err += (state-sp_norm)**2
                    buf_s.append([state, sp_norm])
                    buf_a.append(del_action)
                    buf_r.append(r)

                    # print(buf_s_err, next_state, sp_norm, action)
                    if not step % sp_steps:
                        print(
                            f'step: {step:5d}, state:{de_normalize(tf.squeeze(state).numpy(), env):4.2f}, next_state:{de_normalize(tf.squeeze(next_state).numpy(), env):4.2f}, action: {action:5}')
                    if next_state is None or step == max_steps:
                        break

                    if episode % 50 == 0:
                        # store state action pairs vs. SP
                        with agent.summary_writer.as_default():
                            tf.summary.scalar('trajectory', de_normalize(tf.squeeze(
                                state).numpy(), env), step=step+max_steps*episode//50)
                        with agent.summary_writer2.as_default():
                            tf.summary.scalar(
                                'trajectory', sp, step=step+max_steps*episode//50)
                        with agent.summary_writer.as_default():
                            tf.summary.scalar(
                                'trajectory/action', action, step=step+max_steps*episode//50)

                    if step % N_TD == 0:  # episode complete, update actor and critic weights
                        next_state = tf.convert_to_tensor(
                            np.vstack([next_state]))
                        v_next = agent.critic.predict([state, sp_norm])

                        buf_v_td = []
                        # get discounted rewards
                        for r in buf_r[::-1]:
                            v_next = r + gamma*v_next
                            buf_v_td.append(v_next)

                        buf_v_td.reverse()

                        buf_s = tf.convert_to_tensor(np.vstack(buf_s))
                        buf_a = tf.convert_to_tensor(np.vstack(buf_a))
                        buf_v_td = tf.convert_to_tensor(
                            np.vstack(buf_v_td).astype('float32'))

                        # update actor and critic, currently using monte carlo estimation method
                        c_grad, a_grad, c_loss, a_loss, value, mu, sigma = agent.learn(
                            buf_s, buf_a, buf_v_td)
                        values_arr.append(value.numpy()[0])
                        buf_s, buf_a, buf_r = [], [], []
                        next_state = next_state.numpy()[0, 0]

                    state = next_state
                    step += 1

                # only returning |state-SP| + alpha*|action-prev_action|
                ep_reward_scaled = ep_reward/reward_scale
                with agent.summary_writer.as_default():
                    tf.summary.scalar('episodic reward',
                                      ep_reward_scaled, step=episode)
                    tf.summary.scalar('MSE Error', mse_err, step=episode)

                    if episode % 5 == 0:
                        tf.summary.histogram(
                            'actor/layer1/weights', agent.actor.layers[1].weights[0][0], step=episode)
                        tf.summary.histogram(
                            'actor/layer2/weights', agent.actor.layers[2].weights[0][0], step=episode)
                        tf.summary.histogram(
                            'critic/layer1/weights', agent.actor.layers[1].weights[0][0], step=episode)
                        tf.summary.histogram(
                            'critic/layer2/weights', agent.actor.layers[2].weights[0][0], step=episode)

                        tf.summary.histogram(
                            'gradients actor', a_grad[0], step=episode)
                        tf.summary.histogram(
                            'gradients critic', c_grad[0], step=episode)

                        tf.summary.scalar('Value', tf.math.reduce_sum(
                            values_arr).numpy(), step=episode)
                        tf.summary.scalar(
                            'mu', tf.reduce_mean(mu), step=episode)
                        tf.summary.scalar(
                            'sigma', tf.reduce_mean(sigma), step=episode)
                        tf.summary.scalar(
                            'c_loss', tf.reduce_mean(c_loss), step=episode)
                        tf.summary.scalar(
                            'a_loss', tf.reduce_mean(a_loss), step=episode)
                        tf.summary.flush()

                values_arr = []
                agent.Entropy = Entropy*np.exp(entropy_decay*episode)

                print(
                    f'episode:{episode+1: 3d}, reward(unscaled):{ep_reward: 4.2f}')
                rewards[episode] = ep_reward_scaled

            # tf.summary.trace_export(name='default', step=0, profiler_outdir=agent.log_dir)
            # return mean rewards from the last 10 episodes

    except KeyboardInterrupt:
        print(f"Saving actor and critic weights at : {SAVE_DIR}")
        agent.actor.save_weights(f"{SAVE_DIR}actor_3.h5")
        agent.critic.save_weights(f"{SAVE_DIR}critic_3.h5")

    except Exception as e:
        print(e)
