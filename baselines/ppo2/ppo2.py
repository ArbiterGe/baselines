import os
import time
import functools
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer

from mpi4py import MPI
from baselines.common.tf_util import initialize
from baselines.common.mpi_util import sync_from_root
import gym

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm, training=True, use_entropy_scheduler=False):
        sess = get_session()

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            act_model = policy(nbatch_act, 1, sess)
            train_model = policy(nbatch_train, nsteps, sess)
            # TODO(rachel0) - used for debugging
            #logstd = tf.get_variable(name='pi/logstd')
            #std = tf.exp(logstd)

        # TODO(rachel0) - used for debugging
        #print_log_std = tf.print("Log std dev: ", logstd)
        #print_std = tf.print("Std dev: ", std)
            
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        SCHEDULERENT = tf.placeholder(tf.float32, None)

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        if use_entropy_scheduler:
             # change -entropy by +entropy to penalize and schedule it
            loss = pg_loss + entropy * ent_coef * SCHEDULERENT + vf_loss * vf_coef
        else:
            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        params = tf.trainable_variables('ppo2_model')
        trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        grads_and_var = trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))

        if training:
            _train = trainer.apply_gradients(grads_and_var)

            def train(schedule_ent, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
                advs = returns - values
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                        CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values,SCHEDULERENT:schedule_ent }
                if states is not None:
                    td_map[train_model.S] = states
                    td_map[train_model.M] = masks
                # TODO(rachel0) - used for debugging
                #sess.run([print_std, print_log_std])
                return sess.run(
                    [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                    td_map
                )[:-1]
            self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']


            self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        if MPI.COMM_WORLD.Get_rank() == 0:
            initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        sync_from_root(sess, global_variables) #pylint: disable=E1101

class Runner(AbstractEnvRunner):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            # The returned values are arraws with num of elements = num threads
            # E.g. infos is [ncpu * info], where info is a dictionary per thread
            # and this is per step in teh episode
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for one_thread_info in infos:
                # This special key only appears when the episode ends and summarizes the episode info
                one_thread_one_step_info = one_thread_info.get('episode')
                if one_thread_one_step_info: epinfos.append(one_thread_one_step_info)
            mb_rewards.append(rewards)
            self.env.render2()
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, network, env, total_timesteps, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, using_mujocomanip=False,
          callback_func=None, max_schedule_ent=0, starting_timestep=1,
          logstd_anneal_start=None, logstd_anneal_end=None,
          **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)
    
    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns 
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation. 
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.

    
    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the 
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient
    
    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies, 
                                      should be smaller or equal than number of environments run in parallel. 

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training 
                                      and 0 is the end of the training 

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers. 

    

    '''
    
    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, initial_logstd=logstd_anneal_start, **network_kwargs)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                                max_grad_norm=max_grad_norm, use_entropy_scheduler=(max_schedule_ent != 0))
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=nsteps)
    nupdates = total_timesteps//nbatch
    logger.debug('nsteps', nsteps)
    logger.debug('nenvs', nenvs)
    logger.debug('total_timesteps', total_timesteps)
    logger.debug('nminibatches', nminibatches)
    logger.debug('noptepochs', noptepochs)
    logger.debug('nbatch = nenvs * nsteps ', nbatch)    
    logger.debug('nbatch_train = nbatch // nminibatches ', nbatch_train)    
    logger.debug('nupdates = total_timesteps//nbatch', nupdates)

    tfirststart = time.time()
    schedule_entropy_val = 0.0
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632

        epinfobuf.extend(epinfos)

        if logstd_anneal_start is not None and logstd_anneal_end is not None:
            with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
                mutable_logstd = tf.get_variable(name='pi/logstd')
                update_term = np.ones(mutable_logstd.shape)*(logstd_anneal_end-logstd_anneal_start)/float(nupdates)
                sess = tf.get_default_session()
                sess.run(tf.assign_add(mutable_logstd, update_term))
            
        if max_schedule_ent != 0.0:
            schedule_entropy_val += max_schedule_ent/float(nupdates)

        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(schedule_entropy_val, lrnow, cliprangenow, *slices)) # This loop is what takes long (roberto)
                    #if env.has_renderer:
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and MPI.COMM_WORLD.Get_rank() == 0:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%(update+starting_timestep-1))
            print('Saving to', savepath)
            model.save(savepath)
            if callback_func is not None: callback_func(savepath)
        
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            #logger.logkv("serial_timesteps", update*nsteps)
            #logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            #logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            #print(np.array([epinfo['r'] for epinfo in epinfobuf]))
            # print(np.array([epinfo['r'] for epinfo in epinfobuf]).shape)
            # print(np.sum(np.array([epinfo['r'] for epinfo in epinfobuf]),0))
            # print(safemean(np.sum(np.array([epinfo['r'] for epinfo in epinfobuf]),0)))
            #logger.logkv('eprewmean', safemean(np.sum(np.array([epinfo['r'] for epinfo in epinfobuf]),0)))
            # To publish extra information that is different per task
            if len(epinfobuf):
                if 'add_vals' in epinfobuf[0].keys():
                    for add_val in epinfobuf[0]['add_vals']:
                        logger.logkv(add_val+'mean', safemean([epinfo[add_val] for epinfo in epinfobuf]))

            logger.logkv('percent_viapoints_std', safestd([epinfo['percent_viapoints_'] for epinfo in epinfobuf]))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eprewstd', safestd([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()

            epinfobuf = deque(maxlen=nsteps)

    env.close()
    return model

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def safestd(xs):
    return np.nan if len(xs) == 0 else np.std(xs)



