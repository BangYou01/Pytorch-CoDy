import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy

import utils
from logger import Logger
from video import VideoRecorder

from cody_sac import CodySacAgent
from torchvision import transforms


def parse_args():
    # argparse 模块可以从 sys.argv 解析出command line参数,让人轻松编用户友好的命令行接口。
    # 使用 argparse 的第一步是创建一个 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    # environment
    # 给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的。
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--work_dir', default='.', type=str)  # modify

    # Hyperparameters
    parser.add_argument('--cody_lr', default=1e-3, type=float)
    parser.add_argument('--omega_cody_loss', default=0.1, type=float)

    parser.add_argument('--pre_transform_image_size', default=84, type=int)
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='cody_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--cody_latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')

    parser.add_argument('--log_interval', default=100, type=int)

    # ArgumentParser 通过 parse_args() 方法解析参数。它将检查命令行，把每个参数转换为适当的类型然后调用相应的操作。
    args = parser.parse_args()
    return vars(args)


def evaluate(env, agent, video, num_episodes, L, step, args):
    '''
    Evaluate the agent

    env:
    agent:
    video:
    num_episodes: the number of episodes per evaluation
    '''
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs,args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'cody_sac':
        return CodySacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            cody_latent_dim=args.cody_latent_dim,
            cody_lr=args.cody_lr,
            omega_cody_loss=args.omega_cody_loss
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main(
        domain_name,
        task_name,
        pre_transform_image_size,
        image_size,
        action_repeat,
        frame_stack,
        replay_buffer_capacity,
        agent,
        init_steps,
        num_train_steps,
        batch_size,
        hidden_dim,
        eval_freq,
        num_eval_episodes,
        critic_lr,
        critic_beta,
        critic_tau,
        critic_target_update_freq,
        actor_lr,
        actor_beta,
        actor_log_std_min,
        actor_log_std_max,
        actor_update_freq,
        encoder_type,
        encoder_feature_dim,
        encoder_lr,
        encoder_tau,
        num_layers,
        num_filters,
        cody_latent_dim,
        discount,
        init_temperature,
        alpha_lr,
        alpha_beta,
        save_tb,
        save_buffer,
        save_video,
        save_model,
        load_model,
        detach_encoder,
        cody_lr,
        omega_cody_loss,
        log_interval,
        seed,
        work_dir
):
    # specific seed with good perforcemence
    #seed = 306994
    # check parameters
    args = utils.Namespace(
        domain_name = domain_name,
        task_name = task_name,
        pre_transform_image_size = pre_transform_image_size,
        image_size = image_size,
        action_repeat = action_repeat,
        frame_stack = frame_stack,
        replay_buffer_capacity = replay_buffer_capacity,
        agent = agent,
        init_steps = init_steps,
        num_train_steps = num_train_steps,
        batch_size = batch_size,
        hidden_dim = hidden_dim,
        eval_freq = eval_freq,
        num_eval_episodes = num_eval_episodes,
        critic_lr = critic_lr,
        critic_beta = critic_beta,
        critic_tau = critic_tau,
        critic_target_update_freq = critic_target_update_freq,
        actor_lr = actor_lr,
        actor_beta = actor_beta,
        actor_log_std_min = actor_log_std_min,
        actor_log_std_max = actor_log_std_max,
        actor_update_freq = actor_update_freq,
        encoder_type = encoder_type,
        encoder_feature_dim = encoder_feature_dim,
        encoder_lr = encoder_lr,
        encoder_tau = encoder_tau,
        num_layers = num_layers,
        num_filters = num_filters,
        cody_latent_dim = cody_latent_dim,
        discount = discount,
        init_temperature = init_temperature,
        alpha_lr = alpha_lr,
        alpha_beta = alpha_beta,
        save_tb = save_tb,
        save_buffer = save_buffer,
        save_video = save_video,
        save_model= save_model,
        load_model = load_model,
        detach_encoder = detach_encoder,
        cody_lr = cody_lr,
        omega_cody_loss = omega_cody_loss,
        log_interval = log_interval,
        seed = seed,
        work_dir = work_dir
    )


    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    utils.set_seed_everywhere(args.seed)

    env = dmc2gym.make(
        # dm_control suite env
        domain_name=args.domain_name,
        task_name=args.task_name,
        # Reliable random seed initialization that will ensure deterministic behaviour.
        seed=args.seed,
        visualize_reward=False,
        # Setting from_pixels=True converts proprioceptive observations into image-based.
        # In additional, choose the image dimensions, by setting height and width.
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        # Setting frame_skip argument lets to perform action repeat
        frame_skip=args.action_repeat
    )
 
    env.seed(args.seed)  #Setting env_seed

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)
    
    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + str(ts) + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type
    args.work_dir = args.work_dir + '/' + exp_name

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    # open the file args.json in work_dir and write args into args.json
    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack,args.pre_transform_image_size,args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,  # args.image_size is the size of cropped images
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    #######Add read state_dict model ############
    if args.load_model:
        agent.load_cody(model_dir, 1000000)
        agent.load_init()
    #############################################

    # tensorboard visualization
    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
            if args.save_model and episode % 100 == 0:
                agent.save_cody(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    # log data into train.log
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            # expand one frame image into k frames by copying when resetting env
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1




if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    # parameter dict
    args_parse = parse_args()

    main(**args_parse)
