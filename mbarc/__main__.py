import argparse
from copy import deepcopy
from time import strftime

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from mbarc.logger import Logger
from mbarc.env import make
from mbarc.focal_loss import FocalLoss
from mbarc.model import WorldModel
from mbarc.utils import one_hot_encode

logger = None


def random_exploration(env, steps=10000, verbose=False, log=False):
    buffer = []
    state = env.reset()
    iterator = range(steps)
    if verbose:
        iterator = tqdm(iterator, desc='Random exploration')
    for _ in iterator:
        action = torch.randint(0, env.action_space.n, ()).to(state)
        new_state, reward, done, _ = env.step(action)
        buffer.append((state, action, reward, new_state))

        if done:
            state = env.reset()
        else:
            state = new_state

    rewards = [0] * 2
    for _, _, reward, _ in buffer:
        rewards[int(reward)] += 1
    if verbose:
        print(f'negative rewards: {rewards[0]}, positive rewards: {rewards[1]}')
    if log:
        logger.log({'negative_rewards': rewards[0], 'positive_rewards': rewards[1]})

    return buffer


def get_counts(buffer):
    counts = [0] * 2
    for data in buffer:
        counts[data[2]] += 1
    return counts


def create_buffer(env, steps, ratio):
    buffer = []

    while len(buffer) < steps:
        data = random_exploration(env, steps, verbose=True)

        counts = get_counts(data)
        prob = counts[1] / counts[0] * (1 / ratio - 1)

        for sample in data:
            if sample[2] == 0:
                if float(torch.rand((1,))) < prob:
                    buffer.append(sample)
            else:
                buffer.append(sample)

    return buffer


def get_batch(buffer, batch_size, indices=None):
    if indices is None:
        indices = torch.randint(0, len(buffer), (batch_size,))

    shape = buffer[0][0].shape
    device = buffer[0][0].device

    states = torch.empty((batch_size, env.states, *shape)).to(device)
    actions = torch.empty((batch_size, env.action_space.n)).to(device)
    rewards = torch.empty((batch_size,), dtype=torch.int64).to(device)
    new_states = torch.empty((batch_size, *shape), dtype=torch.int64).to(device)

    for i in range(batch_size):
        state, action, reward, new_state = buffer[indices[i]]
        states[i] = one_hot_encode(state.float(), env.states).permute((2, 0, 1))
        actions[i] = one_hot_encode(action.float(), env.action_space.n)
        rewards[i] = reward
        new_states[i] = new_state

    return states, actions, rewards, new_states


def get_focal_loss(class_counts, beta=0.9999):
    def focal_loss(x, y):
        criterion = FocalLoss(reduction='none')
        loss = criterion(x, y)
        cb = torch.tensor(class_counts).to(y)
        cb = (1 - beta) / (1 - beta ** cb)
        cb = 2 * cb / cb.sum()
        cb = torch.gather(cb, 0, y)
        return torch.mean(cb * loss)

    return focal_loss


def forward(
        world_model,
        batch,
        optimizer=None,
        decouple_reward_head=True,
        decouple_frame_pred_head=False,
        mode='train'
):
    assert mode in ['train', 'eval']
    assert not (decouple_reward_head and decouple_frame_pred_head)

    correct_reconstruct = 0
    correct_rewards = [0] * 2
    total_rewards = [0] * 2

    states, actions, rewards, new_states = batch

    if mode == 'train':
        state_pred, reward_pred = world_model(states, actions)
    else:
        with torch.no_grad():
            state_pred, reward_pred = world_model(states, actions)

    loss_reconstruct = nn.CrossEntropyLoss(reduction='none')(state_pred, new_states)
    loss_reconstruct = torch.max(loss_reconstruct, torch.tensor(0.03).to(state_pred))
    loss_reconstruct = loss_reconstruct.mean() - 0.03

    loss_reward = get_reward_criterion(reward_pred, rewards)

    if decouple_reward_head:
        loss = loss_reconstruct
    elif decouple_frame_pred_head:
        loss = loss_reward
    else:
        loss = loss_reconstruct + loss_reward

    if mode == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    batch_size = len(state_pred)

    for k in range(batch_size):
        if torch.equal(torch.argmax(state_pred[k], 0), new_states[k]):
            correct_reconstruct += 1
        r = torch.argmax(reward_pred[k], 0)
        if r == rewards[k]:
            correct_rewards[r] += 1
        total_rewards[rewards[k]] += 1

    accuracy_reconstruct = 100 * correct_reconstruct / batch_size
    accuracy_reward = 100 * (correct_rewards[0] + correct_rewards[1]) / (total_rewards[0] + total_rewards[1])
    recall = [100 * correct_rewards[i] / max(1, total_rewards[i]) for i in range(2)]

    keys = ['loss', 'loss_reconstruct', 'loss_reward', 'acc_reconstruct', 'acc_reward', 'TNR', 'TPR']
    values = [float(loss), float(loss_reconstruct), float(loss_reward), accuracy_reconstruct, accuracy_reward, *recall]

    if mode == 'eval':
        keys = ['val_' + key for key in keys]

    history = dict(zip(keys, values))

    return history


def train_world_model(
        world_model,
        buffer,
        steps,
        lr=2e-3,
        decouple_reward_head=False,
        batch_size=64,
        eval_buffer=None,
        evaluations=30,
        log=False,
        verbose=True
):
    optimizer = Adam(world_model.parameters(), lr=lr)
    best_model = world_model
    best_val_acc = 0
    iterator = range(steps)
    if verbose:
        iterator = tqdm(iterator, desc='Training world model')
    postfix = {}

    for i in iterator:
        batch = get_batch(buffer, batch_size)
        world_model.train()
        history = forward(world_model, batch, optimizer=optimizer, decouple_reward_head=decouple_reward_head)
        postfix.update(history)

        if evaluations > 0 and eval_buffer is not None:
            if (i + 1) % (steps // evaluations) == 0 or (i + 1) == steps:
                val_history = eval_world_model(world_model, eval_buffer, log=False, verbose=verbose)
                postfix.update(val_history)
                val_acc = val_history['val_acc_reconstruct']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = deepcopy(world_model)
                    postfix.update({'best_val_acc': best_val_acc})

        if verbose:
            iterator.set_postfix(postfix)
        logger.log(postfix)

    return best_model


def eval_world_model(world_model, buffer, log=False, verbose=False):
    world_model.eval()

    batch_size = len(buffer)
    steps = len(buffer) // batch_size
    iterator = range(steps)
    if verbose:
        iterator = tqdm(iterator, desc='Evaluating world model')

    metrics = None
    postfix = {}

    for i in iterator:
        indices = list(range(i * batch_size, (i + 1) * batch_size))
        batch = get_batch(buffer, batch_size, indices)
        history = forward(world_model, batch, mode='eval')

        if metrics is None:
            metrics = torch.tensor(list(history.values()))
        else:
            metrics += torch.tensor(list(history.values()))

        keys = list(history.keys())
        values = metrics / (i + 1)
        postfix = dict(zip(keys, values.numpy()))

        if verbose:
            iterator.set_postfix(postfix)
        logger.log(postfix)

    return postfix


def train_reward_model(
        world_model,
        buffer,
        steps,
        strategy,
        lr=1e-4,
        batch_size=64,
        eval_buffer=None,
        evaluations=30,
        log=False,
        verbose=True
):
    optimizer = Adam(world_model.reward_head.parameters(), lr=lr)

    rewards = torch.empty((len(buffer),), dtype=torch.long)
    reward_count = torch.zeros((2,))
    for i, data in enumerate(buffer):
        rewards[i] = data[2]
        reward_count[int(data[2])] += 1

    if strategy == 'class_balanced':
        weights = 1 - reward_count / reward_count.sum()
        epsilon = 1e-4
        weights += epsilon
        indices_weights = torch.zeros_like(rewards, dtype=torch.float)
        for i in range(2):
            indices_weights[rewards == i] = weights[i]
        indices = torch.multinomial(indices_weights, steps * batch_size, replacement=True)

    elif strategy == 'square_root':
        weights = 1 - reward_count.sqrt() / reward_count.sqrt().sum()
        epsilon = 1e-4
        weights += epsilon
        indices_weights = torch.zeros_like(rewards, dtype=torch.float)
        for i in range(2):
            indices_weights[rewards == i] = weights[i]
        indices = torch.multinomial(indices_weights, steps * batch_size, replacement=True)

    elif strategy == 'progressively_balanced':
        indices = []
        ct = 50
        for t in range(ct + 1):
            weights_cb = 1 - reward_count / reward_count.sum()
            weights_ib = torch.full((2,), 1 / 2)
            prop = t / ct
            weights = (1 - prop) * weights_ib + prop * weights_cb

            epsilon = 1e-4
            weights += epsilon
            indices_weights = torch.zeros_like(rewards, dtype=torch.float)
            for i in range(2):
                indices_weights[rewards == i] = weights[i]

            indices.append(torch.multinomial(
                indices_weights.view((-1)),
                steps * batch_size // (ct + 1),
                replacement=True
            ))
        indices = torch.cat(indices)

    elif strategy == 'mbarc':
        d = torch.tensor([0.25, 0.25, 0.5])
        reward_mask = torch.empty((len(buffer),), dtype=torch.long)
        reward_count = torch.zeros((3,))  # ffw, ctr, or
        for i, data in enumerate(buffer):
            if int(data[2]) == 0:
                near = False
                if i == 0:
                    if int(buffer[i + 1][2]) == 1:
                        near = True
                elif i == len(buffer) - 1:
                    if int(buffer[i - 1][2]) == 1:
                        near = True
                else:
                    if int(buffer[i - 1][2]) == 1 or int(buffer[i + 1][2]) == 1:
                        near = True
                if near:
                    reward_count[1] += 1
                    reward_mask[i] = 1
                else:
                    reward_count[0] += 1
                    reward_mask[i] = 0
            else:
                reward_count[2] += 1
                reward_mask[i] = 2

        weights = [0] * 3
        for i in range(3):
            if reward_count[i] != 0:
                weights[i] = d[i] * steps * batch_size / reward_count[i]

        indices_weights = torch.zeros_like(reward_mask, dtype=torch.float)
        for i in range(3):
            indices_weights[reward_mask == i] = weights[i]
        indices = torch.multinomial(indices_weights, steps * batch_size, replacement=True)

        reward_count = [0] * 3
        for i in indices:
            data = buffer[i]
            if int(data[2]) == 0:
                near = False
                if i == 0:
                    if int(buffer[i + 1][2]) == 1:
                        near = True
                elif i == len(buffer) - 1:
                    if int(buffer[i - 1][2]) == 1:
                        near = True
                else:
                    if int(buffer[i - 1][2]) == 1 or int(buffer[i + 1][2]) == 1:
                        near = True
                if near:
                    reward_count[1] += 1
                else:
                    reward_count[0] += 1
            else:
                reward_count[2] += 1

    iterator = range(len(indices) // batch_size)
    if verbose:
        iterator = tqdm(iterator, desc='Training reward head')
    for i in iterator:
        batch = get_batch(
            buffer,
            batch_size,
            indices[i * batch_size:(i + 1) * batch_size]
        )

        world_model.eval()
        world_model.reward_head.train()
        postfix = forward(
            world_model,
            batch,
            optimizer=optimizer,
            decouple_frame_pred_head=True,
            decouple_reward_head=False
        )

        if evaluations > 0 and eval_buffer is not None:
            if (i + 1) % (len(indices) // batch_size // evaluations) == 0 or (i + 1) == len(indices) // batch_size:
                world_model.eval()
                metrics = eval_world_model(world_model, eval_buffer, log=False, verbose=True)
                postfix.update(metrics)

        logger.log(postfix)
        if verbose:
            iterator.set_postfix(postfix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buffer', type=str, default='data/buffer.pt')
    parser.add_argument('--buffer-test', type=str, default='data/buffer_test.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--env-name', type=str, default='MiniGrid-SimpleCrossingS9N2-v0')
    parser.add_argument('--experiment-name', type=str, default=strftime('%d-%m-%y-%H:%M:%S'))
    parser.add_argument('--load-model', type=str, default=None)
    parser.add_argument('--strategy', type=str, default='mbarc')
    parser.add_argument('--save-model', default=False, action='store_true')
    parser.add_argument('--use-modified-model', default=True, action='store_false')
    parser.add_argument('--use-cbf-loss', default=False, action='store_true')
    parser.add_argument('--use-wandb', default=False, action='store_true')
    config = parser.parse_args()

    assert config.strategy in ['online', 'class_balanced', 'square_root', 'progressively_balanced', 'mbarc']

    args = vars(config)
    max_len = 0
    for arg in args:
        max_len = max(max_len, len(arg))
    for arg in args:
        value = str(getattr(config, arg))
        display = '{:<%i}: {}' % (max_len + 1)
        print(display.format(arg, value))

    buffer = torch.load(config.buffer)
    for i in range(len(buffer)):
        buffer[i] = [buffer[i][j].to(config.device) for j in range(4)]
    buffer_test = torch.load(config.buffer_test)
    for i in range(len(buffer_test)):
        buffer_test[i] = [buffer_test[i][j].to(config.device) for j in range(4)]

    print(f'Buffer counts: {tuple(get_counts(buffer))}')
    print(f'Test buffer counts: {tuple(get_counts(buffer_test))}')

    env = make(config.env_name, config.device)

    world_model = WorldModel(
        env.states,
        env.action_space.n,
        modified_model=config.use_modified_model,
        env_shape=env.observation_space.shape
    ).to(config.device)

    if config.load_model is not None:
        world_model.load_state_dict(torch.load(config.load_model, map_location=config.device))

    if config.use_cbf_loss:
        get_reward_criterion = get_focal_loss(get_counts(buffer))
    else:
        get_reward_criterion = nn.CrossEntropyLoss()

    logger = Logger(config, world_model)

    if config.load_model is None:
        decouple_reward_head = config.strategy != 'online'
        world_model = train_world_model(
            world_model,
            buffer,
            3000,
            decouple_reward_head=decouple_reward_head,
            eval_buffer=buffer_test
        )

    if config.strategy != 'online':
        train_reward_model(world_model, buffer, 1500, config.strategy, eval_buffer=buffer_test)

    if config.save_model:
        torch.save(world_model.state_dict(), 'model.pt')
