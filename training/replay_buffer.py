from collections import deque
import numpy as np

class ReplayBuffer:
    """
    We store transitions (s, a, r, s_next, done, hx, cx, hx_next, cx_next).
    Then do batch updates.
    For demonstration, we store a limited capacity. We do 'clear()' each episode.
    
    Sample a random mini-batch from the buffer to do a single gradient update for both the actor and the critic.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done, hx, cx, hx_next, cx_next):
        # s, a, s_next are all Tensors or np arrays
        self.buffer.append((s, a, r, s_next, done, hx, cx, hx_next, cx_next))

    def sample(self, batch_size):
        # random sample
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        # return them in separate lists
        s_list, a_list, r_list, s_next_list, d_list, hx_list, cx_list, hx_next_list, cx_next_list = zip(*batch)
        return (s_list, a_list, r_list, s_next_list, d_list, hx_list, cx_list, hx_next_list, cx_next_list)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        