

class HistoricalMarket:
    def __init__(self, data):
        self.data = data
        self.current_index = 0
    def sample_initial_state(self, d_market, rng):
        self.current_index = rng.randint(0, len(self.data)-1)
        return self.data[self.current_index]
    def step(self, m_t, rng):
        self.current_index += 1
        if self.current_index>=len(self.data):
            self.current_index=0
        return self.data[self.current_index]