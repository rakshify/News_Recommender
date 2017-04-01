class Adversary:
	def __init__(self, w_star, X, model = "logistic", log_bias = None, link_func=lambda x: exp(x)/(1+exp(x))):
		self.w_star_ = w_star
		self.link = link_func
		self.log_bias_ = log_bias
		self.a_star_reward = np.amax(self.log_bias_)

	def get_adversary_reward(self, X):
		l = len(self.w_star_[X])
		idx = random.randint(0, l - 1)
		reg = self.log_bias_[X]
		return self.w_star_[X][idx], reg
