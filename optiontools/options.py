import numpy as np
from scipy import stats as st

def overrides(interface_class):
	def overrider(method):
		assert(method.__name__ in dir(interface_class))
		return method
	return overrider

class Option:
	""" Abstract model representing option derivatives.

	Attributes
	----------
		So : float
			spot price of the underlying asset
		K : float
			strike price
		rf : float
			continuous risk free rate
		sigma : float
			volatility of the underlying asset
		T : float
			time to maturity of the option
		option_type : str, valid ['call', 'put'], default 'call'
			option type
		q : float, default 0
			expected dividend yield rate

	Methods
	-------
	price : float
		fair price of the option
	"""

	VALID_TYPES = ['call', 'put']

	def __init__(self, So, K, rf, sigma, T, option_type='call', q=0) -> None:
		"""
		Constructs attributes for the option class.

		Parameters
		----------
			So : float
				spot price of the underlying asset
			K : float
				strike price
			rf : float
				continuous risk free rate
			sigma : float
				volatility of the underlying asset
			T : float
				time to maturity of the option
			option_type : str, valid ['call', 'put'], default 'call'
				option type
			q : float, default 0
				expected dividend yield rate
		"""
		if option_type not in self.VALID_TYPES:
			raise Exception('Valid option types are `call` or `put`.')
		self.So = So
		self.K = K
		self.rf = rf
		self.sigma = sigma
		self.T = T
		self.option_type = option_type
		self.q = q

	def __str__(self):
		return f'{self.option_type} option.'

	def price(self):
		pass


class European(Option):
	""" This is a model representing European Options.

	Containing methods to evaluate the option using Black Scholes Merton
	model, and greeks.

	Attributes
	----------
		So : float
			spot price of the underlying asset
		K : float
			strike price
		rf : float
			continuous risk free rate
		sigma : float
			volatility of the underlying asset
		T : float
			time to maturity of the option
		option_type : str, valid ['call', 'put'], default 'call'
			option type
		q : float, default 0
			expected dividend yield rate

	Methods
	-------
	get_d1 : float
		d1 of the option
	get_d2 : float
		d2 of the option
	delta : float
		delta of the option
	vega : float
		vega of the option
	theta : float
		theta of the option
	rho : float
		rho of the option
	epsilon : float
		epsilon of the option
	omega : float
		omega of the option
	gamma : float
		gamma of the option
	vanna : float
		vanna of the option
	charm : float
		charm of the option
	vomma : float
		vomma of the option
	veta : float
		veta of the option
	speed : float
		speed of the option
	zomma : float
		zomma of the option
	color : float
		color of the option
	ultima : float
		ultima of the option
	dual_delta : float
		dual delta of the option
	dual_gamam : float
		dual gamma of the option
	price : float
		fair price of the option
	"""

	def __init__(self, So, K, rf, sigma, T, option_type='call', q=0) -> None:
		"""
		Constructs attributes for the option class.

		Parameters
		----------
			So : float
				spot price of the underlying asset
			K : float
				strike price
			rf : float
				continuous risk free rate
			sigma : float
				volatility of the underlying asset
			T : float
				time to maturity of the option
			option_type : str, valid ['call', 'put'], default 'call'
				option type
			q : float, default 0
				expected dividend yield rate
		
		Returns
		-------
		None
		"""
		super().__init__(So, K, rf, sigma, T, option_type, q)

	def get_d1(self) -> float:
		""" Calculates d1 from Black Scholes Merton model.

		Returns
		-------
		float : d1 of the option
		"""
		return (np.log(self.So/self.K) + (self.rf + 0.5*self.sigma**2)*self.T) /\
				(self.sigma*self.T**0.5)

	def get_d2(self) -> float:
		""" Calculates d2 from Black Scholes Merton model.

		Returns
		-------
		float : d1 of the option
		"""
		return (np.log(self.So/self.K) + (self.rf - 0.5*self.sigma**2)*self.T) /\
                (self.sigma*self.T**0.5)

	def delta(self) -> float:
		""" Calculates delta from Black Scholes Merton model.
		Delta = dV/dS

		Returns
		-------
		float : measuring the rate of change of the option's price with respect
			to the underlying asset's price
		"""
		d1 = self.get_d1()
		if self.option_type == 'put':
			return -np.exp(-self.q * self.T) * st.norm.cdf(-d1)
		else:
			return np.exp(-self.q * self.T) * st.norm.cdf(d1)

	def vega(self) -> float:
		""" Calculates vega from Black Scholes Merton model.
		Vega = dV/d sigma

		Returns
		-------
		float : measuring the sensitivity to volatility
		"""
		d1 = self.get_d1()
		return self.So * np.exp(-self.q * self.T) * st.norm.pdf(d1) * self.T**0.5

	def theta(self) -> float:
		""" Calculates theta from Black Scholes Merton model.
		Theta = dV/dt

		Returns
		-------
		float : measuring the sensitivity to the passage of time
		"""
		d2 = self.get_d2()
		d1 = self.get_d1()
		if self.option_type == 'put':
			return self.rf * self.K * np.exp(-self.rf * self.T) * st.norm.cdf(-d2) -\
				self.q * self.So * np.exp(-self.q * self.T) * st.norm.cdf(-d1) -\
				np.exp(-self.q * self.T) * self.So * st.norm.pdf(d1) * self.sigma / (2 * self.T**0.5)
		else:
			return -self.rf * self.K * np.exp(-self.rf * self.T) * st.norm.cdf(d2) +\
				self.q * self.So * np.exp(-self.q * self.T) * st.norm.cdf(d1) -\
				np.exp(-self.q * self.T) * self.So * st.norm.pdf(d1) * self.sigma / (2 * self.T**0.5)

	def rho(self) -> float:
		""" Calculates rho from Black Scholes Merton model.
		Rho = dV/dr

		Returns
		-------
		float : measuring the sensitivity to the risk free rate
		"""
		d2 = self.get_d2()
		if self.option_type == 'put':
			return -self.K * self.T * np.exp(-self.rf * self.T) * st.norm.cdf(-d2)
		else:
			return self.T * self.K * np.exp(-self.rf * self.T) * st.norm.cdf(d2)
	
	def epsilon(self) -> float:
		""" Calculates epsilon from Black Scholes Merton model.
		Epsilon = dV/dq

		Returns
		-------
		float : measuring the sensitivity to the expected yield rate
		"""
		d1 = self.get_d1()
		if self.option_type == 'put':
			return self.So * self.T * np.exp(-self.q * self.T) * st.norm.cdf(-d1)
		else:
			return -self.So * self.T * np.exp(-self.q * self.T) * st.norm.cdf(d1)

	def omega(self) -> float:
		""" Calculates omega from Black Scholes Merton model.
		Omega = dV/dS * S/V

		Returns
		-------
		float : percentage change in option value per percentage change in
			the underlying price
		"""
		delta = self.delta()
		price = self.price()
		return delta * self.So / price

	def gamma(self) -> float:
		""" Calculates gamma from Black Scholes Merton model.
		Gamma = d^2V/dS^2

		Returns
		-------
		float : measuring the rate of change in delta with respect to changes
			in the underlying price
		"""
		d1 = self.get_d1()
		return np.exp(-self.q * self.T) * st.norm.pdf(d1) * (1/(self.So * self.sigma * self.T**0.5))

	def vanna(self) -> float:
		""" Calculates vanna from Black Scholes Merton model.
		Vanna = d Delta / d sigma

		Returns
		-------
		float : measuring the sensitivity of the option delta with respect to
			the volatility
		"""
		d1 = self.get_d1()
		d2 = self.get_d2()
		return -np.exp(-self.q * self.T) * st.norm.pdf(d1) * d2 / self.sigma

	def charm(self) -> float:
		""" Calculates charm from Black Scholes Merton model.
		Charm = - d Delta / dt

		Returns
		-------
		float : measuring the rate of change of delta over the passage of time
		"""
		d1 = self.get_d1()
		d2 = self.get_d2()
		if self.option_type == 'put':
			return -self.q * np.exp(-self.q * self.T) * st.norm.cdf(-d1) * d1 -\
				np.exp(-self.q * self.T) * st.norm.pdf(d1) * (2*(self.rf - self.q) * self.T - d2 * self.sigma * self.T**0.5) /\
					(2 * self.T * self.sigma * self.T**0.5)

	def vomma(self) -> float:
		""" Calculates vomma from Black Scholes Merton model.
		Vomma = d^2V / d sigma^2

		Returns
		-------
		float : measuring the rate of change to vega as the volatility changes
		"""
		d1 = self.get_d1()
		d2 = self.get_d2()
		vega = self.vega()
		return vega * d1 * d2 / self.sigma

	def veta(self) -> float:
		""" Calculates veta from Black Scholes Merton model.
		Veta = d Vega / dt

		Returns
		-------
		float : measuring the rate of change to vega with respect to the
			passage of time
		"""
		d1 = self.get_d1()
		d2 = self.get_d2()
		return -self.So * np.exp(-self.q * self.T) * st.norm.pdf(d1) * self.T**0.5 *\
			(self.q + ((self.rf - self.q)*d1/(self.sigma * self.T**0.5)) - (1+d1*d2)/(2*self.T))
	
	def speed(self) -> float:
		""" Calculates speed from Black Scholes Merton model.
		Speed = d Gamma / dS

		Returns
		-------
		float : measuring the rate of change in Gamma with respect to changes
			in the underlying price
		"""
		d1 = self.get_d1()
		gamma = self.gamma()
		return -(gamma / self.So) * (d1/(self.sigma * self.T**0.5) + 1)
	
	def zomma(self) -> float:
		""" Calculates zomma from Black Scholes Merton model.
		Zomma = d Gamma / d sigma

		Returns
		-------
		float : measuring the rate of change in Gamma with respect to changes
			in volatility
		"""
		d1 = self.get_d1()
		d2 = self.get_d2()
		gamma = self.gamma()
		return gamma * ((d1*d2-1)/self.sigma)
	
	def color(self) -> float:
		""" Calculates color from Black Scholes Merton model.
		Color = d Gamma / dt

		Returns
		-------
		float : measuring the rate of change in Gamma with respect to passage
		of time
		"""
		d1 = self.get_d1()
		d2 = self.get_d2()
		return -np.exp(-self.q * self.T) * (st.norm.pdf(d1) / (2 * self.So * self.T * self.sigma * self.T**0.5)) *\
			(2 * self.q * self.T + 1 + (2*(self.r - self.q)*self.T - d2*self.sigma * self.T**0.5) * (d1/(self.sigma * self.T**0.5)))

	def ultima(self) -> float:
		""" Calculates ultima from Black Scholes Merton model.
		Ultima = d Vomma / d sigma

		Returns
		-------
		float : measuring the rate of change of vomma with respecto to 
			change in volatility
		"""
		d1 = self.get_d1()
		d2 = self.get_d2()
		vega = self.vega()
		return (-vega/self.sigma**2) * (d1*d2*(1-d1*d2) + d1**2 + d2**2)
	
	def dual_delta(self) -> float:
		""" Calculates dual delta from Black Scholes Merton model.
		Dual Delta = dV / dK

		Returns
		-------
		float : the probability of an option finishing In The Money
		"""
		d2 = self.get_d2()
		if self.option_type == 'put':
			return np.exp(-self.rf * self.T) * st.norm.cdf(-d2)
		else:
			return np.exp(-self.rf * self.T) * st.norm.cdf(d2)
	
	def dual_gamma(self) -> float:
		""" Calculates dual gamma from Black Scholes Merton model.
		Dual Gamma = d^2V / dK^2

		Returns
		-------
		float : measuring the rate of change of dual delta.
		"""
		d2 = self.get_d2()
		return np.exp(-self.rf * self.T) * st.norm.pdf(d2) / (self.K * self.sigma * self.T**0.5)

	@overrides(Option)
	def price(self) -> float:
		""" Calculates the fair price of an European option using the
		Black Scholes Merton model.

		Returns
		-------
		float : the fair price of the option
		"""
		d1 = self.get_d1()
		d2 = self.get_d2()
		if self.option_type == 'put':
			return self.K * np.exp(-self.rf * self.T) * st.norm.cdf(-d2) -\
                 self.So * np.exp(-self.q * self.T) * st.norm.cdf(-d1)
		else:
			return self.So * np.exp(-self.q * self.T) * st.norm.cdf(d1) - \
                self.K * np.exp(-self.rf * self.T) * st.norm.cdf(d2)


class AssetOrNothing(Option):
	""" Model representing an Asset or Nothing option.

	Attributes
	----------
		So : float
			spot price of the underlying asset
		K : float
			strike price
		rf : float
			continuous risk free rate
		sigma : float
			volatility of the underlying asset
		T : float
			time to maturity of the option
		option_type : str, valid ['call', 'put'], default 'call'
			option type
		q : float, default 0
			expected dividend yield rate

	Methods
	-------
	price : float
		fair price of the option
	"""

	def __init__(self, So, K, rf, sigma, T, option_type='call', q=0) -> None:
		"""
		Constructs attributes for the option class.

		Parameters
		----------
			So : float
				spot price of the underlying asset
			K : float
				strike price
			rf : float
				continuous risk free rate
			sigma : float
				volatility of the underlying asset
			T : float
				time to maturity of the option
			option_type : str, valid ['call', 'put'], default 'call'
				option type
			q : float, default 0
				expected dividend yield rate
		
		Returns
		-------
		None
		"""
		super().__init__(So, K, rf, sigma, T, option_type, q)

	@overrides(Option)
	def price(self) -> float:
		""" Calculates the fair price of an Asset or Nothing option.

		Returns
		-------
		float : the fair price of the option
		"""
		d2 = self.get_d2()
		if self.option_type == 'put':
			return self.So * np.exp(-self.q * self.T) * st.norm.cdf(-d2)
		else:
			return self.So * np.exp(-self.q * self.T) * st.norm.cdf(d2)

class CashOrNothing(Option):
	""" Model representing a Cash or Nothing option.

	Attributes
	----------
		So : float
			spot price of the underlying asset
		K : float
			strike price
		rf : float
			continuous risk free rate
		sigma : float
			volatility of the underlying asset
		T : float
			time to maturity of the option
		option_type : str, valid ['call', 'put'], default 'call'
			option type
		q : float, default 0
			expected dividend yield rate
		C : float, default 0
			Cash return of the option

	Methods
	-------
	price : float
		fair price of the option
	"""

	def __init__(self, So, K, rf, sigma, T, option_type='call', q=0, C=0) -> None:
		"""
		Constructs attributes for the option class.

		Parameters
		----------
			So : float
				spot price of the underlying asset
			K : float
				strike price
			rf : float
				continuous risk free rate
			sigma : float
				volatility of the underlying asset
			T : float
				time to maturity of the option
			option_type : str, valid ['call', 'put'], default 'call'
				option type
			q : float, default 0
				expected dividend yield rate
			C : float, default 0
				cash payment if the option is exercised
		
		Returns
		-------
		None
		"""
		super().__init__(So, K, rf, sigma, T, option_type, q)
		self.C = C

	@overrides(Option)
	def price(self) -> float:
		""" Calculates the fair price of a Cash or Nothing option.

		Returns
		-------
		float : the fair price of the option
		"""
		d2 = self.get_d2()
		if self.option_type == 'put':
			return self.C * np.exp(-self.rf * self.T) * st.norm.cdf(-d2)
		else:
			return self.C * np.exp(-self.rf * self.T) * st.norm.cdf(-d2)

class American(Option):
	""" Model representing an American option.

	Attributes
	----------
		So : float
			spot price of the underlying asset
		K : float
			strike price
		rf : float
			continuous risk free rate
		sigma : float
			volatility of the underlying asset
		T : float
			time to maturity of the option
		option_type : str, valid ['call', 'put'], default 'call'
			option type
		q : float, default 0
			expected dividend yield rate
		C : float, default 0
			Cash return of the option

	Methods
	-------
	price : float
		fair price of the option
	"""

	def __init__(self, So, K, rf, sigma, T, option_type='call', q=0, n=1) -> None:
		"""
		Constructs attributes for the option class.

		Parameters
		----------
			So : float
				spot price of the underlying asset
			K : float
				strike price
			rf : float
				continuous risk free rate
			sigma : float
				volatility of the underlying asset
			T : float
				time to maturity of the option
			option_type : str, valid ['call', 'put'], default 'call'
				option type
			q : float, default 0
				expected dividend yield rate
			n : int, default 1
				steps of the decision tree
		
		Returns
		-------
		None
		"""
		super().__init__(So, K, rf, sigma, T, option_type, q)
		self.n = int(n)

	@overrides(Option)
	def price(self) -> float:
		""" Calculates the fair price of an American option using the binomial
		options pricing model.

		Returns
		-------
		float : the fair price of the option
		"""
		dt = self.T / self.n
		u = np.exp(self.sigma * dt**0.5)
		d = 1 / u
		p = (np.exp(self.rf * dt) - d) / (u - d)

		stockval = np.zeros((self.n+1, self.n+1), np.float64)
		optionval = np.zeros((self.n+1, self.n+1), np.float64)

		stockval[0, 0] = self.So

		for i in range(1, self.n+1):
			stockval[i, 0] = stockval[i-1, 0] * u
			for j in range(1, i+1):
				stockval[i, j] = stockval[i-1, j-1] * d

		for j in range(self.n+1):
			if self.option_type == 'put':
				optionval[self.n, j] = np.maximum(0, self.K - stockval[self.n, j])
			else:
				optionval[self.n, j] = np.maximum(0, stockval[self.n, j] - self.K)
		
		for i in range(self.n-1, -1, -1):
			for j in range(i+1):
				if self.option_type == 'put':
					optionval[i, j] = np.maximum(np.maximum(0, self.K - stockval[i, j]), 
					np.exp(-self.rf * dt) * (p * optionval[i+1, j] + (1-p)* optionval[i+1, j+1]))
				else:
					optionval[i, j] = np.maximum(np.maximum(0, stockval[i, j] - self.K), 
					np.exp(-self.rf * dt) * (p * optionval[i+1, j] + (1-p)* optionval[i+1, j+1]))
		return optionval[0, 0]
