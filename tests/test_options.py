from optiontools import options

def test_european_call():
	opt = options.European(50, 52, 0.05, 0.257842, 1, 'call')
	assert opt.price() == 5.373964652786658

def test_european_put():
	assert options.European(50, 52, 0.05, 0.257842, 1, 'put').price() == 4.837894726823784

def test_european_delta():
	assert options.European(50, 52, 0.05, 0.257842, 1, 'put').delta() == -0.4322193008857996 and \
		options.European(50, 52, 0.05, 0.257842, 1, 'call').delta() == 0.5677806991142005

def test_european_gamma():
	assert options.European(50, 52, 0.05, 0.257842, 1, 'put').gamma() == 0.03049699663533816

def test_european_vega():
	assert options.European(50, 52, 0.05, 0.257842, 1, 'put').vega() == 19.658516516122155

def test_european_rho():
	assert options.European(50, 52, 0.05, 0.257842, 1, 'put').rho() == -26.448859771113764 and \
		options.European(50, 52, 0.05, 0.257842, 1, 'call').rho() == 23.015070302923366

def test_european_theta():
	assert options.European(50, 52, 0.05, 0.257842, 1, 'put').theta() == -1.2119526192192964 and \
		options.European(50, 52, 0.05, 0.257842, 1, 'call').theta() == -3.685149122921153

def test_american_call():
	opt = options.American(50, 52, 0.05, 0.257842, 1, 'call', n=2)
	assert opt.price() == 5.2154649693543975

def test_american_put():
	assert options.American(50, 52, 0.05, 0.257842, 1, 'put', n=2).price() == 5.275953021645895
