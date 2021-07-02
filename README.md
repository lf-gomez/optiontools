# Optiontools
## Tools for valuating financial derivatives

Option Tools contains methods for valuating different types of financial derivatives, such as American, European, and Binary Options (Asset-or-Nothing, Cash-or-Nothing).


## Installation

To install the package:
```sh
pip install optiontools
```

## Example
Valuating a European put option:

```
>>> from optiontools.options import European
>>> opt = European(So=50, K=52, rf=0.01, sigma=0.15, T=1, option_type='call')
>>> opt.price()
2.349553922014252
>>> opt.delta()
0.4523189078212078
```

## Available Classes

 - European (*including greeks*)
 - American
 - AssetOrNothing
 - CashOrNothing


## License

MIT