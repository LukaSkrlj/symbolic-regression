from telnetlib import X3PAD
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, exp, square
from pysr import PySRRegressor


def sanitize(jsonData, arrayLength):
    return np.float_(np.array(jsonData.json()['dataset']['data'])[:, 1][:arrayLength])


hashRate = requests.get(
    'https://data.nasdaq.com/api/v3/datasets/BCHAIN/HRATE.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid'
    '=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI')
dificulty = requests.get(
    'https://data.nasdaq.com/api/v3/datasets/BCHAIN/DIFF.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid'
    '=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
averageBlockSize = requests.get(
    'https://data.nasdaq.com/api/v3/datasets/BCHAIN/AVBLS.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid'
    '=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
minerRevenue = requests.get(
    'https://data.nasdaq.com/api/v3/datasets/BCHAIN/MIREV.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid'
    '=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
usdTradeVolume = requests.get(
    'https://data.nasdaq.com/api/v3/datasets/BCHAIN/TRVOU.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid'
    '=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
transactionConfirmTime = requests.get(
    'https://data.nasdaq.com/api/v3/datasets/BCHAIN/ATRCT.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid'
    '=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
costPerTransaction = requests.get(
    'https://data.nasdaq.com/api/v3/datasets/BCHAIN/CPTRA.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid'
    '=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
transactionsPerBlock = requests.get(
    'https://data.nasdaq.com/api/v3/datasets/BCHAIN/NTRBL.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid'
    '=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
marketCap = requests.get(
    'https://data.nasdaq.com/api/v3/datasets/BCHAIN/MKTCP.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid'
    '=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
bitcoinPrice = requests.get(
    'https://data.nasdaq.com/api/v3/datasets/BCHAIN/MKPRU.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid'
    '=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')

arrayLength = 200

x0 = sanitize(hashRate, arrayLength)
x1 = sanitize(dificulty, arrayLength)
x2 = sanitize(averageBlockSize, arrayLength)
x3 = sanitize(minerRevenue, arrayLength)
x4 = sanitize(usdTradeVolume, arrayLength)
x5 = sanitize(transactionConfirmTime, arrayLength)
x6 = sanitize(costPerTransaction, arrayLength)
x7 = sanitize(transactionsPerBlock, arrayLength)
x8 = sanitize(marketCap, arrayLength)

Y = sanitize(bitcoinPrice, arrayLength)

### ENABLE USING PRICE FROM THE DAY BEFORE
x9 = sanitize(bitcoinPrice, arrayLength+1)
new_price_yesterday = []

for i in range(len(x9)-1):
    new_price_yesterday.append(x9[i+1])

x9 = new_price_yesterday

X = np.c_[x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]

model = PySRRegressor(
    procs=4,
    populations=8,
    # ^ 2 populations per core, so one is always running.
    population_size=130,
    # ^ Slightly larger populations, for greater diversity.
    ncyclesperiteration=500, 
    # ^ Generations between migrations.
    niterations=100,  # Increase for precision, reduce for time saving. 100000 will run forever
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
        # Stop early if we find a good and simple equation
    ),
    timeout_in_seconds=60 * 60 * 24,
    # ^ Alternatively, stop after 24 hours have passed.
    maxsize=50, #50 default, will reduce to 25
    # ^ Allow greater complexity.
    maxdepth=15,
    # ^ But, avoid deep nesting.
    binary_operators=["*", "+", "-", "/"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "square",
        "cube",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    constraints={
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
        "exp": 9,
    },
    # ^ Limit the complexity within each argument.
    # "inv": (-1, 9) states that the numerator has no constraint,
    # but the denominator has a max complexity of 9.
    # "exp": 9 simply states that `exp` can only have
    # an expression of complexity 9 as input.
    nested_constraints={
        "square": {"square": 1, "cube": 1, "exp": 0},
        "cube": {"square": 1, "cube": 1, "exp": 0},
        "exp": {"square": 1, "cube": 1, "exp": 0},
    },
    # "square(exp(x))" is not allowed, since "square": {"exp": 0}.
    complexity_of_operators={"/": 2, "exp": 3},
    # ^ Custom complexity of particular operators.
    complexity_of_constants=2,
    # ^ Punish constants more than variables
    select_k_features=4,
    # ^ Train on only the 4 most important features
    progress=True,
    # ^ Can set to false if printing to a file.
    weight_randomize=0.1,
    # ^ Randomize the tree much more frequently
    cluster_manager=None,
    # ^ Can be set to, e.g., "slurm", to run a slurm
    # cluster. Just launch one script from the head node.
    precision=64,
    # ^ Higher precision calculations.
    # warm_start=True,
    # ^ Start from where left off.
    # turbo=True,
    # ^ Faster evaluation (experimental)
    loss="loss(prediction, target) = abs((prediction - target) / target)",
    # ^ Custom loss function (julia syntax)
)

model.fit(X,Y)

x0 = np.array(x0, dtype=float)
x1 = np.array(x1, dtype=float)
x3 = np.array(x3, dtype=float)
x4 = np.array(x4, dtype=float)
x5 = np.array(x5, dtype=float)
x6 = np.array(x6, dtype=float)
x7 = np.array(x7, dtype=float)
x8 = np.array(x8, dtype=float)
x9 = np.array(x9, dtype=float)

result = eval(str(model.sympy()))


plt.figure()
plt.title('Cijene bitcoina u posljednjih ' + str(arrayLength) + ' dana (USD)')
plt.plot(Y)
plt.plot(result, 'r', alpha=0.6)
plt.legend(['Stvarna cijena bitcoina', 'Cijena dobivena sa funkcijom iz simboličke regresije'])
plt.gca().invert_xaxis()
plt.show() 
