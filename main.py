#!/usr/bin/python
# -*- coding: utf-8 -*-
from telnetlib import X3PAD
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, exp, square, log, log10, log2, ceil
from pysr import PySRRegressor

arrayLength = 200

hashRate = None
dificulty = None
averageBlockSize = None
minerRevenue = None
usdTradeVolume = None
transactionConfirmTime = None
costPerTransaction = None
transactionsPerBlock = None
marketCap = None
bitcoinPrice = None
cpi = None
fearResponse = None
ethResponse = None


def sanitize_periods(jsonData, start, end):
    return np.float_(np.array(jsonData.json()['dataset']['data'])[start:end, 1])


def sanitize(jsonData, arrayLength):
    return np.float_((np.array(jsonData.json()['dataset']['data'])[:,1])[:arrayLength])


def sanitize_fear(jsonData, start, end):
    data = jsonData['data']
    values = [int(entry['value']) for entry in data]
    return values[start:end]


def sanitize_cpi(jsonData, start, end):
    data = jsonData['dataset']['data']
    values = [row[1] for row in data]
    return np.float_(values[start:end])


def sanitize_eth(jsonData, start, end):
    data = jsonData['dataset']['data']
    mid_prices = [row[3] for row in data]
    return np.float_(mid_prices[start:end])


 # ##### USING DIFFERENT PERIODS FOR MODEL TRAINING

def get_data_for_different_date_periods():
    periods = [(0, 200), (400, 600), (600, 800), (800, 1000), (1200,
               1400)]
    arrayLenPeriods = periods[-1][1] - periods[0][0]

    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9,x10,x11,x12, Y = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    
    cpiLen = ceil(arrayLenPeriods / 30)
    sanitized_cpi = np.array(sanitize_cpi(cpi, 0, int(cpiLen)),
                             dtype=float)
    cpiArr = []
    for cpiItem in sanitized_cpi:
        repeated_items = np.repeat(cpiItem, 30)  # Repeat CPI item 30 times
        cpiArr.extend(repeated_items)

    for (start, end) in periods:
        x0.extend(sanitize_periods(hashRate, start, end))
        x1.extend(sanitize_periods(dificulty, start, end))
        x2.extend(sanitize_periods(averageBlockSize, start, end))
        x3.extend(sanitize_periods(minerRevenue, start, end))
        x4.extend(sanitize_periods(usdTradeVolume, start, end))
        x5.extend(sanitize_periods(transactionConfirmTime, start, end))
        x6.extend(sanitize_periods(costPerTransaction, start, end))
        x7.extend(sanitize_periods(transactionsPerBlock, start, end))
        x8.extend(sanitize_periods(marketCap, start, end))
        x9.extend(sanitize_periods(bitcoinPrice, start, end))
        x11.extend(sanitize_fear(fearResponse, start, end))
        x12.extend(sanitize_eth(ethResponse, start, end))
        Y.extend(sanitize_periods(bitcoinPrice, start, end))
        x10.extend(cpiArr[start:end])

    new_price_yesterday = [0]

    for i in range(len(x9) - 1):
        new_price_yesterday.append(x9[i])

    x9 = new_price_yesterday

    X = np.c_[x0, x1, x2, x3, x4, x5, x6, x7, x10, x11, x12]

    return (X, Y, arrayLenPeriods)


###### USING LAST <ARRAY-LENGTH> DAYS FOR MODEL TRAINING

def get_data_for_last_z_days():

    arrayLength = 200

    x0 = np.array(sanitize(hashRate, arrayLength), dtype=float)
    x1 = np.array(sanitize(dificulty, arrayLength), dtype=float)
    x2 = np.array(sanitize(averageBlockSize, arrayLength), dtype=float)
    x3 = np.array(sanitize(minerRevenue, arrayLength), dtype=float)
    x4 = np.array(sanitize(usdTradeVolume, arrayLength), dtype=float)
    x5 = np.array(sanitize(transactionConfirmTime, arrayLength),
                  dtype=float)
    x6 = np.array(sanitize(costPerTransaction, arrayLength),
                  dtype=float)
    x7 = np.array(sanitize(transactionsPerBlock, arrayLength),
                  dtype=float)
    x8 = np.array(sanitize(marketCap, arrayLength), dtype=float)
    x11 = np.array(sanitize_fear(fearResponse, 0, arrayLength),
                   dtype=float)
    x12 = np.array(sanitize_eth(ethResponse, 0, arrayLength), dtype=float)

    Y = sanitize(bitcoinPrice, arrayLength)

    # EKONOMSKI FAKTORI
    cpiLen = ceil(arrayLength / 30)
    sanitized_cpi = np.array(sanitize_cpi(cpi, 0, int(cpiLen)),
                             dtype=float)
    x10 = []

    for cpiItem in sanitized_cpi:
        repeated_items = np.repeat(cpiItem, 30)  # Repeat CPI item 30 times
        x10.extend(repeated_items)

    x10 = x10[:arrayLength]

    # ## ENABLE USING PRICE FROM THE DAY BEFORE

    x9 = sanitize(bitcoinPrice, arrayLength + 1)
    new_price_yesterday = []

    for i in range(len(x9) - 1):
        new_price_yesterday.append(x9[i + 1])

    x9 = new_price_yesterday

    X = np.c_[x0, x1, x2, x3, x4, x5, x6, x7, x10, x11, x12]

    return (X, Y)


######## MAIN ############

hashRate = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/HRATE.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI')
dificulty = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/DIFF.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
averageBlockSize = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/AVBLS.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
minerRevenue = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/MIREV.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
usdTradeVolume = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/TRVOU.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
transactionConfirmTime = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/ATRCT.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
costPerTransaction = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/CPTRA.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
transactionsPerBlock = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/NTRBL.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
marketCap = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/MKTCP.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
bitcoinPrice = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/MKPRU.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
cpi = requests.get('https://data.nasdaq.com/api/v3/datasets/RATEINF/CPI_USA.json?api_key=9o96oy7ZEy3hhZt1xb42')
fearResponse = requests.get('https://api.alternative.me/fng/?limit=0')
ethResponse = requests.get('https://data.nasdaq.com/api/v3/datasets/BITFINEX/ETHUSD.json?api_key=9o96oy7ZEy3hhZt1xb42')

fearResponse = fearResponse.json()
ethResponse = ethResponse.json()
cpi = cpi.json()

(X, Y, arrayLenPeriods) = get_data_for_different_date_periods()
# (X, Y) = get_data_for_different_date_periods()

x0, x1, x2, x3, x4, x5, x6, x7, x10, x11, x12 = X.T

model = PySRRegressor(
    procs=4,
    populations=4,
    # ^ 2 populations per core, so one is always running.
    population_size=30,
    # ^ Slightly larger populations, for greater diversity.
    ncyclesperiteration=500, 
    # ^ Generations between migrations.
    niterations=70,  # Increase for precision, reduce for time saving. 100000 will run forever
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
        # Stop early if we find a good and simple equation
    ),
    timeout_in_seconds=60 * 60 * 24,
    # ^ Alternatively, stop after 24 hours have passed.
    maxsize=50, #50 default, will reduce to 25
    # ^ Allow greater complexity.
    maxdepth=10,
    # ^ But, avoid deep nesting.
    binary_operators=["*", "+", "-", "/"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "square",
        "log",
        "log10",
        "log2",
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
    # loss="loss(prediction, target) = abs((prediction - target) / target)",
    loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(X,Y)
# model.predict(X)


result = eval(str(model.sympy()))


plt.figure()
plt.title('Cijene bitcoina u periodima od' + str(arrayLenPeriods) + ' dana (USD)')
# plt.title('Cijene bitcoina u posljednih' + str(arrayLength) + ' dana (USD)')
plt.plot(Y)
plt.plot(result, 'r', alpha=0.6)
plt.legend(['Stvarna cijena bitcoina', 'Cijena dobivena sa funkcijom iz simboličke regresije'])
plt.gca().invert_xaxis()
plt.show() 
