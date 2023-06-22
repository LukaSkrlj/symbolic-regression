import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, exp, square
import sys

np.set_printoptions(threshold=sys.maxsize)

#Each dataset size (column) before averaging
arrayLength = 100
#Average every 'averageCount' items in dataset
averageCount = 5
#Number of pastDays in the past for each averaged dataset
pastDays = 5

if arrayLength % averageCount != 0:
    raise Exception("Average count must be multiplier of array length")

def sanitize(jsonData, start=0, end=arrayLength):
    return np.average(np.float_(np.array(jsonData.json()['dataset']['data'])[:,1][start:end]).reshape(-1, averageCount), axis=1)

hashRate = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/HRATE.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI')
difficulty = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/DIFF.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
averageBlockSize = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/AVBLS.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
minerRevenue = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/MIREV.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')

bitcoinPrice = requests.get('https://data.nasdaq.com/api/v3/datasets/BCHAIN/MKPRU.json?api_key=9o96oy7ZEy3hhZt1xb42&fbclid=IwAR2bTZqq23hglhdRtpV4UrmYq02giUEvtHAL3qyJySLJA5Y9cmkatT403VI%27')
datasets = [hashRate, difficulty, averageBlockSize]

averagedArrayLength = arrayLength // averageCount
X = np.array([[] for a in range(averagedArrayLength)])
print(X)

for i in range(pastDays):
    start = i * averageCount
    for dataset in datasets:
        print("dataset")
        sanitizedDataset = np.array([sanitize(dataset)]).transpose()
        print(sanitizedDataset)
        X = np.hstack((X,sanitizedDataset))
        print(X)
        
Y = sanitize(bitcoinPrice)
print(X,Y)
from pysr import PySRRegressor

model = PySRRegressor(
    procs=4,
    populations=8,
    # ^ 2 populations per core, so one is always running.
    population_size=50,
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
    maxdepth=10,
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
    loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(X,Y)

# result = (((29182.05778922693 - X2) - 1/(square(sin(sin(sin(pow(-1.2891089844873154 * X2,3))))))) - (((4482.529706112431 * exp(cos(square(X1 - -0.31671563304042566)))) * square(cos(square(1.7327020067419614 * X1) + X2))) + 1/(square(sin(sin(pow(-1.2891089844873154 * X2, 3)))))))
# plt.figure()
# plt.title('Cijene bitcoina u posljednjih' + arrayLength + 'dana (USD)')
# plt.plot(Y)
# plt.plot(result, 'r', alpha=0.6)
# plt.legend(['Stvarna cijena bitcoina', 'Cijena dobivena sa funkcijom iz simboličke regresije'])
# plt.show()
