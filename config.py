# config.py
DATA_DIR = "./data"
AMFI_FILE = f"{DATA_DIR}/amfi_flows.csv"            # (optional) monthly flows
INDEX_PRICES_FILE = f"{DATA_DIR}/index_prices.csv"  # daily index prices (Date,index1,index2,...)
VIX_FILE = f"/mnt/data/INDIA VIX_minute.csv"        # file you uploaded — pipeline tries to use it
PUBLICATION_TIME = "15:30"  # 3:30 PM IST cut-off for weekly features
RANDOM_SEED = 42
WEEKDAY_CUTOFF = "FRIDAY"  # canonical weekly decision moment
TOP_K = 3
TRANSACTION_COST_BPS = 20  # 20 bps default
