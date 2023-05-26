import os
import time
import datetime
import functools
import asyncio
import random
import pandas as pd
from itertools import groupby
# import network as net
from datetime import datetime, timedelta
from functools import partial
from pytz import timezone
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError, TimeFrame
from alpaca_trade_api.stream import Stream
from apscheduler.schedulers.background import BackgroundScheduler

# Set up the Alpaca Trade API
API_KEY = ''
SECRET_KEY = ''
BASE_URL = 'https://paper-api.alpaca.markets'
TOO_MANY_REQ_CODE = 429
TRADE_DATA = {}
TRADE_DATA_SIZE = 28_800
FREQ_DELTA_REQ = 3000
PRICE_DELTA_REQ = 0.1
PRICE_DELTA_SELL_MIN = 0
TRAILING_ON = True
TRAILING_STOP_PERCENT = 5
GAIN_DELTA_REQ = 20
MINUTE_TRADE_COUNT_REQ = 10
MINUTE_VOLUME_REQ  = 1000
TRADE_PREMARKET = True

trade_client = tradeapi.REST(API_KEY, SECRET_KEY, base_url=BASE_URL, api_version='v2')
# Initiate Class Instance
stream = Stream(API_KEY, SECRET_KEY, base_url=BASE_URL, data_feed='sip')

# Define a decorator to handle APIError and AttributeError exceptions
def handle_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIError as api_err:
            if api_err.status_code == TOO_MANY_REQ_CODE:
                print(f"Too many API requests. Sleeping for 1 minute...")
                time.sleep(60)
            else:
                print(f"API error with status code {api_err.status_code} thrown")
                print(f"Cannot obtain snapshot for {args}")
            print("Continuing...")
        except AttributeError as attrib_err:
            print(f"Attribute error thrown: {attrib_err}, for symbols: {args}")
            print("Continuing...")
        except Exception as e:
            if func.__name__ == 'market_buy':
                print('Failed to buy stock, continuing...')
            elif func.__name__ == 'market_sell':
                print('Failed to sell stock, continuing...')
            else:
                print(f'Error {e}')
    return wrapper

@handle_exceptions
def market_buy(symbol, qty, time_in_force='ioc'):

    print(f'Buying {qty} shares of {symbol}')

    order = trade_client.submit_order(
        symbol=symbol,
        qty=qty,
        side='buy',
        type='market',
        time_in_force=time_in_force
    )

    return order

@handle_exceptions
def market_sell(symbol, qty, time_in_force='gtc'):

    print(f'Selling {qty} shares of {symbol}')
    order = trade_client.submit_order(
        symbol=symbol,
        qty=qty,
        side='sell',
        type='market',
        time_in_force=time_in_force
    )

    return order

def ext_hour_buy(symbol, limit_price, qty, time_in_force='day'):

    print(f'Buyin {qty} shares of {symbol} at {limit_price}')
    order = trade_client.submit_order(
        symbol=symbol,
        qty=qty,
        side='buy',
        type='limit',
        time_in_force=time_in_force,
        extended_hours=True,
        limit_price=limit_price
    )

    return order

def ext_hour_sell(symbol, limit_price, qty, time_in_force='day'):

    print(f'Selling {qty} shares of {symbol} at {limit_price}')
    order = trade_client.submit_order(
        symbol=symbol,
        qty=qty,
        side='sell',
        type='limit',
        time_in_force=time_in_force,
        extended_hours=True,
        limit_price=limit_price
    )

    return order

def trailing_sell(symbol, qty, percent, time_in_force='day'):
    
    print(f'Submitting trailing sell order for {qty} shares of {symbol} at {percent}%')
    order = trade_client.submit_order(
        symbol=symbol,
        qty=qty,
        side='sell',
        type='trailing_stop',
        time_in_force=time_in_force,
        trail_percent=percent,
    )

    return order


async def handle_trade_updates(data):
    if data.event == 'fill' or data.event == 'partial_fill':
        symbol = data.order['symbol']
        pos_qty = float(data.position_qty)
        print(f'Calling trade handler {symbol} : {pos_qty}')
        TRADE_DATA[symbol]['position_qty'] = pos_qty

        if TRAILING_ON and data.order['side'] == 'buy':
            trailing_sell(data.order['symbol'], pos_qty, TRAILING_STOP_PERCENT)

# ts = trade_client.get_clock().timestamp.timestamp()
async def trade_callback(opening_timestamp, trade):
    # opening_timestamp = ts
    if trade and trade.timestamp:

        trade_timestamp  = trade.timestamp.timestamp()
        seconds_elapsed = int(trade_timestamp - opening_timestamp)
        if seconds_elapsed >= 0 and seconds_elapsed < TRADE_DATA_SIZE:
            curr_data = TRADE_DATA[trade.symbol]['tick_data'][seconds_elapsed]


            if seconds_elapsed >= 1:

                prev_data = TRADE_DATA[trade.symbol]['tick_data'][seconds_elapsed-1]

                if not curr_data['updated']:

                    prev_data['average_price'] /= (prev_data['frequency'] if prev_data['frequency'] else 1)

                    if seconds_elapsed >=2:
                        
                        prev_prev_data = TRADE_DATA[trade.symbol]['tick_data'][seconds_elapsed-2]

                        if prev_prev_data['frequency'] > 0:

                            freq_delta = 100*(prev_data['frequency'] - prev_prev_data['frequency']) / prev_prev_data['frequency']
                            price_delta = 100*(prev_data['average_price'] - prev_prev_data['average_price']) / prev_prev_data['average_price']
                            pos_qty = TRADE_DATA[trade.symbol]['position_qty']
                            # print(f'{trade.symbol}: pos_qty: {pos_qty} freq: {freq_delta} price_delta: {price_delta}')



                            if pos_qty == 0 and freq_delta >= FREQ_DELTA_REQ  and price_delta >= PRICE_DELTA_REQ:

                                # Buy the stock 'ioc'
                                market_buy(trade.symbol, 100)
                                # ext_hour_buy(trade.symbol, round(trade.price*1.002, 2), 100)


                            # elif price_delta < PRICE_DELTA_SELL_MIN and prev_data['frequency'] > 0 and pos_qty > 0:
                            # elif pos_qty > 0 and seconds_elapsed >=3 and (p_price < pp_price < ppp_price or price_delta <= PRICE_DELTA_SELL_MIN):
                        
                                # market_sell(trade.symbol, TRADE_DATA[trade.symbol]['position_qty'])
                                # ext_hour_buy(trade.symbol, round(trade.price*0.1,2), 2, 100)

                    curr_data['updated'] = True

            curr_data['frequency'] += 1
            curr_data['average_price'] += trade.price

# Function to get the market hours
@handle_exceptions
def get_market_hours():
    clock = trade_client.get_clock()
    opening_time = clock.next_open
    closing_time = clock.next_close
    return opening_time, closing_time

# Define a function to get a single stock snapshot and handle exceptions
@handle_exceptions
def get_single_snapshot(symbol):
    snapshot = trade_client.get_snapshot(symbol, 'sip')
    return snapshot

# Define a function to get stock snapshots for a partition of symbols and handle exceptions
@handle_exceptions
def get_snapshots_for_partition(symbols):
    multi_snapshot = trade_client.get_snapshots(symbols, 'sip')
    return multi_snapshot

# Alpaca API cannot handle all symbols at once.
# Divide up stock symbols into smaller lists each containing 1,000 symbols.
# Then, we query Alpaca API once for every 1,000 symbols.
@handle_exceptions
def partition_stocks(stocks):

    num_of_stocks = len(stocks)
    partition_size = 1000
    num_of_partions = num_of_stocks//partition_size
    num_of_remaining_stocks = num_of_stocks % partition_size

    stock_partition = [stocks[i*partition_size:(i+1)*partition_size] for i in range(num_of_partions)]
    # Append remaining symbols if number of symbols was not divisible by 1,000
    if num_of_remaining_stocks:
        stock_partition.append(stocks[num_of_stocks-num_of_remaining_stocks:])
    return stock_partition

@handle_exceptions
def get_stocks():
    stocks = trade_client.list_assets(status='active', asset_class='us_equity')
    return [stock.symbol for stock in stocks if stock.tradable]

@handle_exceptions
def get_snapshots(stock_partition):
    # Contains snapshot for every symbol
    snapshots = []

    # Per 1,000 symbols, query Alpaca API and insert each snapshot in list above.
    for partition in stock_partition:
        multi_snapshot = get_snapshots_for_partition(partition)

        if multi_snapshot is not None:
            snapshots.extend(multi_snapshot.items())
        else:
            for symb in partition:
                snapshot = get_single_snapshot(symb)
                if snapshot is not None:
                    snapshots.extend(snapshot.items()) 
    
    return [snapshot for snapshot in snapshots if snapshot[1] is not None]

@handle_exceptions
def is_gapper(snapshot_data, when='pre-market'):

    if not(snapshot_data.minute_bar and snapshot_data.prev_daily_bar and snapshot_data.daily_bar):
        return False

    prev_price = float
    curr_price = float
    gain_prcnt = float

    if when == 'last-market':

        prev_price = snapshot_data.prev_daily_bar.c
        curr_price = snapshot_data.daily_bar.o
        

    elif when == 'pre-market' or when == 'market':
        prev_price = snapshot_data.daily_bar.c if when == 'pre-market' else snapshot_data.prev_daily_bar.c
        curr_price = snapshot_data.minute_bar.vw
        curr_trade_count = snapshot_data.minute_bar.n
        curr_volume = snapshot_data.minute_bar.v

        if (curr_trade_count < MINUTE_TRADE_COUNT_REQ) or (curr_volume < MINUTE_VOLUME_REQ):
            return False
        
    else:
        return False
    
    gain_prcnt = 100 * (curr_price - prev_price) / prev_price

    return gain_prcnt >= GAIN_DELTA_REQ

# Function to get stocks with 20% or more gains
@handle_exceptions
def get_watchlist(when='pre-market'):

    snapshots = get_snapshots(partition_stocks(get_stocks()))
    
    watch_list = []

    for symb, snapshot_data in snapshots:

        if symb and snapshot_data and is_gapper(snapshot_data, when):

            watch_list.append(symb)


    print(f'Watch list: {watch_list}')
 
    return watch_list

# Function to stream the stocks
@handle_exceptions
def stream_stocks():

    next_opening_time_utc, _ = get_market_hours()
    opening_timestamp = next_opening_time_utc.timestamp()
    trade_callback_with_timestamp = partial(trade_callback, opening_timestamp)
    market_open = trade_client.get_clock().is_open

    print('Getting watchlist...')

    watch_list = get_watchlist('market') if market_open else get_watchlist('pre-market')

    for symb in watch_list:
        TRADE_DATA[symb] = { 'position_qty': 0.0, 'tick_data': [{'frequency': 0, 'average_price': 0.0, 'updated':False} for _ in range(TRADE_DATA_SIZE)] }

    print(f"Streaming stocks with 20% or more gains: {watch_list}")
    # subscribing to event
    stream.subscribe_trade_updates(handle_trade_updates)
    stream.subscribe_trades(trade_callback_with_timestamp, *watch_list)
    stream.run()

class ArtificialTrade:

    def __init__(self, t):
        self.t = t
        self.x = ""
        self.p = 0.0
        self.s = 0
        self.c = []
        self.i = 0
        self.z = ""
        

# @handle_exceptions
def learn():


    stock_partition = partition_stocks(get_stocks())

    # Define the start and end dates
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=60)  # Go a bit further back to account for weekends and holidays

    daily_gappers = []


        # Loop over the stock symbols
    for stocks in stock_partition:

        # Get the historical day_bars for stocks
        day_bars = trade_client.get_bars(symbol=stocks, timeframe=TimeFrame.Day, start=start_date, end=end_date)
        
        for prev_bar, open_bar in zip(day_bars[:-1], day_bars[1:]):
    
            if prev_bar.S != open_bar.S:
                continue

            close_price = prev_bar.c
            open_price = open_bar.o
            gap_delta = 100*(open_price - close_price) / close_price

            if gap_delta >= 20 and open_bar.v >= 1_000_000:
                daily_gappers.append({'symbol':open_bar.S, 'timestamp':open_bar.t, 'o':open_price, 'c':close_price})

    daily_gappers = sorted(daily_gappers, key=lambda x: x['timestamp'].date())

    print('making netinput and netoutput...')

    for day_gapper in daily_gappers:

        symbol = day_gapper['symbol']

        y = day_gapper['timestamp'].year
        m = day_gapper['timestamp'].month
        d = day_gapper['timestamp'].day
        start_hour = 13
        end_hour = 15
        minute = 30


        start_time = pd.Timestamp(year=y, month=m, day=d, hour=start_hour, minute=minute).tz_localize(timezone('UTC'))
    
        end_time = pd.Timestamp(year=y, month=m, day=d, hour=end_hour, minute=minute).tz_localize(timezone('UTC'))

        print(symbol, start_time)

        total_seconds = (end_hour - start_hour)*3600

        trades = trade_client.get_trades(symbol= symbol, start=day_gapper['timestamp'].date(), end=day_gapper['timestamp'].date())

        if len(trades) < 0.5*total_seconds:
            continue

        for trade in trades:
            trade.t = pd.to_datetime(trade.t)

        trades = [trade for trade in trades if trade.t >= start_time and trade.t <= end_time]

        # Create a list of all timestamps from start_time to end_time
        timestamps = [start_time + timedelta(seconds=x) for x in range(total_seconds + 1)]

        # Create a dictionary with timestamps as keys for quick lookup
        tradegroups = {key: list(group) for key, group in groupby(trades, key=lambda x: x.t.floor('S'))}

        stock_data = {
                'vwap': 0.0, 
                'pxv': 0.0,
                'volume': 0,
                'frequency': 0.0,
                'trades': 0,
                'high': 0.0,
                'low': float('inf'),
                'second_aggregates': 
                [
                    {                        
                        'frequency': 0.0,
                        'trades': 0,
                        'volume': 0,
                        'tot_price': 0.0,
                        'avg_price': 0.0,
                        'high': 0.0,
                        'low': 0.0,
                        'close': 0.0       
                    } for _ in range(total_seconds + 1)
                ] 
            } 

        netinput = []
        netoutput = []
        timelist = []

        # Go through all timestamps and add either real or artificial trades
        for timestamp in timestamps:

            tradegroup = tradegroups.get(timestamp) or [ArtificialTrade(timestamp)]

            secfreq = 0 if type(tradegroup[0]) == ArtificialTrade else len(tradegroup) 

            secvol = sum(trade.s for trade in tradegroup)

            sectotprice = sum(trade.p for trade in tradegroup)

            secavgprice = sectotprice / (secfreq or 1)

            sechigh = max(trade.p for trade in tradegroup)

            seclow = float('inf') if type(tradegroup[0]) == ArtificialTrade else min(trade.p for trade in tradegroup)

            secclose = tradegroup[-1].p

            aggrsecindx = int((tradegroup[0].t - start_time).total_seconds())

            curraggrsec = stock_data['second_aggregates'][aggrsecindx]

            secfrommin = aggrsecindx % 60

            currminindx = aggrsecindx - (aggrsecindx % 60)

            prevaggrsec = None if not secfrommin else stock_data['second_aggregates'][aggrsecindx - 1]

            prevmin = None if not currminindx else stock_data['second_aggregates'][currminindx-1]

        

            stock_data['pxv'] += sum(trade.p * trade.s for trade in tradegroup)

            stock_data['volume'] += secvol

            stock_data['vwap'] = stock_data['pxv'] / (stock_data['volume'] or 1)

            stock_data['trades'] += secfreq

            stock_data['frequency'] = stock_data['trades'] / (aggrsecindx + 1)

            stock_data['high'] = max(stock_data['high'], sechigh)

            stock_data['low'] = min(stock_data['low'], seclow)

            curraggrsec['trades'] = secfreq if not prevaggrsec else secfreq + prevaggrsec['trades']

            curraggrsec['frequency'] = curraggrsec['trades'] / (secfrommin + 1)

            curraggrsec['volume'] = secvol if not prevaggrsec else secvol + prevaggrsec['volume']

            curraggrsec['tot_price'] = sectotprice if not prevaggrsec else sectotprice + prevaggrsec['tot_price']

            curraggrsec['avg_price'] = curraggrsec['tot_price'] / (curraggrsec['trades'] or 1)
            
            curraggrsec['high'] = sechigh if not prevaggrsec else max(sechigh, prevaggrsec['high'])

            curraggrsec['low'] = seclow if not prevaggrsec else min(seclow, prevaggrsec['low'])

            curraggrsec['close'] = secclose

            numaggrsec_totsec = (secfrommin + 1) / 60

            # curraggrsecfreq/prevminfreq
            # curraggrsecfreq/dayfreq
            # currsecfreq/curraggrsecfreq
            # currsecfreq/prevminfreq
            # currsecfreq/dayfreq
            curraggrsecfreq_prevminfreq = 0 if not prevmin or not prevmin['frequency'] else curraggrsec['frequency'] / prevmin['frequency']
            curraggrsecfreq_dayfreq = 0 if not stock_data['frequency'] else curraggrsec['frequency']  / stock_data['frequency']
            currsecfreq_curraggrsecfreq = 0 if not curraggrsec['frequency']  else secfreq / curraggrsec['frequency'] 
            currsecfreq_prevminfreq = 0 if not prevmin or not prevmin['frequency'] else secfreq / prevmin['frequency']
            currsecfreq_dayfreq = 0 if not stock_data['frequency'] else secfreq / stock_data['frequency']

            # curraggrsecavgprice/prevminavgprice
            # currsecavgprice/prevminavgprice
            # currsecavgprice/curraggrsecavgprice
            curraggrsecavgprice_prevminavgprice = 0 if not prevmin or not prevmin['avg_price'] else curraggrsec['avg_price'] / prevmin['avg_price']
            currsecavgprice_prevminavgprice = 0 if not prevmin or not prevmin['avg_price'] else secavgprice / prevmin['avg_price']
            currsecavgprice_curraggrsecavgprice = 0 if not curraggrsec['avg_price']else secavgprice / curraggrsec['avg_price']

            # curraggrsechigh/prevminhigh
            # curraggrseclow/prevminlow
            # curraggrsechigh/dayhigh
            # curraggrseclow/daylow
            # currsechigh/prevminhigh
            # currseclow/prevminlow
            # currsechigh/curraggrsechigh
            # currseclow/curraggrseclow
            # currsechigh/dayhigh
            # currseclow/daylow 
            curraggrsechigh_prevminhigh = 0 if not prevmin or not prevmin['high'] else curraggrsec['high'] / prevmin['high']
            curraggrseclow_prevminlow = 0 if not prevmin or not prevmin['low'] else curraggrsec['low'] / prevmin['low']
            curraggrsechigh_dayhigh = 0 if not stock_data['high'] else curraggrsec['high'] / stock_data['high']
            curraggrseclow_daylow = 0 if not stock_data['low'] else curraggrsec['low'] / stock_data['low']
            currsechigh_prevminhigh = 0 if not prevmin or not prevmin['high'] else sechigh / prevmin['high']
            currseclow_prevminlow = 0 if not prevmin or not prevmin['low'] else seclow / prevmin['low']
            currsechigh_curraggrsechigh = 0 if not curraggrsec['high'] else sechigh / curraggrsec['high']
            currseclow_curraggrseclow = 0 if not curraggrsec['low'] else seclow / curraggrsec['low']
            currsechigh_dayhigh = 0 if not stock_data['high'] else sechigh / stock_data['high']
            currseclow_daylow = 0 if not stock_data['low'] else seclow / stock_data['low']

            # curraggrsecavgprice/prevminhigh
            # curraggrsecavgprice/prevminlow
            # curraggrsecavgprice/dayhigh
            # curraggrsecavgprice/daylow
            # currsecavgprice/prevminhigh
            # currsecavgprice/prevminlow
            # currsecavgprice/curraggrsechigh
            # currsecavgprice/curraggrseclow
            # currsecavgprice/dayhigh
            # currsecavgprice/daylow
            curraggrsecavgprice_prevminhigh = 0 if not prevmin or not prevmin['high'] else curraggrsec['avg_price'] / prevmin['high']
            curraggrsecavgprice_prevminlow = 0 if not prevmin or not prevmin['low'] else curraggrsec['avg_price'] / prevmin['low']
            curraggrsecavgprice_dayhigh = 0 if not stock_data['high'] else curraggrsec['avg_price'] / stock_data['high']
            curraggrsecavgprice_daylow = 0 if not stock_data['low'] else curraggrsec['avg_price'] / stock_data['low']
            currsecavgprice_prevminhigh = 0 if not prevmin or not prevmin['high'] else secavgprice / prevmin['high']
            currsecavgprice_prevminlow = 0 if not prevmin or not prevmin['low'] else secavgprice / prevmin['low']
            currsecavgprice_curraggrsechigh = 0 if not curraggrsec['high'] else secavgprice / curraggrsec['high']
            currsecavgprice_curraggrseclow = 0 if not curraggrsec['low'] else secavgprice / curraggrsec['low']
            currsecavgprice_dayhigh = 0 if not stock_data['high'] else secavgprice / stock_data['high']
            currsecavgprice_daylow = 0 if not stock_data['low'] else secavgprice / stock_data['low']

            # curraggrsecvol/prevminvol
            # curraggrsecvol/dayvol
            # currsecvol/prevminvol
            # currsecvol/curraggrsecvol
            # currsecvol/dayvol
            curraggrsecvol_prevminvol = 0 if not prevmin or not prevmin['volume'] else curraggrsec['volume'] / prevmin['volume']
            curraggrsecvol_dayvol =  0 if not stock_data['volume'] else curraggrsec['volume'] / stock_data['volume']
            currsecvol_prevminvol = 0 if not prevmin or not prevmin['volume'] else secvol / prevmin['volume']
            currsecvol_curraggrsecvol = 0 if not curraggrsec['volume'] else secvol / curraggrsec['volume']
            currsecvol_dayvol = 0 if not stock_data['volume'] else secvol / stock_data['volume']

            # curraggrsecavgprice/vwap
            # currsecavgprice/vwap
            curraggrsecavgprice_vwap = 0 if not stock_data['vwap'] else curraggrsec['avg_price'] / stock_data['vwap']
            currsecavgprice_vwap = 0 if not stock_data['vwap'] else secavgprice / stock_data['vwap']

            # curraggrsechigh/vwap
            # curraggrseclow/vwap
            # currsechigh/vwap
            # currseclow/vwap
            curraggrsechigh_vwap = 0 if not stock_data['vwap'] else curraggrsec['high'] / stock_data['vwap']
            curraggrseclow_vwap = 0 if not stock_data['vwap'] else curraggrsec['low'] / stock_data['vwap']
            currsechigh_vwap = 0 if not stock_data['vwap'] else sechigh / stock_data['vwap']
            currseclow_vwap = 0 if not stock_data['vwap'] else seclow / stock_data['vwap']



            inputfornet = (

                numaggrsec_totsec,
                curraggrsecfreq_prevminfreq,
                curraggrsecfreq_dayfreq,
                currsecfreq_curraggrsecfreq,
                currsecfreq_prevminfreq,
                currsecfreq_dayfreq,
                curraggrsecavgprice_prevminavgprice,
                currsecavgprice_prevminavgprice,
                currsecavgprice_curraggrsecavgprice,
                curraggrsechigh_prevminhigh,
                curraggrseclow_prevminlow,
                curraggrsechigh_dayhigh,
                curraggrseclow_daylow,
                currsechigh_prevminhigh,
                currseclow_prevminlow,
                currsechigh_curraggrsechigh,
                currseclow_curraggrseclow,
                currsechigh_dayhigh,
                currseclow_daylow,
                curraggrsecavgprice_prevminhigh,
                curraggrsecavgprice_prevminlow,
                curraggrsecavgprice_dayhigh,
                curraggrsecavgprice_daylow,
                currsecavgprice_prevminhigh,
                currsecavgprice_prevminlow,
                currsecavgprice_curraggrsechigh,
                currsecavgprice_curraggrseclow,
                currsecavgprice_dayhigh,
                currsecavgprice_daylow,
                curraggrsecvol_prevminvol,
                curraggrsecvol_dayvol,
                currsecvol_prevminvol,
                currsecvol_curraggrsecvol,
                currsecvol_dayvol,
                curraggrsecavgprice_vwap,
                currsecavgprice_vwap,
                curraggrsechigh_vwap,
                curraggrseclow_vwap,
                currsechigh_vwap,
                currseclow_vwap
            )

            inputfornet = tuple(round(x, 4) for x in inputfornet)

            if type(tradegroup[0]) != ArtificialTrade:

                netinput.append(inputfornet)
                timelist.append(timestamp)

        interval = 60
        start_index = interval - 1

        for mindataindx in range(start_index, total_seconds, interval):

            minclose = stock_data['second_aggregates'][mindataindx]['close']

            # You can use list comprehension here to simplify the loop
            netoutput += [
                int((100 * (minclose - secaggr['close']) / secaggr['close']) >= 8)
                for secaggr in stock_data['second_aggregates'][mindataindx - start_index:mindataindx + 1]
                if secaggr['close']
            ]


        print(f'netinput len {len(netinput)}')
        print(f'netoutput len {len(netoutput)}')


        strinput = [

                'numaggrsec_totsec',
                'curraggrsecfreq_prevminfreq',
                'curraggrsecfreq_dayfreq',
                'currsecfreq_curraggrsecfreq',
                'currsecfreq_prevminfreq',
                'currsecfreq_dayfreq',
                'curraggrsecavgprice_prevminavgprice',
                'currsecavgprice_prevminavgprice',
                'currsecavgprice_curraggrsecavgprice',
                'curraggrsechigh_prevminhigh',
                'curraggrseclow_prevminlow',
                'curraggrsechigh_dayhigh',
                'curraggrseclow_daylow',
                'currsechigh_prevminhigh',
                'currseclow_prevminlow',
                'currsechigh_curraggrsechigh',
                'currseclow_curraggrseclow',
                'currsechigh_dayhigh',
                'currseclow_daylow',
                'curraggrsecavgprice_prevminhigh',
                'curraggrsecavgprice_prevminlow',
                'curraggrsecavgprice_dayhigh',
                'curraggrsecavgprice_daylow',
                'currsecavgprice_prevminhigh',
                'currsecavgprice_prevminlow',
                'currsecavgprice_curraggrsechigh',
                'currsecavgprice_curraggrseclow',
                'currsecavgprice_dayhigh',
                'currsecavgprice_daylow',
                'curraggrsecvol_prevminvol',
                'curraggrsecvol_dayvol',
                'currsecvol_prevminvol',
                'currsecvol_curraggrsecvol',
                'currsecvol_dayvol',
                'curraggrsecavgprice_vwap',
                'currsecavgprice_vwap',
                'curraggrsechigh_vwap',
                'curraggrseclow_vwap',
                'currsechigh_vwap',
                'currseclow_vwap'
        ]

        if len(netinput) != len(netoutput):

            print('Error')

        # convert your data to DataFrame
        df = pd.DataFrame(netinput, columns=strinput)

        # add 'Output' column
        df['Output'] = netoutput

        # add 'Timestamp' column and set it as index
        df['Timestamp'] = [f'{symbol}_{time_i}' for time_i in timelist]
        df.set_index('Timestamp', inplace=True)

        # write to CSV, append if file already exists
        with open('output_file.csv', 'a') as f:
            df.to_csv(f, header=f.tell()==0)


def main():

    learn()
    return
    # stream_stocks()
    # Set up a scheduler to wake up 5 minutes before the market opens
    scheduler = BackgroundScheduler()
    scheduler.add_job(stream_stocks, 'cron', hour='9', minute='27', second='00', timezone='US/Eastern')
    scheduler.start()

    # Keep the script running
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()