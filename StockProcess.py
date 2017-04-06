

import datetime
import time
import numpy as np
import tushare as ts
import pandas

from errors import *

from Config import *

class Stock(object):
    def __init__(self, code='sh', pred_day = '2017-02-21', period = 50, tick_period = 5):
        self.code = code
        self.pred_day = pred_day
        self.period = period
        self.tick_period = tick_period
        self.init()

    def init(self):
        self.start = self._startday()
        self.end = self._endday()

    def _startday(self):
        pred_day = datetime.datetime.strptime(self.pred_day, '%Y-%m-%d')
        if pred_day.weekday() == 5:
            pred_day += datetime.timedelta(2)
        if pred_day.weekday() == 6:
            pred_day += datetime.timedelta(1)
        count = 0
        curr = pred_day - datetime.timedelta(1)
        while count <= self.period:
            if curr.weekday() != 5 and curr.weekday() != 6:
                count += 1
                curr -= datetime.timedelta(1)
            else:
                curr -= datetime.timedelta(1)
        return curr.strftime('%Y-%m-%d')

    def _endday(self):
        pred_day = datetime.datetime.strptime(self.pred_day, '%Y-%m-%d')
        if pred_day.weekday() == 5:
            pred_day += datetime.timedelta(2)
        if pred_day.weekday() == 6:
            pred_day += datetime.timedelta(1)
        return (pred_day - datetime.timedelta(1)).strftime('%Y-%m-%d')

    def _day_split(self):
        interval = []
        curr = datetime.datetime.strptime(self.start, '%Y-%m-%d')
        end = datetime.datetime.strptime(self.end, '%Y-%m-%d')
        while curr <= end:
            interval.append((curr.strftime('%Y-%m-%d'), (curr + datetime.timedelta(7)).strftime('%Y-%m-%d')))
            curr += datetime.timedelta(8)
            curr.strftime('%Y-%m-%d')
        return interval

    def get_k_history(self, ktype='D'):
        result = []
        # result = ts.get_k_data(self.code, start=self.start, end=self.end, ktype=ktype)
        # return result
        intervals = self._day_split()
        for interval in intervals:
            tmp = ts.get_hist_data(self.code, start=interval[0], end=interval[1], ktype = ktype)
            result.append(tmp)
        def _merge(df1, df2):
            return df1.append(df2)
        result = reduce(_merge,result)
        return result

    def _get_tick_history(self, day):
        return ts.get_tick_data(self.code, date=day)

    @staticmethod
    def workday_minus(day):
        if isinstance(day, str):
            day = datetime.datetime.strptime(day, '%Y-%m-d')
        while day.weekday() == 5 or day.weekday() == 6:
            day -= datetime.timedelta(1)
        return day.strftime('%Y-%m-%d')

    def get_tick_history(self):
        tick_history = pandas.DataFrame({'code':[]})
        curr = datetime.datetime.strptime(self.pred_day, '%Y-%m-%d') - datetime.timedelta(1)
        for i in range(self.tick_period):
            day = self.workday_minus(curr)
            tick_record = self._get_tick_history(day)
            curr = day
            if tick_history.empty:
                tick_history = tick_record
            else:
                tick_history = tick_history.append(tick_record)
        return tick_history

    def get_today(self):
        tmp = ts.get_today_all()
        return tmp[tmp['code'] == self.code]['high']

    def save(self, df, df_name):
        df.to_json(df_name, orient='records', force_ascii=False)

    def read(self, df_name):
        return pandas.read_json(df_name, orient='records')

    def gen_label(self,hist_data,  level = '5min', target = 'high', min_change_ratio = 0.01):
        if datetime.datetime.strptime(self.pred_day, '%Y-%m-%d') >= datetime.datetime.today():
            raise TimeError
        try:
            result = []
            pred_day = datetime.datetime.strptime(self.pred_day, '%Y-%m-%d')
            # before_day = self.workday_minus(pred_day - datetime.timedelta(1))

            tmp = hist_data.loc[:, target]
            tmp = tmp.values.tolist()
            last_value = tmp[SequenceLen-2]
            for curr in tmp[SequenceLen - 1 :]:
                change_ratio = (curr - last_value) / float(last_value)
                print change_ratio
                if change_ratio >= min_change_ratio:
                    result.append(1)
                elif change_ratio <= -min_change_ratio:
                    result.append(-1)
                else:
                    result.append(0)
            return result
        except:
            raise EmptyError

    def gen_target(self, target='high'):
        if datetime.datetime.strptime(self.pred_day, '%Y-%m-%d') >= datetime.datetime.today():
            raise TimeError
        try:
            if target == 'high':
                return float(ts.get_k_data(self.code, start=self.pred_day, end=self.pred_day)['high'])
            if target == 'close':
                return float(ts.get_k_data(self.code, start=self.pred_day, end=self.pred_day)['close'])
            if target == 'open':
                return float(ts.get_k_data(self.code, start=self.pred_day, end=self.pred_day)['open'])
            if target == 'low':
                return float(ts.get_k_data(self.code, start=self.pred_day, end=self.pred_day)['low'])
        except:
            raise EmptyError

    def gen_series(self, is_merge = False, is_label = True):
        train_data = [[], 0]

        hist_data = self.get_k_history('15')
        # print hist_data
        open_series = list(hist_data['open'])
        close_series = list(hist_data['close'])
        high_series = list(hist_data['high'])
        low_series = list(hist_data['low'])
        volume = list(hist_data['volume'])

        if is_merge:
            train_data[0].extend(open_series)
            train_data[0].extend(close_series)
            train_data[0].extend(high_series)
            train_data[0].extend(low_series)
            train_data[0].extend(volume)
        else:
            train_data[0].append(open_series)
            train_data[0].append(close_series)
            train_data[0].append(high_series)
            train_data[0].append(low_series)
            train_data[0].append(volume)

        if is_label:
            train_data[1] = self.gen_label(hist_data, min_change_ratio=0.02)
        else:
            train_data[1] = self.gen_target(hist_data)
        train_data = np.asarray(train_data)
        print train_data[0]
        print len(train_data[0])
        print len(train_data[1])
        return train_data


example_codes = ['000001', '000009', '000024', '000027', '000039', '000060', '000061', '000063']
example_codes_1 = ['601857']

def gen_train_data(codes_list=example_codes):
    # hs300 = ts.get_hs300s()
    # hs_300_codes = list(hs300['code'])
    train_data = [[], []]
    for code in example_codes_1:
        stock = Stock(code)
        try:
            sample, target = stock.gen_series()
            sample = np.asarray(sample)
            sample = np.transpose(sample)
            sample = np.reshape(sample, (sample.shape[0], sample.shape[1]))
            train_data[0].append(sample)
            train_data[1].append(target)
        except EmptyError:
            continue
    X_train, Y_train = np.asarray(train_data[0]), np.asarray(train_data[1])

    return X_train, Y_train

if __name__ == '__main__':
    gen_train_data()
    # print stock.get_tick_history()
    #print stock.get_today()
    #print stock.get_k_history('2017-02-01', '2017-02-02', '60')