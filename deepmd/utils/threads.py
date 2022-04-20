import threading


class newThread (threading.Thread):
    def __init__(self,
                 threadID,
                 func,
                 arg_dict):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.func = func
        self.arg_dict = arg_dict
        self.result = {}

    def run(self):
        self.result = self.func(self.arg_dict)

    def get_result(self):
        return self.result


def para_thread(func,
                arg_dict_full: dict,
                arg_dict_flag: dict,
                thread_num: int = 32,
                result_flag: dict = None):
    thread_pool = []
    result_full = {}
    flag_len = None
    for ii in range(thread_num):
        arg_dict = {}
        for k in arg_dict_full.keys():
            if k in arg_dict_flag.keys() and arg_dict_flag[k] and isinstance(arg_dict_full[k], list):
                temp_len = len(arg_dict_full[k])
                if flag_len is None:
                    flag_len = temp_len
                else:
                    assert(flag_len == temp_len)
                per_num = int(temp_len / thread_num)
                if ii != thread_num - 1:
                    arg_dict[k] = arg_dict_full[k][ii * per_num: (ii+1) * per_num]
                else:
                    arg_dict[k] = arg_dict_full[k][ii * per_num:]
            else:
                arg_dict[k] = arg_dict_full[k]
        thread_pool.append(newThread(ii, func=func, arg_dict=arg_dict))
    for tr in thread_pool:
        tr.start()
    for tr in thread_pool:
        tr.join()
    for tr in thread_pool:
        result_thr = tr.get_result()
        for k in result_thr.keys():
            if k not in result_full.keys():
                result_full[k] = []
            if isinstance(result_thr[k], list):
                if result_flag is not None and k in result_flag.keys() and not result_flag[k]:
                    result_full[k].append(result_thr[k])
                else:
                    result_full[k] += result_thr[k]
            else:
                result_full[k].append(result_thr[k])
    return result_full

# ## An example for multi-thread
# def example_func(arg_dict):
#     ## do some single-thread work
#     result = {'keys': 'value'}
#     return result
#
#
# example_arg_dict = {'input': 'input'}
# example_arg_dict_flag = {'input': True}
# example_result = para_thread(example_func, example_arg_dict, example_arg_dict_flag)
# ## process the output
