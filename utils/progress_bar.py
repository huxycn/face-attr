import sys
import time


TOTAL_BAR_LENGTH = 50
begin_time = time.time()


def progress_bar(batch_idx, batchs, meter_msg, end):
    global begin_time
    # 重置进度条起始时间
    if batch_idx == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH*batch_idx/batchs)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    # used: 已用时间
    used = time.time() - begin_time

    # step: 每个 batch 用时
    step = used / (batch_idx + 1)

    # eta: 预计到达时间
    eta = step * (batchs - batch_idx - 1)

    bar = []
    bar.append(' {}{}/{}'.format(' ' * (len(str(batchs)) - len(str(batch_idx + 1))), batch_idx+1, batchs))

    # 主进度条
    bar.append(' [' + '=' * (cur_len - 1) + '>' + '.' * rest_len + ']')

    bar.append('- {:.2f}s/step - used: {:.2f}s - ETA: {:.2f}s'.format(step, used, eta))

    # 损失、准确率信息
    bar.append(meter_msg)

    bar.append(end)
    bar_str = ' '.join(bar)
    sys.stdout.write(bar_str)
    sys.stdout.flush()
