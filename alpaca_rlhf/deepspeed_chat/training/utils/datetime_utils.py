import time
from datetime import datetime
from datetime import timedelta


def now(date_format='%Y%m%d_%H%M%S'):
    """

    :param date_format:
    :return:
    """
    result = datetime.now()
    result = result.strftime(date_format)
    return result


if __name__ == '__main__':
    print(now())
