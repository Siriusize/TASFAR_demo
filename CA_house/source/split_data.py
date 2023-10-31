DOMAIN_INDEX_DICT = {'rich': 0, 'poor': 1}


def split_fun(longitude, latitude):
    if longitude <= -120:
        y = - longitude * 6 / 5 - 108
        if y >= latitude:
            domain_index = DOMAIN_INDEX_DICT['rich']
        else:
            domain_index = DOMAIN_INDEX_DICT['poor']
    else:
        y = - longitude * 6 / 8 - 54
        if y >= latitude:
            domain_index = DOMAIN_INDEX_DICT['rich']
        else:
            domain_index = DOMAIN_INDEX_DICT['poor']
    return domain_index
