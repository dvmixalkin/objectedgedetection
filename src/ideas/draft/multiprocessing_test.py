import multiprocessing
from itertools import product
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

def merge_names(a, b):
    return '{} & {}'.format(a, b)






if __name__ == '__main__':
    names = ['Brown', 'Wilson', 'Bartlett', 'Rivera', 'Molloy', 'Opie']
    with Pool(processes=4) as p:
        results = list(
            tqdm(
                p.imap_unordered(
                    partial(merge_names, b=names), range(5, 20)
                ), total=len(range(15))
            )
        )

    # with multiprocessing.Pool(processes=3) as pool:
    #     results = pool.starmap(merge_names, product(names, repeat=2))
    print(results)
