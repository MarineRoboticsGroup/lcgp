from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

eps = 1E-3





if __name__ == '__main__':

    path_name = './'
    file_names = [path_name+f for f in listdir(path_name) if isfile(join(path_name, f)) and "relativematrix" in f and "absolutenoise" in f]
    noise_levels = [0, 0.2, 0.4, 0.6, 0.8, 1]
    eig_bins = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, 1), (1, 100)]

    for file in file_names:
        print(file)
        df = pd.read_excel(file, sheet_name="raw_data", index_col=0)
        for noise in noise_levels:
            noise_level_df = df.loc[abs(df['Noise Level (std dev)'] - noise) < eps]

            for eig_bin in eig_bins:
                eig_min, eig_max = eig_bin
                eig_bin_df = noise_level_df.loc[noise_level_df['Min Eigval'] < eig_max ]
                eig_bin_df = eig_bin_df.loc[eig_bin_df['Min Eigval'] > eig_min]

                print(eig_bin)
                print(eig_bin_df)
                eig_bin_df['Max Error'].plot.kde()
                eig_bin_df['Mean Error'].plot.kde()
                plt.xlabel('Error (meters)')
                plt.legend(['Max Error', 'Avg Error'])
                plt.title("%.1f < Eigval < %.1f ; Noise: %.1f"%(eig_min, eig_max, noise))
                plt.ylim((0, 0.7))
                plt.xlim(0, 20)
                plt.show()
                plt.close()

            # new_df['Min Eigval'].plot.kde()
            # new_df.plot.scatter('Min Eigval', 'Max Error')
            # plt.show()
            # new_df.plot.scatter('Min Eigval', 'Mean Error')
            # plt.show()



    # nproc = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(nproc-2)
    # for f_path in files:
    #     MakeSettingPlots(f_path, snl_approaches)
        # pool.apply_async(MakeSettingPlots, args = (f_path, snl_approaches))
    # pool.close()
    # pool.join()



