import derate_matching as drm
import numpy as np

np.set_printoptions(threshold=np.nan, suppress=True, precision=10, linewidth=np.nan)

DEBUG_MODE = True

if DEBUG_MODE:
    q_value = np.array([1, 2, 4, 6, 8])
    for set_num in range(1000, 2000):
        for ncb_f in range(500, 2500):
            for q in q_value:
                DRM = drm.DeRateMatch(q=q, set_num=set_num, ncb_f=ncb_f, k_f=7, f=33, k_p=302, zc=77)
                result = DRM.test_addr_segment()
                if not result:
                    exit(-1)

        if set_num % 10 == 0:
            print(set_num)
else:
    DRM = drm.DeRateMatch(q=6, set_num=1024, ncb_f=500, k_f=103, f=33, k_p=7, zc=77, print_on=False)

    test_row = DRM.disp_data_in_cyc(DRM.bram_addr_row)
    test_col = DRM.disp_data_in_cyc(DRM.bram_addr_col)
    asdf = DRM.disp_data_in_cyc(DRM.addr)
    # print(np.vstack((test_row, test_col, asdf)))
    # print(test_row)

    DRM.data_segment(print_on=not DEBUG_MODE)
