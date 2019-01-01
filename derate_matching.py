import numpy as np
np.set_printoptions(threshold=np.nan, suppress=True, precision=10, linewidth=np.nan)


class DeRateMatch:
    def __init__(self, q, set_num, ncb_f, k_f, f, k_p, zc, print_on=False):
        # fixed parameters: bandwidth should not be smaller than 64 bit
        self.bandwidth = 64

        # input parameters
        self.q = q
        self.set_num = set_num
        self.ncb_f = ncb_f
        self.k_f = k_f
        self.f = f
        self.k_p = k_p
        self.zc = zc

        # derived parameters
        self.two_zc = 2 * zc
        self.e = set_num * q
        self.n = ncb_f + self.two_zc + f + 20
        self.ncb = ncb_f + self.two_zc

        # initialize address
        self.addr, self.bram_addr_row, self.bram_addr_col = self.__init_address__()

        # print important data
        if print_on:
            print('e', self.e)
            print('q', self.q)
            print('e/q', self.set_num)
            print('ncb_f', self.ncb_f)
            print('ncb', self.ncb)
            print('n', self.n)
            print('2zc', self.two_zc)
            print('')

    def __init_address__(self):
        addr = np.arange(self.e, dtype=np.int32)
        addr = addr.reshape(self.q, self.set_num)
        addr = (addr + self.k_f) % self.ncb_f
        # addr = (addr + self.k_f) % self.ncb_f + self.two_zc
        bram_addr_row = addr // self.bandwidth
        bram_addr_col = addr % self.bandwidth
        # if k_p < ncb_f+2*zc:
        #     addr[addr > k_p] += f
        return addr, bram_addr_row, bram_addr_col

    def disp_data_in_cyc(self, in_addr):
        """
        Display data in cycle by cycle
        :param in_addr: self.addr
        :return:
        """
        ceiled_col_diff = int(self.bandwidth * np.ceil(self.set_num / self.bandwidth)) - self.set_num
        ceiled_array = np.zeros((self.q, ceiled_col_diff), dtype=np.int32) - 1
        test = np.concatenate((in_addr, ceiled_array), axis=1)

        test_all = []
        for idx in range(0, test.shape[1], self.bandwidth):
            test_all.append(test[:, idx:idx + self.bandwidth])
        return np.vstack(test_all)

    def data_segment(self, print_on=False):
        """
        addr_pred block: predict the address situation placed in the BRAM
        :return: (length0, length1, length3, lengthr)
        """
        # parameters initialization
        max_cyc_row = self.q
        max_cyc_col = int(np.ceil(self.set_num / self.bandwidth))
        length_total_min = self.set_num % self.bandwidth
        if length_total_min == 0:  # If set_num could fully divide bandwidth, then length_total_min should = bandwidth
            length_total_min = self.bandwidth
        length_all = []

        # get the testing data (result) -----------------------------------------------------
        for cyc_col_idx in range(max_cyc_col):
            for cyc_row_idx in range(max_cyc_row):
                # local parameters initialization
                length0 = 0
                length1 = 0
                lengthr = 0
                length_total = 0
                if cyc_col_idx == max_cyc_col - 1:
                    length_total = length_total_min
                else:
                    length_total = self.bandwidth

                # calculating the head number of this cycle's 64 byte data
                head_num = cyc_col_idx * self.bandwidth + self.k_f + self.set_num * cyc_row_idx
                head_num = head_num % self.ncb_f

                # calculate length0 (assert = fpga acceleration method)
                # length0 happens because BRAM reaches the end of the 64 byte
                length0 = self.bandwidth - head_num % self.bandwidth
                # assert(length0 == np.ceil(head_num/self.bandwidth)*self.bandwidth - head_num)
                if length0 >= length_total:
                    length0 = 0

                # calculate length1
                # length1 happens because BRAM reaches its end, which is self.ncb_f
                # length0 cannot happen after length1
                length1 = self.ncb_f - head_num
                if length1 <= length0:
                    length0 = 0
                elif length1 >= length_total:
                    length1 = 0
                else:
                    length1 = length1 - length0

                # calculate lengthr
                lengthr = length_total - length0 - length1
                assert (lengthr >= 0)

                # append the data to the output
                length_all.append((length0, length1, lengthr))

        # get the golden data (test) ----------------------------------------------------------
        if print_on:
            # prepare the data
            length_test = np.vstack(length_all)
            bram_addr_row_golden_reshaped = self.disp_data_in_cyc(self.bram_addr_row)
            bram_addr_col_golden_reshaped = self.disp_data_in_cyc(self.bram_addr_col)
            addr_reshaped = self.disp_data_in_cyc(self.addr)

            # loop start
            print('index, length_golden, length_test')
            for i in range(bram_addr_row_golden_reshaped.shape[0]):
                num, ind, cnt = np.unique(bram_addr_row_golden_reshaped[i], return_index=True, return_counts=True)
                if num[num == -1].any():
                    ind = ind[1:]
                    cnt = cnt[1:]
                # sorting
                order = np.argsort(ind)
                length_golden_row = cnt[order]
                length_test_row = length_test[i]
                print(i, '\t', length_golden_row, '\t', length_test_row)
                compare = (length_golden_row != length_test_row[length_test_row!=0])
                if np.any(compare):
                    print('bram_addr_row: ', bram_addr_row_golden_reshaped[i])
                    print('bram_addr_col: ', bram_addr_col_golden_reshaped[i])
                    print('addr: ', addr_reshaped[i])
                    print('The previous two rows do not match')

        # return --------------------------------------------------------------------------------
        return np.vstack(length_all)

    def test_addr_segment(self):
        """
        The testing block to  the addr_pred_block.
        Includes the generating of the golden data , output of the addr_pred_block data, and compares to the
        addr_pred_block output data
        :return: True - pass, False - not pass
        """
        # get the gloden data
        bram_addr_row_golden = self.disp_data_in_cyc(self.bram_addr_row)
        length_golden = np.empty(0, dtype=np.int32)
        for each_row in bram_addr_row_golden:
            num, ind, cnt = np.unique(each_row, return_index=True, return_counts=True)
            if num[num == -1].any():
                ind = ind[1:]
                cnt = cnt[1:]
            # sorting
            order = np.argsort(ind)
            sorted_cnt = cnt[order]
            sorted_cnt_nonzero = sorted_cnt[sorted_cnt != 0]
            length_golden = np.append(length_golden, sorted_cnt_nonzero)

        # get the testing data
        length_test = self.data_segment()
        length_test = length_test[length_test != 0]

        # compare length_golden and length_test
        compare = (length_golden != length_test)
        if np.any(compare):
            print('ERROR: length_golden does not match with length_test')
            print('q', self.q)
            print('set_num', self.set_num)
            print('ncb_f', self.ncb_f)
            print('zc', self.zc)
            print('')
            return False
        else:
            return True





