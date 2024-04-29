from    meta_dataset import Metadataset
import  os.path
import  numpy as np


class SeismicNShot:
    def __init__(self, batchsz, k_shot, k_query, imgsz, dir_interpolation, dir_random_denoise,
                    dir_groundroll_denoise, dir_migration, dir_vrms):
        """
        :param batchsz: task num
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz

        self.interpolation = Metadataset(dir_interpolation)
 
        self.random_denoise = Metadataset(dir_random_denoise)

        self.groundroll_denoise = Metadataset(dir_groundroll_denoise)

        self.migration = Metadataset(dir_migration)

        self.vrms = Metadataset(dir_vrms)

        self.inputs_ip = []
        self.labels_ip = []
        for (input, label) in self.interpolation:
            self.inputs_ip.append(input)
            self.labels_ip.append(label)

        self.inputs_rd = []
        self.labels_rd = []
        for (input, label) in self.random_denoise:
            self.inputs_rd.append(input)
            self.labels_rd.append(label)

        self.inputs_gd = []
        self.labels_gd = []
        for (input, label) in self.groundroll_denoise:
            self.inputs_gd.append(input)
            self.labels_gd.append(label)

        self.inputs_mi = []
        self.labels_mi = []
        for (input, label) in self.migration:
            self.inputs_mi.append(input)
            self.labels_mi.append(label)

        self.inputs_vrms = []
        self.labels_vrms = []
        for (input, label) in self.vrms:
            self.inputs_vrms.append(input)
            self.labels_vrms.append(label)

        self.inputs_ip = np.array(self.inputs_ip).astype(np.float)
        self.labels_ip = np.array(self.labels_ip).astype(np.float)

        self.inputs_rd = np.array(self.inputs_rd).astype(np.float)
        self.labels_rd = np.array(self.labels_rd).astype(np.float)

        self.inputs_gd = np.array(self.inputs_gd).astype(np.float)
        self.labels_gd = np.array(self.labels_gd).astype(np.float)

        self.inputs_mi = np.array(self.inputs_mi).astype(np.float)
        self.labels_mi = np.array(self.labels_mi).astype(np.float)

        self.inputs_vrms = np.array(self.inputs_vrms).astype(np.float)
        self.labels_vrms = np.array(self.labels_vrms).astype(np.float)

        self.inputs_ip_train, self.inputs_ip_test = self.inputs_ip[:160], self.inputs_ip[160:]
        self.labels_ip_train, self.labels_ip_test = self.labels_ip[:160], self.labels_ip[160:]

        self.inputs_rd_train, self.inputs_rd_test = self.inputs_rd[:160], self.inputs_rd[160:]
        self.labels_rd_train, self.labels_rd_test = self.labels_rd[:160], self.labels_rd[160:]

        self.inputs_gd_train, self.inputs_gd_test = self.inputs_gd[:160], self.inputs_gd[160:]
        self.labels_gd_train, self.labels_gd_test = self.labels_gd[:160], self.labels_gd[160:]

        self.inputs_mi_train, self.inputs_mi_test = self.inputs_mi[:160], self.inputs_mi[160:]
        self.labels_mi_train, self.labels_mi_test = self.labels_mi[:160], self.labels_mi[160:]

        self.inputs_vrms_train, self.inputs_vrms_test = self.inputs_vrms[:200], self.inputs_vrms[200:]
        self.labels_vrms_train, self.labels_vrms_test = self.labels_vrms[:200], self.labels_vrms[200:]


        self.batchsz = batchsz
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <=20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}

        self.datasets_input = {"train": [self.inputs_ip_train, self.inputs_rd_train, self.inputs_gd_train, 
                                         self.inputs_mi_train, self.inputs_vrms_train], 
                               "test": [self.inputs_ip_test, self.inputs_rd_test, self.inputs_gd_test, 
                                        self.inputs_mi_test, self.inputs_vrms_test]}  # original data cached
        self.datasets_label = {"train": [self.labels_ip_train, self.labels_rd_train, self.labels_gd_train, 
                                         self.labels_mi_train, self.labels_vrms_train], 
                               "test": [self.labels_ip_test, self.labels_rd_test, self.labels_gd_test, 
                                        self.labels_mi_test, self.labels_vrms_test]}  # original data cached

        self.datasets_cache = {"train": self.load_data_cache(self.datasets_input["train"],self.datasets_label["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets_input["test"],self.datasets_label["test"])}

    def load_data_cache(self, data_input_pack, data_label_pack):
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []

                selected = np.random.choice(data_input_pack[i].shape[0], self.k_shot + self.k_query, False)

                # meta-training and meta-test
                x_spt.append(data_input_pack[i][selected[:self.k_shot]])
                x_qry.append(data_input_pack[i][selected[self.k_shot:]])
                y_spt.append(data_label_pack[i][selected[:self.k_shot]])
                y_qry.append(data_label_pack[i][selected[self.k_shot:]])

                # shuffle inside a batch
                perm = np.random.permutation(self.k_shot)
                x_spt = np.array(x_spt).reshape(self.k_shot, 1, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.k_shot, 1, self.resize, self.resize)[perm]
                perm = np.random.permutation(self.k_query)
                x_qry = np.array(x_qry).reshape(self.k_query, 1, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.k_query, 1, self.resize, self.resize)[perm]

                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # [b, setsz, 1, 256, 256]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, self.k_shot, 1, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.float32).reshape(self.batchsz, self.k_shot, 1, self.resize, self.resize)
            # [b, qrysz, 1, 256, 256]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz,  self.k_query, 1, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.float32).reshape(self.batchsz,  self.k_query, 1, self.resize, self.resize)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets_input[mode],self.datasets_label[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

