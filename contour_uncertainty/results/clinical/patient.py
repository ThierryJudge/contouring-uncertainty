import multiprocessing

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from contour_uncertainty.results.clinical.utils import aleatoric_epistemic_uncertainty
from vital.data.camus.config import Label, View, Instant
from vital.metrics.evaluate.clinical.heart_us import compute_left_ventricle_volumes
from vital.utils.format.native import prefix

from contour_uncertainty.data.config import BatchResult
from contour_uncertainty.utils.clinical import metric_error


class PatientMetric:
    PREFIX: str = ""

    def __init__(self):
        self.n_jobs = 8

    def compute(self, a2c_result: BatchResult, a4c_result: BatchResult):
        raise NotImplementedError

    def __call__(self, results: dict[str, dict]):

        if self.n_jobs == 0:
            res = {patient_id: prefix(self.compute(patient_results), self.PREFIX)
                   for patient_id, patient_results in results.items() if
                   ('4CH' in patient_results.keys() and '2CH' in patient_results.keys()) or 
                   ('4C' in patient_results.keys() and '2C' in patient_results.keys())}
        else:
            n_jobs = multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
            pool = multiprocessing.Pool(processes=n_jobs)

            inputs = [patient_results for patient_id, patient_results in results.items()
                      if  ('4CH' in patient_results.keys() and '2CH' in patient_results.keys()) or 
                   ('4C' in patient_results.keys() and '2C' in patient_results.keys())]

            keys = [patient_id for patient_id, patient_results in results.items()
                      if  ('4CH' in patient_results.keys() and '2CH' in patient_results.keys()) or 
                    ('4C' in patient_results.keys() and '2C' in patient_results.keys())]

            vals = pool.imap(self.compute, inputs)

            res = {keys[i]: val for i, val in enumerate(vals)}

        # res = {patient_id: prefix(self.compute(patient_results[View.A2C], patient_results[View.A4C]), self.PREFIX)
        #        for patient_id, patient_results in results.items() if
        #        '4CH' in patient_results.keys() and '2CH' in patient_results.keys()}

        # def call(patient_id, patient_reults):
        #     if '4CH' in patient_results.keys() and '2CH' in patient_results.keys():
        #         a2c = patient_results[View.A2C]
        #         a4c = patient_results[View.A4C]
        #         return prefix(self.compute(a2c, a4c), self.PREFIX)
        #
        # res = {}
        # for patient_id, patient_results in results.items():
        #     if '4CH' in patient_results.keys() and '2CH' in patient_results.keys():
        #         a2c = patient_results[View.A2C]
        #         a4c = patient_results[View.A4C]
        #         res[patient_id] = prefix(self.compute(a2c, a4c), self.PREFIX)

        return pd.DataFrame(res).T


class Volume(PatientMetric):
    # TODO SPLIT INTO ES_VOLUME, ED_VOLUME and EF
    PREFIX = ''
    MIN_VALUE = 0
    MAX_VALUE = 1

    # def compute(self, a2c: BatchResult, a4c: BatchResult):
    def compute(self, patient_results: dict):
        
        try: 
            a2c = patient_results['2CH']
            a4c = patient_results['4CH']
        except: 
            a2c = patient_results['2C']
            a4c = patient_results['4C']
        
        a2c_instants = a2c.instants
        a4c_instants = a4c.instants


        a2c_ED = a2c_instants[Instant.ED]
        a2c_ES = a2c_instants[Instant.ES]
        a4c_ED = a4c_instants[Instant.ED]
        a4c_ES = a4c_instants[Instant.ES]

        Te = a2c.pred_samples.shape[1]
        Ta = a2c.pred_samples.shape[2]

        T = a2c.pred_samples.shape[1]
        assert T == a2c.pred_samples.shape[1]

        # gt_lv_edv, gt_lv_esv = compute_left_ventricle_volumes(
        #     np.isin(a2c.gt_full_res[a2c_instants[Instant.ED]], Label.LV),
        #     np.isin(a2c.gt_full_res[a2c_instants[Instant.ES]], Label.LV),
        #     a2c.voxelspacing,
        #     np.isin(a4c.gt_full_res[a4c_instants[Instant.ED]], Label.LV),
        #     np.isin(a4c.gt_full_res[a4c_instants[Instant.ES]], Label.LV),
        #     a4c.voxelspacing)
        # gt_lv_ef = (gt_lv_edv - gt_lv_esv) / gt_lv_edv
        # print(gt_lv_edv, gt_lv_esv, gt_lv_ef)
        # print("VOXEL", a2c.voxelspacing, a4c.voxelspacing)

        gt_lv_edv, gt_lv_esv = compute_left_ventricle_volumes(
            np.isin(a2c.gt[a2c_instants[Instant.ED]], Label.LV),
            np.isin(a2c.gt[a2c_instants[Instant.ES]], Label.LV),
            a2c.voxelspacing,
            np.isin(a4c.gt[a4c_instants[Instant.ED]], Label.LV),
            np.isin(a4c.gt[a4c_instants[Instant.ES]], Label.LV),
            a4c.voxelspacing)
        
        # print(gt_lv_edv)
        # print(gt_lv_esv)
        # from matplotlib import pyplot as plt 
        # f, ax = plt.subplots(2,2)
        # ax[0,0].imshow(np.isin(a2c.gt[a2c_instants[Instant.ED]], Label.LV))
        # ax[0,1].imshow(np.isin(a2c.gt[a2c_instants[Instant.ES]], Label.LV))
        # ax[1,0].imshow(a4c.gt[a4c_instants[Instant.ED]])
        # ax[1,1].imshow(a4c.gt[a4c_instants[Instant.ES]])
        # plt.savefig('ERROR.png', dpi=300, bbox_inches='tight')
        # exit()
        
        try: 
            gt_lv_ef = (gt_lv_edv - gt_lv_esv) / gt_lv_edv
        except: 
            print(gt_lv_edv)
            print(gt_lv_esv)
            from matplotlib import pyplot as plt 
            f, ax = plt.subplots(2,2)
            ax[0,0].imshow(np.isin(a2c.gt[a2c_instants[Instant.ED]], Label.LV))
            ax[0,1].imshow(np.isin(a2c.gt[a2c_instants[Instant.ES]], Label.LV))
            ax[1,0].imshow(a4c.gt[a4c_instants[Instant.ED]])
            ax[1,1].imshow(a4c.gt[a4c_instants[Instant.ES]])
            plt.savefig('ERROR.png', dpi=300, bbox_inches='tight')
            exit()

        pred_lv_edv, pred_lv_esv = compute_left_ventricle_volumes(
            np.isin(a2c.pred[a2c_instants[Instant.ED]], Label.LV),
            np.isin(a2c.pred[a2c_instants[Instant.ES]], Label.LV),
            a2c.voxelspacing,
            np.isin(a4c.pred[a4c_instants[Instant.ED]], Label.LV),
            np.isin(a4c.pred[a4c_instants[Instant.ES]], Label.LV),
            a4c.voxelspacing)
        pred_lv_ef = (pred_lv_edv - pred_lv_esv) / pred_lv_edv

        # mc_lv_edv, mc_lv_esv, mc_lv_ef = [], [], []

        mc_lv_edv, mc_lv_esv, mc_lv_ef = np.zeros((Te, Ta)), np.zeros((Te, Ta)), np.zeros((Te, Ta))

        for i in range(Te):
            for j in range(Ta):
                try:
                    lv_edv, lv_esv = compute_left_ventricle_volumes(
                        np.isin(a2c.pred_samples[a2c_ED, i, j], Label.LV),
                        np.isin(a2c.pred_samples[a2c_ES, i, j], Label.LV),
                        a2c.voxelspacing,
                        np.isin(a4c.pred_samples[a4c_ED, i, j], Label.LV),
                        np.isin(a4c.pred_samples[a4c_ES, i, j], Label.LV),
                        a4c.voxelspacing)

                    mc_lv_edv[i, j] = lv_edv
                    mc_lv_esv[i, j] = lv_esv
                    mc_lv_ef[i, j] = (lv_edv - lv_esv) / lv_edv
                except:
                    pass

        sample_reject = np.logical_or(mc_lv_ef < self.MIN_VALUE, mc_lv_ef > self.MAX_VALUE)
        mc_lv_ef[sample_reject] = np.nan

        lv_edv_metric_mean, lv_edv_al_var, lv_edv_ep_var, lv_edv_metric_var = aleatoric_epistemic_uncertainty(mc_lv_edv)
        lv_esv_metric_mean, lv_esv_al_var, lv_esv_ep_var, lv_esv_metric_var = aleatoric_epistemic_uncertainty(mc_lv_esv)
        lv_ef_metric_mean, lv_ef_al_var, lv_ef_ep_var, lv_ef_metric_var = aleatoric_epistemic_uncertainty(mc_lv_ef)


        reject = not (self.MIN_VALUE < lv_ef_metric_mean <= self.MAX_VALUE)
        if np.sum(sample_reject) / np.size(sample_reject) > 0.5:
            reject = True
        # print('reject', reject, np.sum(sample_reject) / np.size(sample_reject))

        # ef_error = metric_error(pred_lv_ef, gt_lv_ef)
        # edv_error = metric_error(pred_lv_edv, gt_lv_edv)
        # esv_error = metric_error(pred_lv_esv, gt_lv_esv)

        ef_error = metric_error(lv_ef_metric_mean, gt_lv_ef)
        edv_error = metric_error(lv_edv_metric_mean, gt_lv_edv)
        esv_error = metric_error(lv_esv_metric_mean, gt_lv_esv)

        # reject = False
        return {
            # EF
            'EF_pred': pred_lv_ef,
            'EF_gt': gt_lv_ef,
            'EF_error': ef_error,
            'EF_mc': mc_lv_ef,
            'EF_std': lv_ef_metric_var,
            'EF_mean': lv_ef_metric_mean,
            'EF_aleatoric_std': lv_ef_al_var,
            'EF_epistemic_std': lv_ef_ep_var,
            'EF_reject': reject,
            'EF_sample_reject': sample_reject,
            # EDV
            'EDV_pred': pred_lv_edv,
            'EDV_gt': gt_lv_edv,
            'EDV_error': edv_error,
            'EDV_mc': mc_lv_edv,
            'EDV_std': lv_edv_metric_var,
            'EDV_mean': lv_edv_metric_mean,
            'EDV_aleatoric_std': lv_edv_al_var,
            'EDV_epistemic_std': lv_edv_ep_var,
            # ESV
            'ESV_pred': pred_lv_esv,
            'ESV_gt': gt_lv_esv,
            'ESV_error': esv_error,
            'ESV_mc': mc_lv_esv,
            'ESV_std': lv_esv_metric_var,
            'ESV_mean': lv_esv_metric_mean,
            'ESV_aleatoric_std': lv_esv_al_var,
            'ESV_epistemic_std': lv_esv_ep_var,
        }
