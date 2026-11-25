from experiment_common.timestral_csv.evaluation import MaskedMixed


class SensorMetrics(MaskedMixed):
    def report(self):
        if self.states:
            result = super().report()
            result.setdefault('continuous_run_macro_scaled_mae', None)
            return result
        result = self.all.report()
        result['continuous_run_macro_scaled_mae'] = result['run_macro_scaled_mae']
        result['binary'] = dict(sensor_count=0, sensor_macro_f1=None)
        return result
